# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from var_mapper import VarMapper
from boolean_whf import Clause, ClauseProcessor, Clauses, ClauseSignature, AFSAT_DFTCache, Objective, clause_type_ids
from utils import LogThrottle

logger = logging.getLogger(__name__)
INT64_MAX = int(np.iinfo(np.int64).max)


class UnsatError(Exception):
    pass


class PBSATFormula(object):
    """Parse and process SAT formulas from DIMACS and hybrid-constraint files.

    Supports CNF, XOR, NAE, AMO, EO, EK, and CARD constraints, plus optional
    prefix files. A PBSATFormula instance is single-use and tied to its initial input problem

    Compactification is on by default to reduce the size of assignment vectors, and a variable ID map is constructed
    to recover original variable IDs when required.

    Attributes:
        clause_sets (dict[ClauseSignature, Clauses]): Clause groups keyed by signature (type, length, cardinality (opt))
        unit_prefix (set[int]): Unit literals derived while parsing constraints.
        n_var (int): Active variable count after loading.
        n_clause (int): Loaded clause count.
        workers (int): Worker threads for clause processing.
        n_devices (int): Device count passed to clause processors.
        disk_cache (AFSAT_DFTCache | None): Optional cache for compiled artifacts.
        compactify (bool): Whether to remap input literals to dense internal IDs.

    Examples:
        >>> formula = PBSATFormula(workers=4, n_devices=1, compactify=False)
        >>> formula.read_DIMACS("problem.cnf")
        >>> objectives = formula.process_clauses_to_array()
        >>> prefixes = formula.process_prefix("prefix.txt")
    """

    def __init__(
        self,
        workers: int = 1,
        n_devices: int = 1,
        disk_cache: str = "",
        file: str = "",
        compactify: bool = True,
    ) -> None:
        self.clause_sets: dict[ClauseSignature, Clauses] = {}
        self.n_devices: int = n_devices
        self.workers: int = workers
        self.compactify: bool = compactify
        self.unit_prefix: set[int] = set()
        self.var_mapper: VarMapper = VarMapper()
        self.disk_cache: AFSAT_DFTCache | None = None
        self.n_var: int = 0
        self.n_clause: int = 0
        self.seen_max_var: int = 0
        self.seen_clauses: int = 0
        self._loaded_file: str | None = None
        if disk_cache:
            self.disk_cache = AFSAT_DFTCache(disk_cache)

        if file:
            self.read_DIMACS(file)

    def read_DIMACS(self, dimacs_file: str) -> None:
        """
        Parse SAT problem instances from files in DIMACS format or a hybrid format documented below.
        We assume consistent formatting after the first clause. Mixed format files will not parse correctly.
        This method supports multiple input formats:

        Standard DIMACS CNF -
         - Lines beginning with 'c' or '*' are treated as comments
         - Problem line 'p cnf <vars> <clauses>' specifies the number of variables and clauses
         - Each clause is a space-separated list of integers ending with 0
         - Example: "1 -2 3 0" represents ($x_1 \vee \neg x_2 \vee x_3$)

        Hybrid DIMACS format -
         - As per above with the following changes:
         - All constraint lines either start with "h" (n.b. "h x" ONLY can also be "1 x") OR the clause type e.g. below.
         - The "h" must be present on all constraints if it is present for any of them.
         Then the following applies:
         - Each constraint begins with a letter (excluding CNF) specifying its type:
           * '' : Standard CNF clause (e.g. "h 1 2 0" or "1 2 0")
           * 'x[or]': XOR (xor) constraint (e.g. "h x .. 0" or "x .. 0", or "xor .. 0", or "1 x .. 0")
           * 'n[ae]': NAE (not-all-equal) constraint (e.g. "h n .. 0" or "n .. 0", or "nae .. 0")
           * 'a[mo]': AMO (at-most-one) constraint (e.g. "h a .. 0", "h amo .. 0", or "a .. 0")
           * 'e[o]': EO (exactly-one) constraint (e.g. "h e .. 0", "h eo .. 0", or "eo .. 0")
           * 'k|ek': EK (exactly-K) constraint. EK has an additional formatting quirk:
              - Format: "k <k> .. 0" or "h k <k> .. 0" or "h ek <k> .. 0"
              - Where <k> is a positive integer
              - Example: "ek 2 1 2 3 0" means "exactly 2 of {$x_1, x_2, x_3$} must be true"
           * '[car]d': CARD (cardinality) constraint. CARD has an additional formatting quirk:
              - Format: "card <k> .. 0" or "h d <k> .. 0" or "d <k> .. 0"
              - Where <k> is a non-zero integer OR has a positive integer with inequality prefix ('<','<=','>','>=')
              - If no inequality prefix is supplied, then:
                - For positive k the default is '>=' (at least k true)
                - For negative k the default is '<' (at most k-1 true)
              - Example: "card 2 1 2 3 0" means "at least 2 of {$x_1, x_2, x_3$} must be true" ('>=')
              - Example: "card >=2 1 2 3 0" means "at least 2 of {$x_1, x_2, x_3$} must be true"
              - Example: "card >2 1 2 3 0" means "more than 2 of {$x_1, x_2, x_3$} must be true"
              - Example: "card -3 1 2 3 4 0" means "*fewer* than 3 of {$x_1, x_2, x_3, x_4$} must be true" ('<')
              - Example: "card <3 1 2 3 4 0" means "fewer than 3 of {$x_1, x_2, x_3, x_4$} must be true"
              - Example: "card <=2 1 2 3 0" means "at most 2 of {$x_1, x_2, x_3$} must be true"

        Args:
            dimacs_file (str): Path to the DIMACS format file to be parsed
        """
        if self._loaded_file is not None:
            raise RuntimeError(
                f"Problem already loaded ('{self._loaded_file}')! "
                "Create a new PBSATFormula instance for another problem file."
            )

        self.clause_sets = {}
        self.unit_prefix = set()
        self.seen_max_var = 0
        self.seen_clauses = 0

        throttle = LogThrottle(logger)
        used_input_vars: set[int] = set()

        def __process_clause(idx: int, line: str, tokens: list[str]) -> None:
            """
            Helper function to processes and validates a single clause, updating self.clause_sets
            Args:
                idx (int): Line number in the file for error reporting.
                line (str): The original line text for error messages.
                tokens (list[str]): Tokenized clause components including optional clause type,
                                   cardinality value (for card/ek clauses), literals, and trailing zero.
            Raises:
                ValueError: If the clause is malformed, has unknown type, or violates clause semantics.
                UnsatError: If the clause or its implications create an immediate unsatisfiability
                           (e.g., conflicting unit literals, card > n).
            """
            lit_offset = 0
            clause_type = "cnf"  # default assumption

            if tokens[0] == "h" or (tokens[0] == "1" and tokens[1] == "x"):  # "1 x" valid when using "h"
                lit_offset = 1
            if tokens[lit_offset].isalpha():
                # First part of clause is always after the (optional) h and clause type (unless cnf)
                clause_type = tokens[lit_offset] if tokens[lit_offset].isalpha() else "cnf"
                lit_offset += 1
            canon_types = {"d": "card", "x": "xor", "n": "nae", "e": "eo"}
            clause_type = canon_types[clause_type] if clause_type in canon_types else clause_type
            if clause_type not in clause_type_ids:
                logger.error(f"Unknown clause type: {line}")
                raise ValueError

            if lit_offset >= len(tokens) - 1:
                logger.error(f"Line {idx}: Malformed clause: {line}")
                raise ValueError

            card = 0

            # Normalise k for CARD/EK where it can be negative or a non-trivial inequality
            if clause_type in ("card", "ek"):
                # Catch implicit inequality (is just an int) - e.g. >= card or < card (for negative)
                try:
                    card = int(tokens[lit_offset])
                # Must have an explicit inequality indicator or is malformed/typo
                except ValueError:
                    # Check for strict inequality. All cardinality clauses are normalised to default forms:
                    # ">=" or "<", so we adjust the sign and value of k (card).
                    ineq = tokens[lit_offset]
                    if ineq[0] in ("<", ">") and clause_type == "card":
                        negate = ineq[0] == "<"
                        equality = ineq[1] == "="
                        skip = negate + equality
                        card = int(ineq[skip:])
                        if card < 0:
                            logger.error(f"Line {idx}: Inequality cardinality must be positive: {line}")
                            raise ValueError
                        card = (-1) ** negate * (card + (not (negate ^ equality)))
                    else:
                        logger.error(f"Line {idx}: EK constraint cannot have inequality: {line}")
                        raise ValueError
                lit_offset += 1

            lits = [int(val) for val in tokens[lit_offset:-1]]  # drop trailing 0
            n = len(lits)

            if 0 in lits:
                logger.error(f"Line {idx}: Clause literals must be non-zero integers: {line}")
                raise ValueError

            if any(abs(lit) > INT64_MAX for lit in lits):
                logger.error(f"Line {idx}: Literal index exceeds signed 64-bit range: {line}")
                raise ValueError

            self.seen_clauses += 1
            if n:
                abs_lits = [abs(lit) for lit in lits]
                self.seen_max_var = max(self.seen_max_var, max(abs_lits))
                used_input_vars.update(abs_lits)

            # Clause extracted. Check for errors in spec, correct generic edge cases.
            if n == 0:
                throttle.debug("empty", f"Line {idx}: Skipping empty clause")
                return

            if n == 1:
                match clause_type:
                    case "nae" | "xor":
                        logger.error(f"Line {idx}: Length 1 NAE/XOR clause has no SAT semantics (UNSAT): {line}")
                        raise UnsatError

                    case "amo":
                        throttle.debug("u_amo", f"Line {idx}: Skipping length 1 AMO clause (trivially SAT): {line}")
                        return

                    case "eo" | "cnf":
                        throttle.debug("u_eo", f"Line {idx}: Prefixing unit literals enoded as EO: {line}")
                        if -lits[0] in self.unit_prefix:
                            logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                            raise UnsatError
                        else:
                            self.unit_prefix.add(lits[0])
                            return
                    case _:
                        pass

            # Correct CARD/EK edge cases -- flag, correct if possible.
            if clause_type in ("card", "ek"):
                if card < 0:
                    # $CARD_{k<0}(X)\equiv CARD_{n-k+1}(\lnot X)$, $EK_{-k}(X) \equiv EK_k(\lnot X)$
                    throttle.debug("neg_k", f"Line {idx}: Normalising -ve {clause_type.upper()} to +ve negated: {line}")
                    lits = [-lit for lit in lits]
                    if clause_type == "card":
                        card = n + card + 1
                    else:
                        card = -card

                if card > n:
                    logger.error(f"Line {idx}: CARD/EK claues with card > #lits (always UNSAT): {line}")
                    raise UnsatError

                if card == n:
                    throttle.debug("k=n", f"Line {idx}: Prefixing {n} unit literals enoded as CARD/EK-{n}: {line}")
                    for lit in lits:
                        if -lit in self.unit_prefix:
                            logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                            raise UnsatError
                        else:
                            self.unit_prefix.add(lit)
                    return

                if card == 0:
                    if clause_type == "card":
                        throttle.debug("c0", f"Line {idx}: Skipping CARD-0 clause (trivially SAT): {line}")
                        return
                    else:
                        throttle.debug("ek0", f"Line {idx}: Prefixing negated EK-0 clause (trivially SAT): {line}")
                        for lit in lits:
                            if lit in self.unit_prefix:
                                logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                                raise UnsatError
                            else:
                                self.unit_prefix.add(-lit)
                        return

                if card == 1:
                    if clause_type == "card":
                        throttle.debug("c1", f"Line {idx}: Adjusting non-trivial CARD-1 clause to CNF: {line}")
                        clause_type = "cnf"
                    else:
                        throttle.debug("ek1", f"Line {idx}: Adjusting non-trivial EK-1 clause to EO: {line}")
                        clause_type = "eo"
                    card = 0

                if card == n - 1:
                    if clause_type == "card":
                        throttle.debug("cn1", f"Line {idx}: Adjusting CARD-(n-1) clause to negated AMO : {line}")
                        lits = [-lit for lit in lits]
                        clause_type = "amo"
                        card = 0
                    else:
                        throttle.debug("ekn1", f"Line {idx}: Adjusting EK-(n-1) clause to negated EO: {line}")
                        lits = [-lit for lit in lits]
                        clause_type = "eo"
                        card = 0

            if n == 2:
                match clause_type:
                    case "nae" | "eo":
                        throttle.debug("2xor", f"Line {idx}: Adjusting Length 2 NAE/EO clause to XOR: {line}")
                        clause_type = "xor"
                    case _:
                        pass

            self.clause_sets.setdefault(ClauseSignature(clause_type, n, card), []).append(lits)

        try:
            with open(dimacs_file, "r") as f:
                for idx, ln in enumerate(f):
                    tokens = ln.split()
                    line = ln.strip()

                    # Skip comments / empties
                    if len(line) == 0 or tokens[0] == "c" or tokens[0] == "*":
                        pass

                    # Problem metadata
                    elif tokens[0] == "p":
                        if len(tokens) in (4, 5):
                            # 4 was -2/-1 and 5 was -3/-2, but I think that's
                            # always just 2 and 3 right? n_var is always index 2 and n_clause is always 3?
                            self.n_var = int(tokens[2]) #was -(len(tokens)-2)
                            self.n_clause = int(tokens[3]) #was  -(len(tokens)-3)
                        else:
                            logger.error(f"Line {idx}: Malformed problem specification: {line}")
                            raise ValueError

                    # Process contraint
                    else:
                        if len(tokens) < 2 or tokens[-1] != "0":
                            logger.error(f"Line {idx}: Malformed clause: {line}")
                            raise ValueError
                        try:
                            __process_clause(idx, line, tokens)
                        except UnsatError as e:
                            print('s UNSATISFIABLE')
                            raise e
                        except ValueError as e:
                            logger.error(f"Error processing clause on line {idx}: {line}")
                            raise e

                throttle.flush()

                if self.seen_clauses != self.n_clause or self.seen_max_var != self.n_var:
                    logger.warning(
                        f"Metadata mismatch! Header specified {self.n_var} variables and {self.n_clause} clauses, "
                        + f"but we processed {self.seen_max_var} variables and {self.seen_clauses} clauses!"
                    )
                    self.n_clause = self.seen_clauses

                if self.compactify:
                    self.var_mapper.build_map(used_input_vars)
                    self.n_var = len(self.var_mapper.used_input_vars)
                    self.clause_sets = self.var_mapper.remap_clause_set(self.clause_sets)
                    self.unit_prefix = self.var_mapper.remap_prefix(self.unit_prefix)
                else:
                    # Stay in input-id space for low-overhead workflows like validation.
                    self.n_var = max(self.seen_max_var, self.n_var)

                logger.info(
                    f"Processed file: {dimacs_file}, with {len(self.clause_sets)} objectives (clause sets)"
                    f" - a total of {self.n_clause} clauses over {self.n_var} active variables"
                )
        except FileNotFoundError as e:
            print(f"Error: File '{dimacs_file}' not found")
            raise e
        except Exception as e:
            print(f"Error processing file: {e}")
            raise e
        self._loaded_file = dimacs_file

    def process_clauses_to_array(self) -> tuple[Objective, ...]:
        """
        Process and group clauses for efficient parallel computation.
        This method organizes clauses into groups based on their signatures and lengths,
        then processes them in parallel to create Objective instances.
        Grouping strategy:
            - Singletons (unique clause signatures): Grouped into single array by clause length
            - Non-singletons (multiple clauses per signature): Kept as separate arrays
            - Unique-length singletons: Combined into a single padded group
        The method uses multithreading to process clause groups in parallel, with each
        group being transformed into an Objective by the ClauseProcessor.
        Returns:
            tuple[Objective, ...]: A sorted tuple of Objective instances, ordered by
                the number of literals in their clauses (ascending).
        Notes:
            - Using multiple workers with XLA's "all" persistent cache mode can cause
              cache conflicts. If persistent caching is enabled for all components,
              workers should be set to 1 to avoid race conditions.
            - The number of workers is capped at the minimum of clause groups count
              and the configured worker limit.
        """

        class Singleton(NamedTuple):
            # A singleton is a clause that is unique in its overall signature for the problem
            sig: ClauseSignature
            clause: Clause

        class ClauseGroup(NamedTuple):
            sigs: list[ClauseSignature]
            clauses: Clauses

        clause_grps: list[ClauseGroup] = list()
        singletons_by_len: dict[int, list[Singleton]] = dict()
        padded_group: list[Singleton] = list()

        for set_signature, set_clauses in self.clause_sets.items():
            # Gather singletons by common length for more efficient processing
            if len(set_clauses) == 1:
                singletons_by_len.setdefault(set_signature.len, []).append(Singleton(set_signature, set_clauses[0]))

            # Homogenous set, so no grouping required
            else:
                clause_grps.append(ClauseGroup([set_signature], set_clauses))

        for singletons in singletons_by_len.values():
            # Collect the unique single lengthers for padded processing
            if len(singletons) == 1:
                padded_group.extend(singletons)
            else:
                sigs, clause_lists = zip(*singletons)
                clause_grps.append(ClauseGroup(list(sigs), list(clause_lists)))

        if padded_group:
            padded_sigs, padded_clause_lists = zip(*padded_group)
            clause_grps.append(ClauseGroup(list(padded_sigs), list(padded_clause_lists)))

        def parallel_clause_process(clause_grps: list[ClauseGroup], workers: int = 1) -> list[Objective]:
            processor = ClauseProcessor(self.n_devices, self.disk_cache)

            res: list[Objective] = []
            with ThreadPoolExecutor(max_workers=workers) as tpool:
                tasks = [tpool.submit(processor.process, grp.sigs, grp.clauses) for grp in clause_grps if grp]
                for task in tasks:
                    res.append(task.result())

            return res

        # N.B. Using multiple workers can cause XLA cache conflicts if using "all" persistent caches.
        # All is broken for some jaxopt optimizers however, so we don't use it. If we do, workers should be 1 to avoid
        # race conditions deep in XLA (see jax.config.update("jax_persistent_cache_enable_xla_caches", "all"))
        objectives = parallel_clause_process(clause_grps, workers=min(len(clause_grps), self.workers))
        objectives = tuple(sorted(objectives, key=lambda x: x.clauses.lits.shape[-1]))
        self.n_clause = sum([o.clauses.lits.shape[0] for o in objectives])
        return objectives

    def process_prefix(self, prefix_file: str) -> NDArray | None:
        def __lits_to_prefix(lits: Iterable[int]) -> NDArray:
            vec = np.zeros(self.n_var + 1, dtype=int)
            lit_vec = np.array([int(lit) for lit in lits], dtype=int)
            vec[abs(lit_vec[lit_vec < 0])] = 1
            vec[abs(lit_vec[lit_vec > 0])] = -1
            return vec

        try:
            vecs = []
            if prefix_file:
                skipped = 0
                total = 0
                with open(prefix_file, "r") as f:
                    for idx, line in enumerate(f):
                        raw_tokens = line.strip().split()
                        if not raw_tokens or raw_tokens[0] in ("c", "#", "*"):
                            continue
                        total += 1
                        try:
                            prefix_lits_raw: set[int] = set()
                            for idx, token in enumerate(raw_tokens):
                                lit = int(token)
                                if lit == 0:
                                    if idx == len(raw_tokens) - 1:
                                        raise ValueError("Prefix literals must be non-zero integers")
                                    continue
                                if abs(lit) > INT64_MAX:
                                    raise ValueError("Prefix literal exceeds signed 64-bit range")
                                prefix_lits_raw.add(lit)

                            if self.compactify and self.var_mapper.input_to_dense_var:
                                prefix_lits: set[int] = set()
                                for lit in prefix_lits_raw:
                                    dense_lit = self.var_mapper.input_to_dense(lit, strict=False) if self.compactify else lit
                                    if dense_lit is None or dense_lit == 0:
                                        logger.warning(f"Line {idx}: Invalid prefix literal {lit} is ignored")
                                        continue
                                    if abs(dense_lit) > self.n_var:
                                        logger.warning(
                                            f"Line {idx}: Prefix literal {lit} is out of range for loaded variables and is ignored"
                                        )
                                        continue
                                    prefix_lits.add(dense_lit)
                            else:
                                prefix_lits = prefix_lits_raw

                            # Invert and check for overlap implies a conflict between the problem and the prefix.
                            neg_lits = set(-lit for lit in prefix_lits)
                            self_conflict = prefix_lits.intersection(neg_lits)
                            if self_conflict:
                                logger.warning(f"Conflict ({self_conflict}) within prefix-{idx}- skipping: {line}")
                                skipped += 1
                                continue
                            unit_conflict = self.unit_prefix.intersection(neg_lits)
                            if unit_conflict:
                                logger.warning(
                                    f"Conflict ({unit_conflict}) with unit literals in prefix-{idx}- skipping: {line}"
                                )
                                skipped += 1
                                continue

                            merged = prefix_lits | self.unit_prefix
                            vecs.append(__lits_to_prefix(set(sorted(merged, key=lambda x: abs(x)))))
                        except (ValueError, TypeError):
                            logger.warning(f"Line {idx}: Invalid prefix entry: {line.strip()}")
                            continue
                if total > 0 and skipped == total:
                    logger.error("All prefixes skipped due to conflicts with unit literals or malformation")
                    raise UnsatError

            if not vecs and self.unit_prefix:
                # No prefix file or no valid file prefixes — unit literals are the sole prefix
                vecs.append(__lits_to_prefix(self.unit_prefix))

            if vecs:
                prefixes = np.delete(np.stack(vecs), 0, axis=1)  # purge leading zeros.
                return prefixes
            else:
                return None

        except FileNotFoundError as e:
            print(f"Error: File '{prefix_file}' not found")
            raise e

        except Exception as e:
            print(f"Error processing prefix file: {e}")
            raise e
