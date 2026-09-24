# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from sat_loader import Clauses

try:
    import galois
except ImportError:
    galois = None

logger = logging.getLogger(__name__)


class XorRREFMetadata(NamedTuple):
    """RREF of the XOR system in padded index form.

    Row i determines variable `dependent_indices[i]` as the XOR of the free variables
    `row_vars[i, row_mask[i]]` plus `b_final[i]`. Rows are padded to K = the maximum row
    weight, mirroring `clauses.lits` / `clauses.mask`. RREF rows are sparse in practice
    (mean weight ~3 on SHA-1), so this is two orders of magnitude smaller than the dense
    (n_dep, n_free) coefficient block, which is almost entirely zeros.
    """

    row_vars: Array  # (n_dep, K) int32 variable ids of each row's free support, padded
    row_mask: Array  # (n_dep, K) bool, False on padding
    row_sign: Array  # (n_dep,) 1 - 2 * b_final
    b_final: Array  # (n_dep,)
    dependent_indices: Array  # (n_dep,)
    free_indices: Array  # (n_free,) host bookkeeping; not read by the projector
    clause_count: int
    variable_count: int


class XorRREFObjective(NamedTuple):
    meta: XorRREFMetadata


class XorPreprocessResult(NamedTuple):
    """Outcome of XOR preprocessing.

    `meta` is None either because the system needs no projection (it was fully resolved by
    unit propagation, or there were no XOR clauses) or because it is inconsistent -- check
    `unsat` to tell those apart. `derived` holds variables newly implied by propagation, as
    variable index -> bit; the caller must feed these back into the unit prefix so they reach
    `fixed_vars` / `x0`.
    """

    meta: XorRREFMetadata | None
    derived: dict[int, int]
    unsat: bool


def unit_bits_from_literals(literals: Iterable[int]) -> dict[int, int]:
    """Map signed literals to {variable index: bit}, where bit 1 means the variable is TRUE.

    `_lits_to_prefix` maps a positive literal to `vec = -1` and the solver reads `x < 0` as
    TRUE, so a positive literal is bit 1.
    """
    return {abs(int(lit)) - 1: 1 if int(lit) > 0 else 0 for lit in literals}


def literals_from_unit_bits(bits: dict[int, int]) -> list[int]:
    """Inverse of `unit_bits_from_literals`."""
    return [(var + 1) if bit else -(var + 1) for var, bit in bits.items()]


def _xor_rows_from_clause_sets(xor_clause_sets: list[Clauses]) -> list[tuple[set[int], int]]:
    """One (variable set, parity) pair per XOR clause.

    Mirrors the matrix builder exactly: a repeated variable cancels (`matrix[row, col] ^= 1`
    twice), and parity starts at 1 and flips once per negative literal, so the row asserts
    `XOR_j bit_j = parity`.
    """
    rows: list[tuple[set[int], int]] = []
    for clause_set in xor_clause_sets:
        for clause in clause_set:
            variables: set[int] = set()
            parity = 1
            for lit in clause:
                lit = int(lit)
                variables ^= {abs(lit) - 1}
                if lit < 0:
                    parity ^= 1
            rows.append((variables, parity))
    return rows


def _propagate_xor_units(
    rows: list[tuple[set[int], int]], known: dict[int, int]
) -> tuple[list[tuple[set[int], int]], dict[int, int], bool]:
    """Substitute known bits out of the XOR rows and cascade.

    Substituting a known variable moves it to the right-hand side: drop it from the row and
    XOR its bit into the parity. A row left with one variable implies it, which may cascade;
    a row left with none is either redundant (parity 0) or a refutation (parity 1).

    Occurrence list plus a worklist, so each literal is removed at most once -- O(nnz)
    amortised, measured at a few percent of the time it takes to read the instance.

    Returns the surviving rows, the newly derived assignments, and whether the system is
    inconsistent.
    """
    rows = [(set(variables), parity) for variables, parity in rows]
    occurrences: dict[int, list[int]] = {}
    for idx, (variables, _) in enumerate(rows):
        for var in variables:
            occurrences.setdefault(var, []).append(idx)

    assignments = dict(known)
    queue = list(known)
    alive = [True] * len(rows)
    derived: dict[int, int] = {}
    unsat = False

    def settle(idx: int) -> None:
        """Retire a row that has fallen to zero or one variable."""
        nonlocal unsat
        variables, parity = rows[idx]
        if len(variables) > 1:
            return
        alive[idx] = False
        if not variables:
            if parity == 1:
                unsat = True
            return
        implied = next(iter(variables))
        if implied in assignments:
            if assignments[implied] != parity:
                unsat = True
            return
        assignments[implied] = parity
        derived[implied] = parity
        queue.append(implied)

    # Cancellation above can already leave a row with one variable or none.
    for idx in range(len(rows)):
        settle(idx)

    head = 0
    while head < len(queue):
        var = queue[head]
        head += 1
        bit = assignments[var]
        for idx in occurrences.get(var, ()):
            if not alive[idx]:
                continue
            variables, parity = rows[idx]
            if var not in variables:
                continue
            variables.discard(var)
            rows[idx] = (variables, parity ^ bit)
            settle(idx)

    remaining = [rows[idx] for idx in range(len(rows)) if alive[idx]]
    return remaining, derived, unsat


def _rref_gf2_numpy(matrix: NDArray, parity: NDArray) -> tuple[NDArray, NDArray, bool]:
    n_rows, n_cols = matrix.shape
    aug = np.concatenate([matrix, parity[:, None]], axis=1).astype(np.uint8)

    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break

        candidates = np.where(aug[pivot_row:, col] == 1)[0]
        if candidates.size == 0:
            continue

        pivot = int(candidates[0] + pivot_row)
        if pivot != pivot_row:
            aug[[pivot_row, pivot]] = aug[[pivot, pivot_row]]

        for rr in range(n_rows):
            if rr != pivot_row and aug[rr, col] == 1:
                aug[rr, :] ^= aug[pivot_row, :]

        pivot_row += 1

    coeff = aug[:, :n_cols]
    rhs = aug[:, n_cols]
    inconsistent = bool(np.any((np.sum(coeff, axis=1) == 0) & (rhs == 1)))
    return coeff, rhs, inconsistent


def _padded_row_supports(free_part: NDArray, free_idx: NDArray) -> tuple[NDArray, NDArray]:
    """Convert the dense (n_dep, n_free) 0/1 block into padded (n_dep, K) variable ids + mask."""
    n_dep = free_part.shape[0]
    rows, cols = np.nonzero(free_part)
    counts = np.bincount(rows, minlength=n_dep)
    width = max(int(counts.max()) if counts.size else 0, 1)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if n_dep else np.zeros(0, dtype=np.int64)
    positions = np.arange(rows.size) - offsets[rows]

    row_vars = np.zeros((n_dep, width), dtype=np.int32)
    row_mask = np.zeros((n_dep, width), dtype=bool)
    row_vars[rows, positions] = free_idx[cols]
    row_mask[rows, positions] = True
    return row_vars, row_mask


def _build_xor_rref_metadata_from_matrix(
    matrix: NDArray, parity: NDArray, xor_vars: list[int], clause_count: int
) -> tuple[XorRREFMetadata | None, dict[int, int], bool]:
    """Reduce the system and read off the dependent/free split.

    Returns the metadata, any dependents the reduction *forces* outright, and whether the
    system is inconsistent. A forced dependent is a row left with no free support: `x_d = b`.
    Only elimination can expose these -- they need a linear combination of clauses, so unit
    propagation structurally cannot find them (`a^b=0` with `a^b^c=1` implies `c=1`, but
    neither clause is ever a unit). Since RREF pivot columns are unit vectors, a forced
    dependent appears in no other row, so hoisting it needs no re-elimination and cannot
    cascade within the system: one pass is exact.
    """
    n_cols = matrix.shape[1]

    if galois is not None:
        GF2 = galois.GF(2)
        augmented = np.concatenate([matrix, parity[:, None]], axis=1)
        rref = np.array(GF2(augmented).row_reduce(), dtype=np.uint8)
        coeff = rref[:, :n_cols]
        rhs = rref[:, n_cols]
        inconsistent = bool(np.any((np.sum(coeff, axis=1) == 0) & (rhs == 1)))
    else:
        coeff, rhs, inconsistent = _rref_gf2_numpy(matrix, parity)

    if inconsistent:
        logger.warning("XOR system is inconsistent after RREF; XOR projection is disabled")
        return None, {}, True

    active_rows = np.where(np.sum(coeff, axis=1) > 0)[0]
    xor_vars_np = np.array(xor_vars, dtype=np.int32)

    if active_rows.size == 0:
        dep_idx = np.array([], dtype=np.int32)
        free_idx = xor_vars_np
        free_part = np.zeros((0, n_cols), dtype=np.uint8)
        b_final = np.zeros((0,), dtype=np.float32)
    else:
        dep_local = np.array([int(np.argmax(coeff[row])) for row in active_rows], dtype=np.int32)
        free_mask = np.ones(n_cols, dtype=bool)
        free_mask[dep_local] = False
        free_local = np.where(free_mask)[0].astype(np.int32)

        dep_idx = xor_vars_np[dep_local]
        free_idx = xor_vars_np[free_local]
        free_part = coeff[np.ix_(active_rows, free_local)].astype(np.uint8)
        b_final = rhs[active_rows].astype(np.float32)

    # Rows the reduction leaves with no free support force their dependent outright.
    forced: dict[int, int] = {}
    if dep_idx.size:
        forced_rows = free_part.sum(axis=1) == 0
        if forced_rows.any():
            forced = {
                int(var): int(bit)
                for var, bit in zip(dep_idx[forced_rows], b_final[forced_rows].astype(np.int64))
            }
            keep = ~forced_rows
            dep_idx = dep_idx[keep]
            free_part = free_part[keep]
            b_final = b_final[keep]
            logger.info("XOR RREF forced %d dependent variables outright", len(forced))

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "XOR RREF breakdown: clauses=%d vars=%d dependent=%d free=%d",
            clause_count,
            n_cols,
            int(dep_idx.size),
            int(free_idx.size),
        )
        if dep_idx.size <= 64:
            logger.info("XOR dependent vars (0-indexed): %s", dep_idx.tolist())
        if free_idx.size <= 64:
            logger.info("XOR free vars (0-indexed): %s", free_idx.tolist())

    row_vars, row_mask = _padded_row_supports(free_part, free_idx)
    meta = XorRREFMetadata(
        row_vars=jnp.array(row_vars),
        row_mask=jnp.array(row_mask),
        row_sign=jnp.array(1.0 - 2.0 * b_final, dtype=np.float32),
        b_final=jnp.array(b_final),
        dependent_indices=jnp.array(dep_idx),
        free_indices=jnp.array(free_idx),
        clause_count=clause_count,
        variable_count=n_cols,
    )
    return meta, forced, False


def preprocess_xor_system(
    xor_clause_sets: list[Clauses], unit_bits: dict[int, int] | None = None, build_rref: bool = True
) -> XorPreprocessResult:
    """Propagate known units through the XOR system, then build the RREF of what remains.

    Substituting the units out before reduction, rather than leaving them to collide with the
    pivot choice, is what keeps the projection's "every XOR clause satisfied by construction"
    invariant intact under a prefix: a variable that is no longer in the system cannot be
    chosen as a pivot. It also shrinks the system, sparsifies the rows, and turns up further
    implied literals for free.

    Two tiers of inference happen here, and they are gated differently:

    1. Unit propagation over the clauses. This is plain problem simplification -- it needs no
       reduction and is valid whether or not the projection is used -- so callers should run
       it unconditionally (`build_rref=False` stops after this tier).
    2. Forced dependents revealed by the reduction itself. These require a linear combination
       of clauses to expose, so only elimination can find them, and they are therefore
       inherently gated on the RREF being built at all.

    NOTE: tier 1 propagates only *within* the XOR system. Running full CNF+XOR unit
    propagation finds substantially more for the same cost and should be done wherever
    possible -- standalone especially, which has no upstream preprocessing. Implied literals
    can simply join the unit prefix without rewriting any clause arrays; only the further
    optimisation of *shrinking* those arrays needs a per-clause-type story (and the PB types
    each need their own propagator, defaulting to none, which loses simplification but never
    correctness).
    """
    rows = _xor_rows_from_clause_sets(xor_clause_sets)
    if not rows:
        return XorPreprocessResult(None, {}, False)

    remaining, derived, unsat = _propagate_xor_units(rows, dict(unit_bits or {}))
    if unsat:
        logger.warning("XOR system is inconsistent under unit propagation")
        return XorPreprocessResult(None, derived, True)

    if not build_rref:
        return XorPreprocessResult(None, derived, False)

    n_rows = len(remaining)
    xor_vars = sorted({var for variables, _ in remaining for var in variables})
    if n_rows == 0 or not xor_vars:
        logger.info("XOR system fully resolved by unit propagation; no projection needed")
        return XorPreprocessResult(None, derived, False)

    var_to_col = {var_idx: col_idx for col_idx, var_idx in enumerate(xor_vars)}
    n_cols = len(xor_vars)
    matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)
    parity = np.zeros(n_rows, dtype=np.uint8)
    for row, (variables, row_parity) in enumerate(remaining):
        for var_idx in variables:
            matrix[row, var_to_col[var_idx]] ^= 1
        parity[row] = row_parity

    meta, forced, unsat = _build_xor_rref_metadata_from_matrix(matrix, parity, xor_vars, n_rows)
    if unsat or meta is None:
        return XorPreprocessResult(None, derived, True)
    # Tier 2. Forced pivots appear in no other row, so hoisting them cannot cascade within the
    # reduced system, and they cannot collide with tier 1 -- those variables were substituted
    # out before reduction and are no longer columns at all.
    derived.update(forced)
    return XorPreprocessResult(meta, derived, False)
