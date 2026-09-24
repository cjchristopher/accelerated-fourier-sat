# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""Unit propagation over the whole hybrid formula, as preprocessing.

Implied literals are returned for the caller to add to the unit prefix, where they become
fixed variables. Nothing here rewrites or removes clauses: a clause the propagation satisfies
simply evaluates as satisfied under the pinned assignment, so correctness never depends on the
clause arrays being shrunk. (Shrinking them before the JAX stage is a separate, optional
optimisation that needs its own per-type rewrite rules.)

After the loader's normalisation every clause type except XOR is a bound on how many of its
literals are true:

    cnf   >= 1          amo   <= 1          eo    == 1
    card  >= k          ek    == k          nae   1 .. n-1

(`card` with negative k is normalised to positive k over negated literals, so `>= k` is the
only stored form.) One counting propagator therefore covers them all, and XOR gets a parity
propagator. A clause type with no registered rule is skipped: that loses simplification but
never correctness, so a new PB type is safe to add before it has a propagator.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

from boolean_whf import Clauses, ClauseSignature

logger = logging.getLogger(__name__)


def _count_bounds(signature: ClauseSignature, n: int) -> tuple[int, int] | None:
    """(lo, hi) bounds on the number of true literals, or None if the type is not a count."""
    match signature.type:
        case "cnf":
            return 1, n
        case "amo":
            return 0, 1
        case "eo":
            return 1, 1
        case "card":
            return signature.card, n
        case "ek":
            return signature.card, signature.card
        case "nae":
            return 1, n - 1
        case _:
            return None


class PropagationResult(NamedTuple):
    derived: set[int]  # newly implied literals (signed, 1-indexed)
    unsat: bool


def propagate_units(clause_sets: dict[ClauseSignature, Clauses], units: Iterable[int]) -> PropagationResult:
    """Propagate `units` through every clause to a fixpoint.

    Occurrence lists plus a worklist: each literal occurrence is visited once per assignment of
    its variable, so the whole pass is O(total literals).
    """
    value: dict[int, bool] = {}  # variable (1-indexed) -> True/False
    queue: list[int] = []
    unsat = False

    def assign(lit: int) -> None:
        nonlocal unsat
        var, val = abs(lit), lit > 0
        if var in value:
            if value[var] != val:
                unsat = True
            return
        value[var] = val
        queue.append(var)

    # Per-clause state. Counting clauses track true/unassigned occurrence counts; XOR clauses
    # track the unassigned variables and the parity still required of them.
    kind: list[str] = []
    lits_of: list[list[int]] = []
    bounds: list[tuple[int, int]] = []
    n_true: list[int] = []
    n_free: list[int] = []
    xor_parity: list[int] = []
    # Variables of each clause whose assignment has not been *processed* yet. This, not
    # `value`, is the clause's view of what is free: a variable can be assigned but still
    # queued, and forcing must then go through assign() so a clash is caught, not skipped.
    pending: list[set[int]] = []
    alive: list[bool] = []
    occurrences: dict[int, list[int]] = {}
    skipped: set[str] = set()

    for signature, clauses in clause_sets.items():
        for clause in clauses:
            lits = [int(lit) for lit in clause]
            n = len(lits)
            if signature.type == "xor":
                # XOR_j bit_j = parity, parity = 1 ^ (#negative literals mod 2), bit = variable is
                # TRUE. A variable repeated an even number of times cancels, exactly as in the RREF
                # builder, so store each surviving variable once (as a positive literal).
                parity = 1
                odd: set[int] = set()
                for lit in lits:
                    odd ^= {abs(lit)}
                    if lit < 0:
                        parity ^= 1
                lits = sorted(odd)
                n = len(lits)
                kind.append("xor")
                bounds.append((0, 0))
                xor_parity.append(parity)
            else:
                bound = _count_bounds(signature, n)
                if bound is None:
                    skipped.add(signature.type)
                    continue
                kind.append("count")
                bounds.append(bound)
                xor_parity.append(0)
            idx = len(lits_of)
            lits_of.append(lits)
            n_true.append(0)
            n_free.append(n)
            alive.append(True)
            pending.append({abs(lit) for lit in lits})
            for var in pending[-1]:
                occurrences.setdefault(var, []).append(idx)

    if skipped:
        logger.info("Unit propagation has no rule for clause types %s; skipped", sorted(skipped))

    def settle(idx: int) -> None:
        """Check a clause after its counts change; retire it or force its free literals."""
        nonlocal unsat
        if kind[idx] == "xor":
            if n_free[idx] == 0:
                alive[idx] = False
                if xor_parity[idx] != 0:
                    unsat = True
            elif n_free[idx] == 1:
                alive[idx] = False
                var = next(iter(pending[idx]))
                # the remaining bit must equal the outstanding parity
                assign(var if xor_parity[idx] else -var)
            return

        lo, hi = bounds[idx]
        t, u = n_true[idx], n_free[idx]
        if t > hi or t + u < lo:
            alive[idx] = False
            unsat = True
        elif t >= lo and t + u <= hi:
            alive[idx] = False  # satisfied whatever the rest do
        elif t == hi:
            alive[idx] = False
            for lit in lits_of[idx]:
                if abs(lit) in pending[idx]:
                    assign(-lit)
        elif t + u == lo:
            alive[idx] = False
            for lit in lits_of[idx]:
                if abs(lit) in pending[idx]:
                    assign(lit)

    for lit in units:
        assign(int(lit))
    seeded = set(value)

    # Clauses can be decided before any assignment (e.g. repeated variables in an XOR).
    for idx in range(len(lits_of)):
        if kind[idx] == "xor":
            settle(idx)

    head = 0
    while head < len(queue) and not unsat:
        var = queue[head]
        head += 1
        val = value[var]
        for idx in occurrences.get(var, ()):
            if not alive[idx]:
                continue
            pending[idx].discard(var)
            for lit in lits_of[idx]:
                if abs(lit) != var:
                    continue
                n_free[idx] -= 1
                if kind[idx] == "xor":
                    xor_parity[idx] ^= int(val)
                elif (lit > 0) == val:
                    n_true[idx] += 1
            settle(idx)

    derived = {var if val else -var for var, val in value.items() if var not in seeded}
    return PropagationResult(derived, unsat)
