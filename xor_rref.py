# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from sat_loader import Clauses

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

    A repeated variable cancels (`x ^ x = 0`, so an even count drops out), and parity starts
    at 1 and flips once per negative literal, so the row asserts `XOR_j bit_j = parity`.
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


def _rref_gf2_sparse(rows: list[tuple[set[int], int]], n_cols: int) -> tuple[list[tuple[int, list[int], int]], bool]:
    """Gauss-Jordan elimination over GF(2) on sparse rows.

    Each row is a set of column indices, so eliminating one row from another is a symmetric
    difference costing the pivot row's weight, and a column -> rows occurrence map names
    exactly the rows to eliminate. Work scales with the non-zeros rather than rows x columns:
    on SHA-1 r80 (23.8k x 36k, ~16 non-zeros per reduced row) this takes ~1 s, where a dense
    byte-matrix reduction (`galois.row_reduce`) took ~55 s and ~1.7 GB. Sets rather than
    packed bitsets because RREF rows here stay far below 1% dense; a system that filled in
    densely would favour packed words instead.

    Pivot columns are taken in ascending order, so the result is *the* reduced row echelon
    form, which is unique: identical to any dense reduction of the same matrix. Which row
    supplies a pivot does not change that result, so the lightest candidate is taken to keep
    intermediate fill down.

    Returns (pivot column, sorted non-pivot support, rhs) per pivot row in ascending pivot
    order, and whether the system is inconsistent (a row reduced to `0 = 1`).
    """
    supports = [set(cols) for cols, _ in rows]
    rhs = [parity for _, parity in rows]
    occurrences: dict[int, set[int]] = {}
    for idx, cols in enumerate(supports):
        for col in cols:
            occurrences.setdefault(col, set()).add(idx)

    pivots: list[tuple[int, int]] = []  # (pivot column, row index)
    is_pivot_row = [False] * len(supports)
    for col in range(n_cols):
        candidates = [idx for idx in occurrences.get(col, ()) if not is_pivot_row[idx]]
        if not candidates:
            continue
        pivot = min(candidates, key=lambda idx: (len(supports[idx]), idx))
        is_pivot_row[pivot] = True
        pivots.append((col, pivot))
        pivot_support = supports[pivot]
        # Full Gauss-Jordan: clear the column from earlier pivot rows too, not just later ones.
        for idx in list(occurrences[col]):
            if idx == pivot:
                continue
            support = supports[idx]
            for var in pivot_support:
                if var in support:
                    support.discard(var)
                    occurrences[var].discard(idx)
                else:
                    support.add(var)
                    occurrences[var].add(idx)
            rhs[idx] ^= rhs[pivot]

    inconsistent = any(rhs[idx] and not supports[idx] for idx in range(len(supports)))
    reduced = [(col, sorted(supports[idx] - {col}), rhs[idx]) for col, idx in pivots]
    return reduced, inconsistent


def _build_xor_rref_metadata(
    rows: list[tuple[set[int], int]], xor_vars: list[int], clause_count: int
) -> tuple[XorRREFMetadata | None, dict[int, int], bool]:
    """Reduce the system and read off the dependent/free split.

    `rows` are (column set, parity) pairs over columns indexing the sorted `xor_vars`.

    Returns the metadata, any dependents the reduction *forces* outright, and whether the
    system is inconsistent. A forced dependent is a row left with no free support: `x_d = b`.
    Only elimination can expose these -- they need a linear combination of clauses, so unit
    propagation structurally cannot find them (`a^b=0` with `a^b^c=1` implies `c=1`, but
    neither clause is ever a unit). Since RREF pivot columns are unit vectors, a forced
    dependent appears in no other row, so hoisting it needs no re-elimination and cannot
    cascade within the system: one pass is exact.
    """
    n_cols = len(xor_vars)
    reduced, inconsistent = _rref_gf2_sparse(rows, n_cols)
    if inconsistent:
        logger.warning("XOR system is inconsistent after RREF; XOR projection is disabled")
        return None, {}, True

    xor_vars_np = np.array(xor_vars, dtype=np.int32)
    pivot_cols = {col for col, _, _ in reduced}
    free_idx = xor_vars_np[[col for col in range(n_cols) if col not in pivot_cols]]

    # Rows the reduction leaves with no free support force their dependent outright.
    forced = {int(xor_vars_np[col]): int(bit) for col, support, bit in reduced if not support}
    if forced:
        reduced = [row for row in reduced if row[1]]
        logger.info("XOR RREF forced %d dependent variables outright", len(forced))

    dep_idx = xor_vars_np[[col for col, _, _ in reduced]]
    b_final = np.array([bit for _, _, bit in reduced], dtype=np.float32)

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

    # Padded (n_dep, K) form, K = max row weight (at least 1 so the arrays are never empty).
    width = max((len(support) for _, support, _ in reduced), default=0)
    row_vars = np.zeros((len(reduced), max(width, 1)), dtype=np.int32)
    row_mask = np.zeros_like(row_vars, dtype=bool)
    for row, (_, support, _) in enumerate(reduced):
        row_vars[row, : len(support)] = xor_vars_np[support]
        row_mask[row, : len(support)] = True

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
    rows = [({var_to_col[var_idx] for var_idx in variables}, parity) for variables, parity in remaining]

    meta, forced, unsat = _build_xor_rref_metadata(rows, xor_vars, n_rows)
    if unsat or meta is None:
        return XorPreprocessResult(None, derived, True)
    # Tier 2. Forced pivots appear in no other row, so hoisting them cannot cascade within the
    # reduced system, and they cannot collide with tier 1 -- those variables were substituted
    # out before reduction and are no longer columns at all.
    derived.update(forced)
    return XorPreprocessResult(meta, derived, False)
