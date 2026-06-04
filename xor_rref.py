# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

try:
    import galois
except ImportError:
    galois = None

logger = logging.getLogger(__name__)


class XorRREFMetadata(NamedTuple):
    rref_free_part: Array
    b_final: Array
    dependent_indices: Array
    free_indices: Array
    clause_count: int
    variable_count: int


class XorRREFObjective(NamedTuple):
    meta: XorRREFMetadata


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


def _build_xor_rref_metadata_from_matrix(
    matrix: NDArray, parity: NDArray, xor_vars: list[int], clause_count: int
) -> XorRREFMetadata | None:
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
        return None

    active_rows = np.where(np.sum(coeff, axis=1) > 0)[0]
    xor_vars_np = np.array(xor_vars, dtype=np.int32)

    if active_rows.size == 0:
        dep_idx = np.array([], dtype=np.int32)
        free_idx = xor_vars_np
        free_part = np.zeros((0, n_cols), dtype=np.float32)
        b_final = np.zeros((0,), dtype=np.float32)
    else:
        dep_local = np.array([int(np.argmax(coeff[row])) for row in active_rows], dtype=np.int32)
        free_mask = np.ones(n_cols, dtype=bool)
        free_mask[dep_local] = False
        free_local = np.where(free_mask)[0].astype(np.int32)

        dep_idx = xor_vars_np[dep_local]
        free_idx = xor_vars_np[free_local]
        free_part = coeff[np.ix_(active_rows, free_local)].astype(np.float32)
        b_final = rhs[active_rows].astype(np.float32)

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

    return XorRREFMetadata(
        rref_free_part=jnp.array(free_part),
        b_final=jnp.array(b_final),
        dependent_indices=jnp.array(dep_idx),
        free_indices=jnp.array(free_idx),
        clause_count=clause_count,
        variable_count=n_cols,
    )


def build_xor_rref_metadata_from_clause_sets(xor_clause_sets: Iterable[list[list[int]]]) -> XorRREFMetadata | None:
    clause_sets = list(xor_clause_sets)
    if not clause_sets:
        return None

    xor_vars_set: set[int] = set()
    n_rows = 0
    for clause_set in clause_sets:
        for clause in clause_set:
            n_rows += 1
            for lit in clause:
                xor_vars_set.add(abs(lit) - 1)

    if n_rows == 0 or not xor_vars_set:
        return None

    xor_vars = sorted(xor_vars_set)
    var_to_col = {var_idx: col_idx for col_idx, var_idx in enumerate(xor_vars)}
    n_cols = len(xor_vars)
    matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)
    parity = np.zeros(n_rows, dtype=np.uint8)

    row = 0
    for clause_set in clause_sets:
        for clause in clause_set:
            base_parity = 1
            for lit in clause:
                var_idx = abs(lit) - 1
                matrix[row, var_to_col[var_idx]] ^= 1
                if lit < 0:
                    base_parity ^= 1
            parity[row] = base_parity
            row += 1

    return _build_xor_rref_metadata_from_matrix(matrix, parity, xor_vars, n_rows)


def build_xor_rref_metadata_from_clauses(xor_clauses: list[list[int]]) -> XorRREFMetadata | None:
    return build_xor_rref_metadata_from_clause_sets([xor_clauses])


