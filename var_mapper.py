# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy as np


logger = logging.getLogger(__name__)

Assignment: TypeAlias = tuple[int, ...] | tuple[bool, ...]


class VarMapper:
    """Canonical variable-id map, literal conversion, and assignment formatting API."""

    def __init__(self):
        self.dense_to_input_var: list[int] = []
        self.input_to_dense_var: dict[int, int] = {}
        self.used_input_vars: tuple[int, ...] = tuple()
        self.max_input_var: int = 0
        self.sparse_vars: bool = False

    def build_map(self, used_vars: set[int]) -> None:
        sorted_vars = sorted(used_vars)
        self.used_input_vars = tuple(sorted_vars)
        self.max_input_var = sorted_vars[-1] if sorted_vars else 0
        self.sparse_vars = self.max_input_var > len(sorted_vars) if sorted_vars else False
        self.input_to_dense_var = {var: idx for idx, var in enumerate(sorted_vars, start=1)}
        self.dense_to_input_var = [0] + sorted_vars

        if self.sparse_vars:
            logger.warning(
                "Sparse variable IDs detected: max input variable id %s with only %s used variables. "
                "Compacting to dense internal IDs for assignment vectors (e.g. %s->%s).",
                self.max_input_var,
                len(self.used_input_vars),
                self.max_input_var,
                len(self.used_input_vars),
            )

    def input_to_dense(self, lit: int, strict: bool = True) -> int | None:
        dense_var = self.input_to_dense_var.get(abs(lit))
        if dense_var is None:
            if strict:
                raise ValueError(f"Unknown input literal {lit}: variable {abs(lit)} not present in loaded formula")
            return None
        return dense_var if lit > 0 else -dense_var

    def dense_to_input(self, lit: int, strict: bool = False) -> int:
        dense_var = abs(lit)
        if dense_var == 0:
            return 0
        if not self.dense_to_input_var:
            return lit
        if dense_var >= len(self.dense_to_input_var):
            if strict:
                raise ValueError(f"Dense literal {lit} is out of range for loaded variable map")
            return lit
        input_var = int(self.dense_to_input_var[dense_var])
        if input_var == 0:
            if strict:
                raise ValueError(f"Dense literal {lit} has no mapped input variable")
            return lit
        return input_var if lit > 0 else -input_var

    def map_to_input(self, dense_id: int) -> int:
        """Map a compact variable/literal id to input id, preserving sign for literals."""
        return self.dense_to_input(dense_id, strict=False)

    def remap_literals(self, literals: Iterable[int], strict: bool = True) -> list[int]:
        remapped: list[int] = []
        for lit in literals:
            dense_lit = self.input_to_dense(lit, strict=strict)
            if dense_lit is None:
                continue
            remapped.append(dense_lit)
        return remapped

    def remap_literal_set(self, literals: set[int], strict: bool = True) -> set[int]:
        remapped: set[int] = set()
        for lit in literals:
            dense_lit = self.input_to_dense(lit, strict=strict)
            if dense_lit is None:
                continue
            remapped.add(dense_lit)
        return remapped

    def remap_clause_set(self, clause_sets: dict[Any, list[list[int]]]) -> dict[Any, list[list[int]]]:
        if not self.input_to_dense_var:
            return clause_sets

        remapped_clause_sets: dict[Any, list[list[int]]] = {}
        for signature, clauses in clause_sets.items():
            remapped_clause_sets[signature] = [self.remap_literals(clause, strict=True) for clause in clauses]
        return remapped_clause_sets

    def remap_prefix(self, literals: set[int]) -> set[int]:
        if not self.input_to_dense_var:
            return literals
        return self.remap_literal_set(literals, strict=True)

    def assn_str(self, assignment: Assignment, binary: bool = False, inc_zero: bool = False, bit_zero: str = "1") -> str:
        """Format an assignment tuple in canonical signed form for output.

        Canonical form uses {-1, 0, 1} where:
        -1 => assigned true, 1 => assigned false, 0 => unknown/unassigned.
        """
        arr = np.asarray(assignment)
        if arr.dtype == np.bool_:
            signed = np.where(arr, -1, 1).astype(int)
        else:
            signed = np.sign(arr).astype(int)
        canonical_assn = tuple(int(v) for v in signed.tolist())

        if binary:
            return self.bin_str(canonical_assn, bit_zero)
        else:
            return self.lits_str(canonical_assn, inc_zero)

    def bin_str(self, assignment: tuple[int, ...], unassigned: str = "1") -> str:
        """Build binary assignment string using input variable ids as bit indices.

        If a dense->input map is provided, output length expands to the highest referenced
        input variable id and unmapped/unassigned input ids are filled with a default bit.
        """
        if not self.dense_to_input_var:
            return "".join("0" if s > 0 else ("1" if s < 0 else "-") for s in assignment)

        max_input_var = 0
        for dense_var in range(1, len(assignment) + 1):
            mapped = self.map_to_input(dense_var)
            if mapped > max_input_var:
                max_input_var = mapped

        if max_input_var <= 0:
            return ""

        out = [unassigned] * max_input_var
        for dense_var, s in enumerate(assignment, start=1):
            input_var = self.map_to_input(dense_var)
            if input_var <= 0:
                continue

            out[input_var - 1] = "0" if s > 0 else ("1" if s < 0 else "-")

        return "".join(out)

    def lits_str(self, assignment: tuple[int, ...], inc_zero: bool = False) -> str:
        """Build space-separated literal assignment string using input-format variable ids."""
        literals: list[str] = []
        for var_idx_0, s in enumerate(assignment):
            lit = self.map_to_input(var_idx_0 + 1)
            if s == 0 and inc_zero:
                literals.append(f"?{lit}")
            elif s > 0:
                literals.append(f"-{lit}")
            else:
                literals.append(f"{lit}")
        return " ".join(literals)