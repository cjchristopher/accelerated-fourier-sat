#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""
Quick validation script to check which clauses are unsatisfied by a given solution.

Usage:
    python validate_solution.py <problem_file> <solution_file>

Example:
    python validate_solution.py tests/costas/no_break/costas_9.hybrid solution.sol

The solution file should contain space-separated literals where:
    - Positive integer N means variable N is TRUE
    - Negative integer -N means variable N is FALSE
    e.g: 
    -1 -2 -3 -4 -5 -6 -7 -8 9 -10 -11 12 -13 

Notes:
        - Validation runs with compactify=False, so literals are interpreted directly
            in input-ID space (no variable remapping).
        - If a solution comes from a differently encoded instance (e.g. extra
            auxiliaries), the user is responsible for ensuring variable-ID semantics
            still match the problem being checked.
"""
# ruff: disable[E402]
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

fparent = Path(__file__).resolve().parent
if fparent == Path.cwd():
    sys.path.insert(1, os.path.abspath("../"))
else:
    sys.path.insert(1, str(fparent.parent))

import numpy as np
from numpy.typing import NDArray

from sat_loader import PBSATFormula

# ruff: enable[E402]
INT64_MAX = int(np.iinfo(np.int64).max)


def load_solution_file(sol_file: str) -> str:
    """Load solution string from a .sol file."""
    with open(sol_file, "r") as f:
        # Read all lines, skip comments, join remaining content
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith(("#", "c", "*")):
                lines.append(line)
        return " ".join(lines)


def parse_solution(solution_str: str, formula: PBSATFormula) -> NDArray:
    """
    Parse a solution string like "-1 2 -3 4 5" into an assignment array.

    Args:
        solution_str: Space-separated literals (positive = true, negative = false)
        formula: Parsed SAT formula in input-ID space (compactify=False)

    Returns:
        Array of shape (n_var,) where index i stores variable (i+1) as -1 (true),
        +1 (false), or 0 (unassigned)

    Warning:
        Literals outside [1, n_var] are ignored with a warning. This allows
        solutions to include additional literals not used by the loaded problem,
        but semantic compatibility is the caller's responsibility.
    """
    lits = [int(x) for x in solution_str.strip().split()]

    # Assignment array: -1 means satisfied (literal matches), +1 means unsatisfied
    # This matches the convention used in afsat where sign * x[lits] < 0 means satisfied
    assignment = np.zeros(formula.n_var, dtype=np.int8)

    for lit in lits:
        if lit == 0:
            print("Warning: Ignoring zero literal in solution file")
            continue

        if abs(lit) > INT64_MAX:
            raise ValueError(f"Literal {lit} exceeds signed 64-bit range")

        if abs(lit) > formula.n_var:
            print(f"Warning: Variable {abs(lit)} is out of range for this formula, skipping")
            continue

        var = abs(lit) - 1
        # Positive lit means var is TRUE -> assignment[var] = -1
        # Negative lit means var is FALSE -> assignment[var] = +1
        assignment[var] = -1 if lit > 0 else 1

    return assignment


def _unsat_mask_for_type(clause_type: str, signed_assignments: NDArray, card: int) -> NDArray[np.bool_]:
    neg_count = np.sum(signed_assignments < 0, axis=1)

    if clause_type == "xor":
        return neg_count % 2 == 0
    if clause_type == "cnf":
        return np.min(signed_assignments, axis=1) > 0
    if clause_type == "eo":
        return neg_count != 1
    if clause_type == "amo":
        return neg_count > 1
    if clause_type == "nae":
        return ~((np.min(signed_assignments, axis=1) < 0) & (np.max(signed_assignments, axis=1) > 0))
    if clause_type == "card":
        if card < 0:
            return neg_count >= abs(card)
        return neg_count < card
    if clause_type == "ek":
        return neg_count != card
    raise ValueError(f"Unknown clause type: {clause_type}")


def get_unsatisfied_clauses(formula: PBSATFormula, assignment: NDArray) -> list[dict]:
    """
    Find all clauses unsatisfied by the given assignment.

    Returns a list of dicts with clause information:
        - 'clause_type': str (cnf, xor, eo, etc.)
        - 'literals': list of ints
        - 'cardinality': int (for card/ek clauses)
        - 'objective_idx': which objective group it belongs to
        - 'clause_idx': index within that objective
    """
    unsatisfied = []

    # Iterate clause sets directly to avoid Objective/FFT construction overhead.
    for obj_idx, (signature, clause_list) in enumerate(formula.clause_sets.items()):
        if not clause_list:
            continue

        clause_arr = np.asarray(clause_list, dtype=np.int64)
        signs = np.sign(clause_arr).astype(np.float32)
        lit_idx = np.abs(clause_arr) - 1
        signed_assignments = signs * assignment[lit_idx]

        unsat_mask = _unsat_mask_for_type(signature.type, signed_assignments, signature.card)
        unsat_indices = np.where(unsat_mask)[0]

        for clause_idx in unsat_indices:
            clause_lits = clause_arr[clause_idx].tolist()
            clause = set(int(x) for x in clause_lits)
            assigned = set(int(-assignment[abs(x) - 1] * abs(x)) for x in clause_lits)

            unsatisfied.append(
                {
                    "clause_type": signature.type,
                    "cardinality": signature.card,
                    "literals": clause,
                    "assigned": assigned,
                    "objective_idx": obj_idx,
                    "clause_idx": int(clause_idx),
                }
            )
    return unsatisfied


def format_clause(clause_info: dict) -> str:
    """Format a clause info dict as a human-readable string."""
    ctype = clause_info["clause_type"]
    lits = sorted(list(clause_info["literals"]), key=lambda x: abs(x))
    assignment = sorted(list(clause_info["assigned"]), key=lambda x: abs(x))
    pad = max([len(str(x)) for x in lits] + [len(str(x)) for x in assignment])
    card = clause_info["cardinality"]

    lits_str = " ".join(f"{str(lit):>{pad}}" for lit in lits)
    assn_str = " ".join(f"{str(lit):>{pad}}" for lit in assignment)

    if ctype in ("card", "ek"):
        retstr = f"{ctype.upper():>4}({card:>2}): {lits_str}"
    else:
        retstr = f"{ctype.upper():>6}: {lits_str}"
    retstr += f"\n{' ':>9}: {assn_str}"
    return retstr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a solution against a SAT problem and report unsatisfied clauses"
    )
    parser.add_argument("problem_file", help="Path to the problem file (DIMACS/hybrid format)")
    parser.add_argument("solution_file", help="Path to .sol file containing solution literals")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show satisfied clause count too")

    args = parser.parse_args()

    # Load the problem
    print(f"Loading problem from: {args.problem_file}")
    formula = PBSATFormula(workers=1, compactify=False)
    formula.read_DIMACS(args.problem_file)

    print(f"Problem has {formula.n_var} variables and {formula.n_clause} clauses")

    # Load and parse solution
    print(f"Loading solution from: {args.solution_file}")
    solution_str = load_solution_file(args.solution_file)
    assignment = parse_solution(solution_str, formula)
    assigned_count = int(np.sum(assignment != 0))
    print(f"Solution assigns {assigned_count} variables")

    # Find unsatisfied clauses
    unsatisfied = get_unsatisfied_clauses(formula, assignment)

    # Report results
    if not unsatisfied:
        print("\n✓ All clauses are SATISFIED!")
    else:
        print(f"\n✗ Found {len(unsatisfied)} UNSATISFIED clause(s):\n")
        for i, clause_info in enumerate(unsatisfied, 1):
            print(f"{i}. {format_clause(clause_info)}")
            if args.verbose:
                print(f"      (objective {clause_info['objective_idx']}, clause {clause_info['clause_idx']})")

    if args.verbose:
        total_clauses = formula.n_clause
        sat_count = total_clauses - len(unsatisfied)
        print(f"\nSummary: {sat_count}/{total_clauses} clauses satisfied ({100 * sat_count / total_clauses:.2f}%)")

    return len(unsatisfied)


if __name__ == "__main__":
    sys.exit(main())
