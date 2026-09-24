#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""
Verification harness for the XOR RREF projection and its interaction with prefixes.

Usage:
    python check_xor_rref.py <problem_file> [--trials N] [--cube K] [--strict]

Example:
    python check_xor_rref.py 16b_3t_lfsr_100.pbo
    python check_xor_rref.py tests/sha1/preimage/r20/sha1sat_preimage_r20_hb10_seed123123_xor.cnf

The RREF reparameterisation exists to make every XOR clause satisfied *by construction*:
the projector derives each dependent variable from the free ones, so any assignment it
produces should have zero unsatisfied XOR clauses regardless of where the free variables
sit. This script checks that invariant directly, against an oracle that evaluates the
original XOR clauses and shares no code with the reduction.

It runs the production path -- `seq_eval_verify`, whose aux output *is* the projected
`x_eval` the optimiser actually sees -- so the checks cannot drift from what the solver does.

Checks, by prefix mode:

    none            no fixed variables. Only applies to instances with no unit literals:
                    once units are substituted out of the XOR system the RREF is built
                    relative to them being fixed, and every production prefix includes them.
    units           the problem's implied unit literals (including any derived by
                    propagation), exactly as a standalone run applies them. The baseline
                    invariant; must hold.
    units+free      units plus randomly fixed RREF-free variables. Must hold: a fixed free
                    variable already enters the projector's row parity, so dependents are
                    derived consistently with it.
    units+dep       units plus randomly fixed RREF-dependent variables. KNOWN BROKEN before
                    plan step E7: the RREF was built without the cube, so a fixed dependent
                    keeps its prefix value and its row is no longer guaranteed.
    units+cube      units plus a random cube over arbitrary variables, i.e. what the Dagster
                    worker receives.

Exit status is non-zero if a required check fails. KNOWN-BROKEN checks are reported but
not fatal unless --strict is given; once E7 lands, run with --strict and they must pass.
"""
# ruff: disable[E402]
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

fparent = Path(__file__).resolve().parent
if fparent == Path.cwd():
    sys.path.insert(1, os.path.abspath("../"))
else:
    sys.path.insert(1, str(fparent.parent))

import numpy as np
from numpy.typing import NDArray

from boolean_whf import clause_type_ids
from sat_loader import PBSATFormula
from solvers.optimisers import build_eval_verify, seq_eval_verify
from xor_rref import XorRREFMetadata, preprocess_xor_system, unit_bits_from_literals

# ruff: enable[E402]

REQUIRED = "required"
KNOWN_BROKEN = "known-broken-until-E7"


def xor_rows_from_clause_sets(xor_clause_sets) -> list[list[int]]:
    """Flatten the XOR clause sets the RREF is built from into literal lists."""
    return [[int(lit) for lit in clause] for clause_set in xor_clause_sets for clause in clause_set]


def oracle_unsat_rows(x: NDArray, rows: list[list[int]]) -> NDArray:
    """Independent XOR evaluator. Returns a bool array, True where the row is UNSAT.

    Mirrors the semantics in UNSAT_RULES["xor"]: with `assignment = sign(lit) * x[|lit|-1]`,
    a XOR clause is satisfied exactly when an odd number of its entries are negative.
    Implemented here from the raw literals so it shares no code with the reduction.
    """
    unsat = np.zeros(len(rows), dtype=bool)
    for i, lits in enumerate(rows):
        neg = 0
        for lit in lits:
            val = x[abs(lit) - 1] if lit > 0 else -x[abs(lit) - 1]
            if val < 0:
                neg += 1
        unsat[i] = (neg % 2) == 0
    return unsat


def rref_support_form(meta: XorRREFMetadata) -> tuple[NDArray, NDArray, int]:
    """The metadata's padded (n_dep, K) variable ids and mask, plus the true max row weight."""
    row_vars = np.asarray(meta.row_vars).astype(np.int64)
    row_mask = np.asarray(meta.row_mask)
    width = int(row_mask.sum(axis=1).max()) if row_mask.size else 0
    return row_vars, row_mask, width


def oracle_unsat_rref_rows(
    x: NDArray, dep: NDArray, row_vars: NDArray, row_mask: NDArray, b_final: NDArray
) -> NDArray:
    """Which *RREF* rows the assignment violates, as opposed to which original clauses.

    Row i asserts x[dep_i] = (1 - 2*b_i) * prod(the free variables in row i). The two counts
    differ: every original clause is a GF(2) combination of RREF rows, so one violated row
    shows up in every original clause whose expansion includes it an odd number of times.
    That amplification is why the clause count can exceed the number of violated rows.
    """
    if dep.size == 0:
        return np.zeros(0, dtype=bool)
    negatives = ((x[row_vars] < 0) & row_mask).sum(axis=1)
    derived_negative = ((negatives + b_final) % 2) == 1
    return derived_negative != (x[dep] < 0)


def xor_slice_of_verifier(objs) -> NDArray:
    """Boolean mask selecting the XOR entries of the concatenated verifier output."""
    xor_id = clause_type_ids["xor"]
    parts = []
    for obj in objs:
        types = np.asarray(obj.clauses.types).reshape(-1)
        n = obj.clauses.lits.shape[0]
        if types.size == n:
            parts.append(types == xor_id)
        else:  # homogeneous objective carrying a single type id
            parts.append(np.full(n, bool(np.all(types == xor_id))))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def check_metadata(meta: XorRREFMetadata, rows: list[list[int]], resolved: set[int]) -> list[str]:
    """Structural invariants of the metadata itself.

    `resolved` are variables unit propagation substituted out of the system; they are neither
    dependent nor free, so the covering invariant is dep u free u resolved == XOR fragment.
    """
    problems = []
    dep = np.asarray(meta.dependent_indices)
    free = np.asarray(meta.free_indices)
    xor_vars = {abs(lit) - 1 for lits in rows for lit in lits}

    if dep.size != np.unique(dep).size:
        problems.append(f"dependent_indices has duplicates ({dep.size - np.unique(dep).size})")
    if np.intersect1d(dep, free).size:
        problems.append(f"dependent and free sets overlap ({np.intersect1d(dep, free).size} vars)")
    covered = set(dep.tolist()) | set(free.tolist()) | resolved
    if not covered >= xor_vars:
        problems.append(f"dep u free u resolved misses {len(xor_vars - covered)} XOR variables")
    if set(dep.tolist()) & resolved or set(free.tolist()) & resolved:
        problems.append("a propagated-away variable is still dependent or free")
    row_vars, row_mask = np.asarray(meta.row_vars), np.asarray(meta.row_mask)
    if row_vars.shape != row_mask.shape or row_vars.shape[0] != dep.size:
        problems.append(f"row_vars {row_vars.shape} / row_mask {row_mask.shape} vs n_dep {dep.size}")
    elif row_mask.size and not np.isin(row_vars[row_mask], free).all():
        problems.append("a row references a variable that is not free")
    elif row_mask.size and (row_mask.sum(axis=1) == 0).any():
        problems.append(f"{int((row_mask.sum(axis=1) == 0).sum())} rows have empty support (unhoisted forced rows)")
    if np.asarray(meta.b_final).shape != (dep.size,):
        problems.append(f"b_final is {np.asarray(meta.b_final).shape}, expected {(dep.size,)}")
    return problems


def build_prefix(
    mode: str,
    formula: PBSATFormula,
    meta: XorRREFMetadata,
    n_var: int,
    rng: np.random.Generator,
    cube_size: int,
) -> NDArray | None:
    """Return a prefix vector in {-1, 0, +1}, or None if the mode does not apply here.

    Every mode except `none` includes the unit literals, because every prefix in production
    does: `process_prefix` returns them even with no prefix file, and `process_prefix_line`
    merges them into each cube. Once unit propagation substitutes those variables out of the
    XOR system (E3), the RREF is built *relative to* them being fixed, so a run with no fixed
    variables is not a configuration the solver can be in -- `none` only applies to instances
    that have no unit literals at all.
    """
    dep = np.asarray(meta.dependent_indices)
    free = np.asarray(meta.free_indices)

    if mode == "none":
        if formula.unit_prefix:
            return None
        return np.zeros(n_var, dtype=np.int8)

    # units only; process_prefix_line with an empty cube merges the unit prefix
    base = np.asarray(formula.process_prefix_line([]), dtype=np.int8)
    if not formula.unit_prefix and mode == "units":
        return None
    if mode == "units":
        return base

    if mode in ("units+free", "units+dep"):
        pool = free if mode == "units+free" else dep
        if pool.size == 0:
            return None
        k = min(16, pool.size)
        picked = rng.choice(pool, size=k, replace=False)
        base[picked] = rng.choice(np.array([-1, 1], dtype=np.int8), size=k)
        return base

    if mode == "units+cube":
        unit_vars = {abs(lit) - 1 for lit in formula.unit_prefix}
        candidates = np.array([v for v in range(n_var) if v not in unit_vars], dtype=np.int64)
        if candidates.size == 0:
            return None
        k = min(cube_size, candidates.size)
        picked = rng.choice(candidates, size=k, replace=False)
        signs = rng.choice(np.array([-1, 1]), size=k)
        cube = [int(sign) * (int(var) + 1) for var, sign in zip(picked, signs)]
        # process_prefix_line merges unit_prefix and raises UnsatError on conflict.
        return np.asarray(formula.process_prefix_line([str(lit) for lit in cube]), dtype=np.int8)

    raise ValueError(f"unknown prefix mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the XOR RREF projection against an independent GF(2) oracle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("problem_file", help="Problem instance path")
    parser.add_argument("--trials", type=int, default=8, help="Random assignments per mode (default 8)")
    parser.add_argument("--cube", type=int, default=8, help="Cube literals for units+cube (default 8)")
    parser.add_argument("--seed", type=int, default=20260904, help="RNG seed")
    parser.add_argument("--strict", action="store_true", help="Treat known-broken checks as failures")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    formula = PBSATFormula(
        workers=1, n_devices=1, disk_cache="", file=args.problem_file, compactify=False
    )
    objs = formula.process_clauses_to_array()
    rows = xor_rows_from_clause_sets(formula.xor_clause_sets)
    if not rows:
        print(f"{args.problem_file}: no XOR clauses; nothing to check")
        return 0

    # Mirror prepare_problem: full-formula propagation first, then the XOR preprocessing.
    from propagation import propagate_units

    file_units = len(formula.unit_prefix)
    propagated = propagate_units(formula.clause_sets, formula.unit_prefix)
    if propagated.unsat:
        print(f"{args.problem_file}: formula is inconsistent under unit propagation (instance is UNSAT)")
        return 0
    formula.unit_prefix |= propagated.derived
    seed_bits = unit_bits_from_literals(formula.unit_prefix)
    xor_result = preprocess_xor_system(formula.xor_clause_sets, seed_bits)
    if xor_result.unsat:
        print(f"{args.problem_file}: XOR system is inconsistent (instance is UNSAT)")
        return 0
    if xor_result.derived:
        # Mirror what prepare_problem does, so the prefix modes below see the same unit set.
        from xor_rref import literals_from_unit_bits

        formula.unit_prefix |= set(literals_from_unit_bits(xor_result.derived))
    meta = xor_result.meta
    if meta is None:
        print(f"{args.problem_file}: XOR system fully resolved by unit propagation; no projection")
        return 0
    resolved = {abs(lit) - 1 for lit in formula.unit_prefix} - set(
        np.asarray(meta.dependent_indices).tolist()
    ) - set(np.asarray(meta.free_indices).tolist())

    n_var = formula.n_var
    dep = np.asarray(meta.dependent_indices)
    free = np.asarray(meta.free_indices)
    unit_vars = {abs(lit) - 1 for lit in formula.unit_prefix}
    units_on_dep = len(unit_vars & set(dep.tolist()))

    print(f"instance      : {args.problem_file}")
    print(f"variables     : {n_var}   clauses: {formula.n_clause}")
    print(f"XOR rows      : {len(rows)}   dependent: {dep.size}   free: {free.size}")
    print(
        f"unit literals : {len(formula.unit_prefix)} ({file_units} from the file, "
        f"{len(propagated.derived)} propagated, {len(xor_result.derived)} from XOR)   "
        f"of which land on a dependent: {units_on_dep}"
    )

    row_vars, row_mask, width = rref_support_form(meta)
    b_final = np.asarray(meta.b_final).astype(np.int64)
    weights_per_row = row_mask.sum(axis=1)
    dense_bytes = int(dep.size) * int(free.size) * 4  # what the old (n_dep, n_free) f32 block cost
    padded_bytes = int(np.asarray(meta.row_vars).nbytes + np.asarray(meta.row_mask).nbytes)
    print(
        f"row weight    : mean {weights_per_row.mean():.1f}  max {int(weights_per_row.max()) if dep.size else 0}"
        f"  (K for E4)   nnz {int(weights_per_row.sum())}"
    )
    print(
        f"free_part     : old dense {dense_bytes / 1e6:.1f} MB   padded (n_dep, K) "
        f"{padded_bytes / 1e6:.1f} MB   ratio {dense_bytes / max(padded_bytes, 1):.0f}x"
    )

    failures = 0

    meta_problems = check_metadata(meta, rows, resolved)
    print("\nmetadata invariants:", "OK" if not meta_problems else "FAIL")
    for problem in meta_problems:
        print(f"    - {problem}")
    failures += len(meta_problems)

    eval_fns, verify_fns = build_eval_verify(objs, False)
    seq_evals, seq_verifies = seq_eval_verify(eval_fns, verify_fns, xor_rref_meta=meta)
    weights = tuple(jnp.ones((obj.clauses.lits.shape[0],), dtype=float) for obj in objs)
    xor_mask = xor_slice_of_verifier(objs)

    modes = [
        ("none", REQUIRED),
        ("units", REQUIRED),
        ("units+free", REQUIRED),
        ("units+dep", KNOWN_BROKEN),
        ("units+cube", KNOWN_BROKEN),
    ]

    print(
        f"\n{'prefix mode':<16} {'fixed':>7} {'on dep':>7} {'RREF rows':>10} "
        f"{'clauses (max/mean)':>20}  {'verifier':>9}  result"
    )
    print("-" * 88)

    for mode, severity in modes:
        rng = np.random.default_rng(args.seed)
        try:
            prefix = build_prefix(mode, formula, meta, n_var, rng, args.cube)
        except Exception as err:  # a conflicting cube is a legitimate skip, not a failure
            print(f"{mode:<16} {'-':>7} {'-':>7} {'-':>22}  {'-':>9}  SKIP ({type(err).__name__}: {err})")
            continue
        if prefix is None:
            print(f"{mode:<16} {'-':>7} {'-':>7} {'-':>22}  {'-':>9}  SKIP (not applicable)")
            continue

        fixed_np = prefix != 0
        n_fixed = int(fixed_np.sum())
        n_fixed_dep = int(fixed_np[dep].sum()) if dep.size else 0
        fixed_vars = jnp.asarray(fixed_np)

        counts = []
        row_counts = []
        verifier_agrees = True
        for _ in range(args.trials):
            x = rng.uniform(-1.0, 1.0, size=n_var).astype(np.float32)
            x = np.where(fixed_np, prefix.astype(np.float32), x)
            _, (x_eval, _) = seq_evals(jnp.asarray(x), fixed_vars, weights)
            x_host = np.asarray(x_eval)

            oracle = oracle_unsat_rows(x_host, rows)
            counts.append(int(oracle.sum()))
            row_counts.append(int(oracle_unsat_rref_rows(x_host, dep, row_vars, row_mask, b_final).sum()))

            unsat_all = np.asarray(seq_verifies(x_eval)).reshape(-1)
            if int(unsat_all[xor_mask].sum()) != int(oracle.sum()):
                verifier_agrees = False

        worst, mean = max(counts), sum(counts) / len(counts)
        worst_rows = max(row_counts)
        ok = worst == 0
        if ok and verifier_agrees:
            result = "PASS"
        elif severity is REQUIRED or args.strict:
            result = "FAIL"
            failures += 1
        else:
            result = "KNOWN"
        if not verifier_agrees:
            result += " (verifier disagrees with oracle)"

        print(
            f"{mode:<16} {n_fixed:>7} {n_fixed_dep:>7} {worst_rows:>10} {worst:>8} / {mean:>9.1f}  "
            f"{'agrees' if verifier_agrees else 'DIFFERS':>9}  {result}"
        )

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("all required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
