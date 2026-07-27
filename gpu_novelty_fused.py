# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""
GPU-accelerated beam search SAT solver with fused inner loop.

This version keeps the entire beam search loop on GPU, only returning to CPU
for clause reweighting and progress reporting. Uses jax.lax.while_loop for
early termination support.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import sys

# from argparse import ArgumentParser as ArgParse
from argparse import SUPPRESS
from time import perf_counter as time
from typing import NamedTuple

from jsonargparse import ArgumentParser as ArgParse

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = " ".join(  # noqa: FLY002
    [
        "--xla_enable_fast_math=true",
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_enable_fast_min_max=true",
        "--xla_gpu_enable_cublaslt=true",
        "--xla_gpu_autotune_gemm_rtol=1e-6",
        "--xla_gpu_exhaustive_tiling_search=true",
        # "--xla_gpu_require_complete_aot_autotune_results=true",
    ]
)

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import io_callback
from jax.sharding import Mesh, NamedSharding
from tqdm.auto import tqdm

from boolean_whf import ClauseArrays, clause_type_ids
from samplers import SAMPLERS, sample_assignments
from sat_loader import PBSATFormula
from utils import (
    LOG_LEVELS,
    NoveltyConfig,
    get_gpu_l2_cache_size,
)
from var_mapper import VarMapper

logger = logging.getLogger(__name__)

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_memory_fitting_level", "O3")
jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_compiler_enable_remat_pass", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

print = functools.partial(print, flush=True)

VerifyFn: TypeAlias = Callable[[Array, Array], tuple[Array, Array]]

DEFAULT_INNER_ITERS = 10


# fmt: off
class BeamState(NamedTuple):
    """State carried through beam search iterations."""
    points: Array           # (batch_size, n_vars) current beam
    best_candidate: Array   # (n_vars,) best assignment seen
    best_unsat: Array       # () scalar, best unsat count
    raw_unsat: Array        # (n_expanded,) unsat counts from latest expansion
    clause_totals: Array    # (n_clauses,) accumulated for reweighting
    iter_count: Array       # () iteration counter
    rng_key: Array          # PRNG state
    done: Array             # () bool, early exit flag
    weights: Array          # (n_clauses,) clause weights
# fmt: on


def build_verifier(cls: tuple[ClauseArrays, ...]) -> VerifyFn:
    """Build a verification function that returns weighted scores and unsat mask."""

    def single_verifier(cl: ClauseArrays) -> Callable[[Array], Array]:
        lits = cl.lits
        sign = cl.sign
        mask = cl.mask
        cards = cl.cards
        types = cl.types
        clause_count = lits.shape[0]
        is_negated = sign < 0

        unsat_rules = {
            "xor": lambda x: jnp.sum(x, axis=-1, where=mask) % 2 == 0,
            "cnf": lambda x: ~jnp.any(x, axis=-1, where=mask),
            "eo": lambda x: jnp.sum(x, axis=-1, where=mask) != 1,
            "amo": lambda x: jnp.sum(x, axis=-1, where=mask) > 1,
            "nae": lambda x: ~(jnp.any(x, axis=-1, where=mask) & jnp.any(~x, axis=-1, where=mask)),
            "card": lambda x: jnp.where(
                cards < 0,
                jnp.sum(x, axis=-1, where=mask) >= jnp.abs(cards),
                jnp.sum(x, axis=-1, where=mask) < cards,
            ),
            "ek": lambda x: jnp.sum(x, axis=-1, where=mask) != cards,
        }

        def verify(x: Array) -> Array:
            clause_assigned = x[:, lits] ^ is_negated
            unsat = jnp.zeros((x.shape[0], clause_count), dtype=int)
            for clause_type, rule in unsat_rules.items():
                type_id = clause_type_ids[clause_type]
                unsat_clauses = rule(clause_assigned)
                unsat += jnp.where(types == type_id, unsat_clauses, 0)
                # unsat = unsat | jnp.where(types == type_id, unsat_clauses, unsat)
            return unsat

        return verify

    verifiers = [single_verifier(cl) for cl in cls]

    def combined_verify(x: Array, weights: Array) -> tuple[Array, Array]:
        all_unsat = [v(x) for v in verifiers]
        combined = jnp.concatenate(all_unsat, axis=-1)
        weighted_scores = jnp.sum(combined.astype(jnp.float32) * weights, axis=-1)
        return weighted_scores, combined

    return combined_verify


def get_mesh(devices: list) -> tuple[Mesh, NamedSharding]:
    """Create mesh and sharding for batch parallelism."""
    mesh = Mesh(np.array(devices).reshape((len(devices),)), ("batch",))
    batch_sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    return mesh, batch_sharding


def make_gpu_inner_loop(
    n_vars: int,
    n_clauses: int,
    batch_size: int,
    n_keep: int,
    n_cull: int,
    top_m: int,
    counting: bool,
    verifier: VerifyFn,
    flip_mask: Array,
    n_flip: int,
    # Single-prefix support (all points share one prefix)
    fixed_mask_1d: Array | None = None,
    prefix_bools_1d: Array | None = None,
    # Multi-prefix support (vmap over prefix groups)
    all_fixed_masks: Array | None = None,
    all_prefix_bools: Array | None = None,
    n_prefix: int = 0,
    sample_method: str = "bias",
):
    """Factory to create GPU inner loop with constants closed over.

    Prefix modes:
      - No prefix:      flip_mask is (n_vars, n_vars), n_flip == n_vars.
      - Single prefix:  flip_mask is (n_free, n_vars), n_flip == n_free.
                         fixed_mask_1d / prefix_bools_1d are (n_vars,).
      - Multi-prefix:   flip_mask is (n_prefix, max_n_free, n_vars), n_flip == max_n_free.
                         Padded zero-rows produce duplicate candidates (harmless waste
                         proportional to the gap between longest and shortest prefix).
                         Expand and select are vmapped over prefix groups — no
                         per-point prefix tracking needed.
    """

    n_expanded = batch_size * top_m
    single_prefix = fixed_mask_1d is not None
    multi_prefix = all_fixed_masks is not None

    # Per-group constants for multi-prefix
    if multi_prefix:
        assert all_fixed_masks is not None and all_prefix_bools is not None
        ppg = batch_size // n_prefix  # points per group
        n_cull_pg = n_cull // n_prefix
        n_keep_pg = ppg - n_cull_pg

    # Host-side solution collection
    collected_solutions: list[tuple[bool, ...]] = []

    def host_collect_solutions(potential_sols: np.ndarray, sol_indices: np.ndarray) -> None:
        """Host callback - extracts actual solutions using sentinel-based filtering."""
        for i, idx in enumerate(sol_indices):
            if idx >= 0:
                collected_solutions.append(tuple(potential_sols[i].tolist()))

    def get_solutions() -> list[tuple[bool, ...]]:
        return collected_solutions

    def clear_solutions() -> None:
        collected_solutions.clear()

    # ─── Expand / select for no-prefix and single-prefix ─────────────────

    def beam_expand(points: Array, weights: Array, rng_key: Array) -> tuple[Array, Array, Array]:
        """Expand all points to their best neighbors (no-prefix / single-prefix)."""
        keys = jax.random.split(rng_key, batch_size)

        def single_point_step(point: Array, key: Array) -> tuple[Array, Array, Array]:
            candidates = point ^ flip_mask  # (n_flip, n_vars)
            weighted_scores, unsat_masks = verifier(candidates, weights)
            unsat_counts = jnp.sum(unsat_masks, axis=-1)
            noise = jax.random.uniform(key, shape=(n_flip,), minval=0, maxval=0.5)
            noisy_scores = weighted_scores + noise

            if top_m > 1:
                top_idx = jnp.argsort(noisy_scores)[:top_m]
                return candidates[top_idx], unsat_counts[top_idx], unsat_masks[top_idx]
            else:
                best = jnp.argmin(noisy_scores)
                return candidates[best], unsat_counts[best], unsat_masks[best]

        neighbors, unsats, masks = jax.vmap(single_point_step)(points, keys)
        return neighbors.reshape(-1, n_vars), unsats.reshape(-1), masks.reshape(-1, n_clauses)

    def select_and_refill(expanded: Array, unsat_masks: Array, weights: Array, fill_key: Array) -> Array:
        """Select top candidates and optionally refill (no-prefix / single-prefix)."""
        scores = jnp.sum(unsat_masks.astype(jnp.float32) * weights, axis=-1)
        sorted_idx = jnp.argsort(scores)
        kept = expanded[sorted_idx[:n_keep]]

        if n_cull > 0:
            random_fill, _ = sample_assignments(fill_key, n_cull, n_vars, sample_method)
            random_fill = random_fill < 0
            if single_prefix:
                assert fixed_mask_1d is not None and prefix_bools_1d is not None
                random_fill = jnp.where(fixed_mask_1d, prefix_bools_1d, random_fill)
            return jnp.concatenate([kept, random_fill], axis=0)
        return kept

    # ─────────────────────────────────────────────────────────────────────

    def make_inner_loop(n_iters: int):
        """Create JIT-compiled inner loop for given iteration count.

        Weights are read dynamically from BeamState.weights — no recompilation
        needed when weights change between restarts.
        """

        def body_fn(state: BeamState) -> BeamState:
            weights = state.weights
            rng_key, step_key, fill_key = jax.random.split(state.rng_key, 3)

            if multi_prefix:
                # ─── Multi-prefix: double vmap (prefix_groups × points) ──
                assert all_fixed_masks is not None and all_prefix_bools is not None
                grouped = state.points.reshape(n_prefix, ppg, n_vars)
                group_step_keys = jax.random.split(step_key, n_prefix)
                group_fill_keys = jax.random.split(fill_key, n_prefix)

                def expand_group(
                    group_pts: Array,
                    group_mask: Array,
                    gkey: Array,
                ) -> tuple[Array, Array, Array]:
                    pkeys = jax.random.split(gkey, ppg)

                    def step(point: Array, key: Array) -> tuple[Array, Array, Array]:
                        candidates = point ^ group_mask  # (n_flip, n_vars)
                        w_scores, u_masks = verifier(candidates, weights)
                        u_counts = jnp.sum(u_masks, axis=-1)
                        noise = jax.random.uniform(key, shape=(n_flip,), minval=0, maxval=0.5)
                        noisy = w_scores + noise
                        if top_m > 1:
                            idx = jnp.argsort(noisy)[:top_m]
                            return candidates[idx], u_counts[idx], u_masks[idx]
                        else:
                            b = jnp.argmin(noisy)
                            return candidates[b], u_counts[b], u_masks[b]

                    nbrs, us, ms = jax.vmap(step)(group_pts, pkeys)
                    return nbrs.reshape(-1, n_vars), us.reshape(-1), ms.reshape(-1, n_clauses)

                expanded_g, unsats_g, masks_g = jax.vmap(expand_group)(
                    grouped,
                    flip_mask,
                    group_step_keys,
                )
                # expanded_g: (n_prefix, ppg*top_m, n_vars)
                # unsats_g:   (n_prefix, ppg*top_m)
                # masks_g:    (n_prefix, ppg*top_m, n_clauses)

                # Flatten for global operations
                expanded = expanded_g.reshape(-1, n_vars)
                unsats = unsats_g.reshape(-1)
                unsat_masks = masks_g.reshape(-1, n_clauses)

                # Accumulate
                new_clause_totals = state.clause_totals + unsat_masks.sum(axis=0).astype(jnp.float32)

                # Global best
                best_idx = jnp.argmin(unsats)
                iter_best = expanded[best_idx]
                iter_best_unsat = unsats[best_idx]
                is_better = iter_best_unsat < state.best_unsat
                new_best_candidate = jnp.where(is_better, iter_best, state.best_candidate)
                new_best_unsat = jnp.minimum(iter_best_unsat, state.best_unsat)

                # Per-group selection with prefix-aware refill
                def select_group(
                    g_exp: Array,
                    g_masks: Array,
                    g_fixed: Array,
                    g_bools: Array,
                    g_fkey: Array,
                ) -> Array:
                    scores = jnp.sum(g_masks.astype(jnp.float32) * weights, axis=-1)
                    sorted_idx = jnp.argsort(scores)
                    kept = g_exp[sorted_idx[:n_keep_pg]]
                    if n_cull_pg > 0:
                        fill, _ = sample_assignments(g_fkey, n_cull_pg, n_vars, sample_method)
                        fill = fill < 0
                        fill = jnp.where(g_fixed, g_bools, fill)
                        return jnp.concatenate([kept, fill], axis=0)
                    return kept

                selected_g = jax.vmap(select_group)(
                    expanded_g,
                    masks_g,
                    all_fixed_masks,
                    all_prefix_bools,
                    group_fill_keys,
                )
                new_points = selected_g.reshape(batch_size, n_vars)

            else:
                # ─── No-prefix / single-prefix ──────────────────────────
                expanded, unsats, unsat_masks = beam_expand(state.points, weights, step_key)

                new_clause_totals = state.clause_totals + unsat_masks.sum(axis=0).astype(jnp.float32)

                best_idx = jnp.argmin(unsats)
                iter_best = expanded[best_idx]
                iter_best_unsat = unsats[best_idx]
                is_better = iter_best_unsat < state.best_unsat
                new_best_candidate = jnp.where(is_better, iter_best, state.best_candidate)
                new_best_unsat = jnp.minimum(iter_best_unsat, state.best_unsat)

                new_points = select_and_refill(expanded, unsat_masks, weights, fill_key)

            # ─── Solution handling (shared) ────────────────────────────
            sol_mask = unsats == 0
            n_sols = sol_mask.sum()

            if counting:
                sol_indices = jnp.where(sol_mask, size=n_expanded, fill_value=-1)[0]
                potential_sols = expanded[jnp.maximum(sol_indices, 0)]
                io_callback(
                    host_collect_solutions,
                    (),
                    potential_sols,
                    sol_indices,
                    ordered=False,
                )
                new_done = jnp.array(False)
            else:
                new_done = n_sols > 0

            return BeamState(
                points=new_points,
                best_candidate=new_best_candidate,
                best_unsat=new_best_unsat,
                raw_unsat=unsats,
                clause_totals=new_clause_totals,
                iter_count=state.iter_count + 1,
                rng_key=rng_key,
                done=new_done,
                weights=weights,
            )

        def cond_fn(state: BeamState) -> Array:
            return (~state.done) & (state.iter_count < n_iters)

        @jax.jit
        def gpu_inner_loop(init_state: BeamState) -> BeamState:
            return jax.lax.while_loop(cond_fn, body_fn, init_state)

        return gpu_inner_loop

    return make_inner_loop, get_solutions, clear_solutions


def run_beam_search(
    config: NoveltyConfig,
    n_vars: int,
    n_clauses: int,
    cls: tuple[ClauseArrays, ...],
    prefixes: np.ndarray | None = None,
    *,
    var_mapper: VarMapper,
) -> float:
    """Run parallel beam search SAT solver with fused GPU inner loop."""

    runtime_common = config.runtime_common
    runtime_novelty = config.runtime_novelty
    optimiser_cfg = config.optimiser
    output_cfg = config.output_logging

    timeout = runtime_common.timeout_sec
    batch_size = runtime_novelty.beam_per_device
    n_devices = runtime_common.n_devices
    max_iters = optimiser_cfg.max_iters
    restart_thresh = runtime_common.restart_f
    sample_method = runtime_common.pt_sampler
    unsat_h = int(runtime_common.unsat_thresh * n_clauses) if runtime_common.unsat_thresh else 0
    rand_seed = runtime_common.rand_seed
    counting = runtime_common.counting
    top_m = runtime_novelty.top_m
    beta = runtime_novelty.beta
    weight_decay = runtime_common.weight_decay
    benchmark = runtime_common.benchmark
    binary_v = output_cfg.binary_v

    # Initialize devices and mesh
    devices = jax.devices("gpu")[:n_devices]
    mesh, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    seed = int(time()) if rand_seed else 42
    logger.info(f"seed={seed}, rand_seed={rand_seed}")
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)

    verifier = jax.jit(build_verifier(cls))
    weights = jnp.ones(n_clauses, dtype=jnp.float32)

    # ── Prefix handling ──────────────────────────────────────────────────
    n_prefix = prefixes.shape[0] if prefixes is not None else 0
    single_prefix = n_prefix == 1
    multi_prefix = n_prefix > 1
    ttfs = 0

    if single_prefix:
        assert prefixes is not None
        # Reduced flip_mask: only rows for free (unfixed) variables.
        free_indices = np.where(prefixes[0] == 0)[0]
        n_flip = len(free_indices)
        flip_mask = jnp.eye(n_vars, dtype=bool)[free_indices]  # (n_free, n_vars)
        fixed_mask_1d = jnp.array(prefixes[0] != 0)  # (n_vars,) True where fixed
        prefix_bools_1d = jnp.array(prefixes[0] < 0)  # (n_vars,) True where var=True
        logger.info(f"Single prefix: {n_flip} free vars (reduced from {n_vars})")
    elif multi_prefix:
        assert prefixes is not None
        # Per-prefix reduced flip masks, padded to max_n_free with zero-rows.
        # Zero-row XOR produces a duplicate of the original point (harmless waste).
        all_free_indices = [np.where(prefixes[k] == 0)[0] for k in range(n_prefix)]
        n_free_per_prefix = [len(fi) for fi in all_free_indices]
        max_n_free = max(n_free_per_prefix)
        n_flip = max_n_free
        eye = np.eye(n_vars, dtype=bool)
        padded_masks = np.zeros((n_prefix, max_n_free, n_vars), dtype=bool)
        for k, fi in enumerate(all_free_indices):
            padded_masks[k, : len(fi)] = eye[fi]
        flip_mask = jnp.array(padded_masks)  # (n_prefix, max_n_free, n_vars)
        all_fixed_masks = jnp.array(prefixes != 0)  # (n_prefix, n_vars)
        all_prefix_bools = jnp.array(prefixes < 0)  # (n_prefix, n_vars)
        waste_pct = 100 * (1 - sum(n_free_per_prefix) / (n_prefix * max_n_free))
        logger.info(
            f"Multi-prefix: {n_prefix} vectors, max_n_free={max_n_free}/{n_vars}, padding waste={waste_pct:.1f}%"
        )
    else:
        flip_mask = jnp.eye(n_vars, dtype=bool)
        n_flip = n_vars
    # ─────────────────────────────────────────────────────────────────────

    # Determine inner loop size
    inner_iters = restart_thresh if restart_thresh > 0 else DEFAULT_INNER_ITERS

    # Adjust max_iters to multiple of inner_iters
    if max_iters > 0 and max_iters % inner_iters != 0:
        old = max_iters
        max_iters = ((max_iters // inner_iters) + 1) * inner_iters
        logger.info(f"Adjusted max_iters {old} -> {max_iters} (multiple of {inner_iters})")

    # Auto-select batch size
    if batch_size == -1:
        logger.info("Guessing optimal batch size")
        l2_cache_size = get_gpu_l2_cache_size(devices[0])
        if l2_cache_size is not None:
            gpu_mem_target = int(l2_cache_size * 0.90) * n_devices * 2
            logger.info(f"Targeting total cache: {l2_cache_size / (1024 * 1024):.1f} MB per GPU")
        else:
            gpu_mem_target = devices[0].memory_stats()["bytes_limit"] * 0.01
            logger.info("Cache size unknown, using 1% VRAM heuristic")

        dtype_sz = jnp.dtype(cls[0].lits.dtype).itemsize
        all_obj_sz = sum([np.prod([*ca.lits.shape, dtype_sz]) for ca in cls])
        flip_sz = (n_clauses * n_flip + n_flip * n_vars) * jnp.dtype(bool).itemsize
        batch_size = int(np.floor((gpu_mem_target - all_obj_sz) / flip_sz)) * n_devices
        logger.info(
            f"Batch size: {batch_size} (n_flip={n_flip}, consumed by clauses: {all_obj_sz / (1024 * 1024):.1f} MB)"
        )

    # Batch alignment: must be divisible by n_devices and n_prefix (if any).
    alignment = math.lcm(max(n_prefix, 1), n_devices)
    batch_size = (batch_size // alignment) * alignment
    batch_size = max(batch_size, alignment)

    # Recompute n_cull/n_keep after final batch_size is settled.
    n_cull = int(batch_size * beta)
    n_keep = batch_size - n_cull
    logger.info(
        f"Throwing away {n_cull} points after each batch of {batch_size} (before flip) points (keeping {n_keep})"
    )

    # Create GPU loop factory
    make_inner_loop, get_solutions, clear_solutions = make_gpu_inner_loop(
        n_vars=n_vars,
        n_clauses=n_clauses,
        batch_size=batch_size,
        n_keep=n_keep,
        n_cull=n_cull,
        top_m=top_m,
        counting=counting,
        verifier=verifier,
        flip_mask=flip_mask,
        n_flip=n_flip,
        # Single-prefix constants (None when unused)
        fixed_mask_1d=fixed_mask_1d if single_prefix else None,
        prefix_bools_1d=prefix_bools_1d if single_prefix else None,
        # Multi-prefix tables (None when unused)
        all_fixed_masks=all_fixed_masks if multi_prefix else None,
        all_prefix_bools=all_prefix_bools if multi_prefix else None,
        n_prefix=n_prefix,
        sample_method=sample_method,
    )

    # Initialize points
    points, _ = sample_assignments(init_key, batch_size * top_m, n_vars, sample_method)
    points = points < 0

    # Apply prefix values to initial points
    if single_prefix:
        points = jnp.where(fixed_mask_1d, prefix_bools_1d, points)
    elif multi_prefix:
        # Replicate prefix bools across batch in contiguous groups
        rep = (batch_size * top_m) // n_prefix
        replicated_mask = jnp.repeat(all_fixed_masks, rep, axis=0)
        replicated_bools = jnp.repeat(all_prefix_bools, rep, axis=0)
        points = jnp.where(replicated_mask, replicated_bools, points)

    if top_m > 1:
        if multi_prefix:
            # Per-group selection to maintain structural prefix grouping
            ppg_init = (batch_size * top_m) // n_prefix
            ppg_final = batch_size // n_prefix
            grouped = points.reshape(n_prefix, ppg_init, n_vars)

            def _init_select_group(group_pts: Array) -> Array:
                scores, _ = verifier(group_pts, weights)
                idx = jnp.argsort(scores)[:ppg_final]
                return group_pts[idx]

            points = jax.vmap(_init_select_group)(grouped).reshape(batch_size, n_vars)
        else:
            weighted_scores, _ = verifier(points, weights)
            top_indices = jnp.argsort(weighted_scores)[:batch_size]
            points = points[top_indices]
    points = jax.device_put(points, batch_sharding)

    # Initial state
    state = BeamState(
        points=points,
        best_candidate=points[0],
        best_unsat=jnp.array(n_clauses, dtype=jnp.int32),
        raw_unsat=jnp.full((batch_size * top_m,), n_clauses, dtype=jnp.int32),
        clause_totals=jnp.zeros(n_clauses, dtype=jnp.float32),
        iter_count=jnp.array(0, dtype=jnp.int32),
        rng_key=rng_key,
        done=jnp.array(False),
        weights=weights,
    )

    total_iters: int = 0
    batches_done: int = 0
    restart_ct: int = 0
    all_unsats: list[int] = []
    best_assignment: tuple[bool, ...]
    best_unsat_host: int = n_clauses
    found_sol: bool = False

    pbar = None
    if not benchmark:
        pbar = tqdm(
            total=timeout,
            desc="iter 0 (best=undef)",
            bar_format="{l_bar}{bar}|{elapsed}/{total_fmt} {postfix}",
        )

    t0 = time()
    last_update = t0
    ttfs = 0

    # Compile inner loop once (weights are dynamic via BeamState)
    gpu_loop = make_inner_loop(inner_iters)

    while time() - t0 < timeout:
        if max_iters > 0 and total_iters >= max_iters:
            break

        # Reset per-batch state (keep points and best)
        state = state._replace(
            clause_totals=jnp.zeros(n_clauses, dtype=jnp.float32),
            iter_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
        )

        # ===== GPU INNER LOOP =====
        state = gpu_loop(state)
        jax.block_until_ready(state)

        total_iters += inner_iters
        batches_done += 1

        # Reuse UNSAT values already produced in the GPU loop state.
        batch_unsats = np.array(state.raw_unsat).astype(int).flatten().tolist()
        all_unsats.extend(batch_unsats)

        # Track best assignment
        best_unsat_val = int(state.best_unsat)
        if best_unsat_val < best_unsat_host:
            best_assignment = tuple(np.array(state.best_candidate).tolist())
            best_unsat_host = best_unsat_val

        if best_unsat_val == 0 and not found_sol:
            ttfs = time() - t0

        # Threshold stop for MAXSAT-style runs (implicit early stop when threshold is set).
        if unsat_h and best_unsat_val <= unsat_h:
            if not found_sol:
                ttfs = time() - t0
            break

        # Check for early exit (non-counting mode)
        if state.done and not counting:
            best_assignment = tuple(np.array(state.best_candidate).tolist())
            break

        # Report solutions found (counting mode)
        n_found = len(get_solutions())
        if counting and n_found > 0 and not benchmark:
            logger.info(f"Solutions so far: {n_found}")

        # Reweight clauses
        if restart_thresh > 0:
            clause_totals = np.array(state.clause_totals)
            if np.any(clause_totals):
                worst = max(float(clause_totals.max()), 1.0)
                old_w = np.array(state.weights)
                new_w = weight_decay * old_w + (1 - weight_decay) * clause_totals / worst
                state = state._replace(weights=jnp.array(new_w, dtype=jnp.float32))
            restart_ct += 1

        # Progress update
        now = time()
        if pbar is not None and now - last_update > 0.5:
            elapsed = now - t0
            pbar.n = min(elapsed, timeout)
            pbstr = f"{total_iters % inner_iters}/{inner_iters}" if restart_thresh else f"{total_iters}"
            pbar.set_description(f"restart {restart_ct} ({pbstr} -- best={best_unsat_val})")
            pbar.set_postfix_str(f"{total_iters / elapsed:.1f} it/s")
            pbar.refresh()
            last_update = now

    if pbar is not None:
        pbar.close()
    solve_time = time() - t0

    # Final output
    best_unsat_val = min(int(state.best_unsat), best_unsat_host)
    all_sols: dict[tuple[bool, ...], int] = {}
    first_sol: tuple[bool, ...] | None = None

    if counting:
        for sol in get_solutions():
            all_sols[sol] = all_sols.get(sol, 0) + 1
        if all_sols:
            first_sol = next(iter(all_sols.keys()))
    elif best_unsat_val == 0:
        first_sol = best_assignment
        all_sols[first_sol] = 1

    sol_info = "GPU Beam Search (Greedy Novelty (Bit Flip)) SAT"
    print(f"c {'-' * len(sol_info)}")
    print(f"c {sol_info}")
    print(f"c {'-' * len(sol_info)}")

    if len(all_sols):
        print("s SATISFIABLE")
        for sol_i, sol in enumerate(all_sols.keys()):
            if not counting and first_sol != sol:
                continue
            sol_str = var_mapper.assn_str(sol, binary=binary_v)
            if counting:
                logger.info(f"{sol_i + 1}: v {sol_str} 0")
                if not sol_i:
                    print("c ENUMERATING - example solution (of possibly many):")
                    print(f"v {sol_str} 0")
            elif first_sol == sol:
                print(f"v {sol_str} 0")
                break
    else:
        assign_str = var_mapper.assn_str(best_assignment, binary=binary_v, inc_zero=True)
        print("s UNKNOWN")
        print("c Best found MAX-SAT assignment (zero energy variables omitted or -):")
        print(f"o {best_unsat_val}")
        print(f"v {assign_str} 0")

    print(f"c {'-' * len(sol_info)}")

    if logger.isEnabledFor(logging.INFO):
        if ttfs:
            logger.info(f"X-TTFS {ttfs}")
        logger.info("START EXP")
        logger.info(f"X-CLAUSES {n_clauses}")
        logger.info(f"X-VARS {n_vars}")
        logger.info(f"X-GPU {n_devices}")
        logger.info(f"X-PPBATCH {batch_size}")
        logger.info(f"X-PPGPUBATCH {batch_size // n_devices} PPBATCH/GPU")
        logger.info(f"X-BATCHES {batches_done}")
        logger.info(f"X-PTOTAL {batches_done * batch_size} BATCHES*PPBATCH")
        logger.info(f"X-LOOP {solve_time}")
        if len(all_sols):
            logger.info(f"X-SOLS {sum(all_sols.values())}")
            logger.info(f"X-UQSOLS {len(all_sols.keys())}")
        else:
            logger.info("X-SOLS 0")
            logger.info("X-UQSOLS 0")
        logger.info(f"X-RAWITER {[total_iters]}")
        logger.info(f"X-RAWUNSAT {all_unsats}")
        logger.info(f"X-BEAM {batch_size}")
        logger.info("END EXP")

    return solve_time


# =============================================================================
# Entry Point
# =============================================================================


def main(problem_file: str, config: NoveltyConfig) -> None:
    """Main entry point."""
    if not problem_file:
        raise ValueError("No problem file specified")

    stamp1 = time()
    sat_parser = PBSATFormula(
        workers=4,
        n_devices=config.runtime_common.n_devices,
        disk_cache=config.invocation.disk_cache,
        file=problem_file,
    )
    stamp2 = time()
    read_time = stamp2 - stamp1

    cls = tuple(obj.clauses for obj in sat_parser.process_clauses_to_array())
    n_var = sat_parser.n_var
    n_clause = sat_parser.n_clause
    stamp1 = time()
    process_time = stamp1 - stamp2

    # Process prefixes (includes unit literals from DIMACS parsing).
    prefixes: np.ndarray | None = None
    if config.invocation.prefix_file or sat_parser.unit_prefix:
        prefixes = sat_parser.process_prefix(config.invocation.prefix_file)
        assert prefixes is not None
        n_prefix = prefixes.shape[0]
        n_fixed = np.count_nonzero(prefixes[0]) if n_prefix == 1 else [np.count_nonzero(p) for p in prefixes]
        logger.info(f"Prefix: {n_prefix} vector(s), fixed vars: {n_fixed} / {n_var}")

    t_solve = run_beam_search(
        config,
        n_var,
        n_clause,
        cls,
        prefixes=prefixes,
        var_mapper=sat_parser.var_mapper,
    )

    logger.info(f"Time reading input: {read_time}")
    logger.info(f"Time processing to Arrays: {process_time}")
    logger.info(f"Time spent solving: {t_solve}")


if __name__ == "__main__":
    n_devices = len(jax.devices("gpu"))
    logger.info(jax.devices("gpu"))

    parser = ArgParse(description="GPU Beam Search SAT Solver (Fused)")
    parser.add_argument("problem_file", type=str, help="Input file (.cnf, .hybrid, .opb)")
    parser.add_class_arguments(NoveltyConfig, nested_key="config")
    parser.set_defaults(
        **{
            "config.runtime_common.n_devices": n_devices,
            "config.optimiser.name": "novelty",
            "config.optimiser.max_iters": 0,
        }
    )
    parser.link_arguments(
        "config.runtime_common.benchmark",
        "config.runtime_common.progress_enabled",
        compute_fn=lambda benchmark: not benchmark,
        apply_on="parse",
    )

    def make_option_group(title: str, section: str):
        group = parser.add_argument_group(title)

        def add_opt(*flags: str, field: str, **kwargs) -> None:
            kwargs.setdefault("default", SUPPRESS)
            group.add_argument(*flags, dest=f"config.{section}.{field}", **kwargs)

        return add_opt

    # Backward-compatible short aliases grouped by config sections.
    io_opts = make_option_group("IO Aliases", "invocation")
    io_opts("--cache", type=str, field="disk_cache", help="Disk cache for FFT matrices")
    io_opts("-p", "--prefix", type=str, field="prefix_file", help="Prefix file (fixed variable assignments)")

    runtime_common_opts = make_option_group("Runtime Common Aliases", "runtime_common")
    runtime_common_opts("-t", "--timeout", type=int, field="timeout_sec", help="Timeout in seconds")
    runtime_common_opts("-g", "--gpus", type=int, field="n_devices", help="Number of GPUs")
    runtime_common_opts("-c", "--counting", action="store_true", field="counting", help="Count solutions mode")
    runtime_common_opts("-s", "--seed", action="store_true", field="rand_seed", help="Random seed from time")
    runtime_common_opts("-m", "--sampler", type=str, field="pt_sampler", choices=SAMPLERS, help="Initial point sampler")
    runtime_common_opts("-r", "--restart_f", type=int, field="restart_f", help="Batches before reweighting (0 = never)")
    runtime_common_opts("-a", "--alpha", type=float, field="weight_decay", help="Weight decay (0.0-1.0)")
    runtime_common_opts("-e", "--benchmark", action="store_true", field="benchmark", help="Benchmarking (less output)")
    runtime_common_opts("--progress", action="store_false", field="benchmark", help="Display progress stats")
    runtime_common_opts("-u", "--unsat_thresh", type=float, field="unsat_thresh", help="MAXSAT - #UNSAT stop threshold")

    runtime_novelty_opts = make_option_group("Runtime Novelty Aliases", "runtime_novelty")
    runtime_novelty_opts("-b", "--beam", type=int, field="beam_per_device", help="Beam width (-1 for auto)")
    runtime_novelty_opts("-m", "--top-m", type=int, field="top_m", help="Top m neighbors per point")
    runtime_novelty_opts("--beta", type=float, field="beta", help="Cull fraction (0.0-1.0)")

    optimiser_opts = make_option_group("Optimiser Aliases", "optimiser")
    optimiser_opts("-i", "--iters", type=int, field="max_iters", help="Max iterations (0=unlimited)")

    output_opts = make_option_group("Output Aliases", "output_logging")
    output_opts("-d", "--debug", choices=LOG_LEVELS, field="debug_level", help=f"Set logging level ({LOG_LEVELS})")
    output_opts("--binary_v", action="store_true", field="binary_v", help="Short form solution string")
    output_opts("--stdout_log", action="store_true", field="stdout_log", help="Logs output to stdout instead of stderr")

    args = parser.parse_args()
    instantiated = parser.instantiate(args)
    config = instantiated.config
    config.runtime_common.n_devices = config.runtime_common.n_devices or n_devices

    logging.basicConfig(
        level=getattr(logging, config.output_logging.debug_level.upper()),
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        format="c %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout if config.output_logging.stdout_log else sys.stderr),
            # Optional: logging.FileHandler('sat_loader.log')  # Also log to file
        ],
    )

    main(args.problem_file, config)

    # flip in all direction * and * no flip.
    # keep some worst as well?
