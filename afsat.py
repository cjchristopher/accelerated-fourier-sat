# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# ruff: disable[E402]
from __future__ import annotations

import functools
import logging
import math
import os
import sys
import shutil
from argparse import ArgumentParser as ArgParse
from collections import Counter, defaultdict
from contextlib import nullcontext
from time import perf_counter as time
from typing import TypeAlias, overload

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"  # Use full memory allocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = " ".join(
    [
        "--xla_enable_fast_math=true",
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_enable_fast_min_max=true",
        "--xla_gpu_enable_cublaslt=true",
        "--xla_gpu_autotune_gemm_rtol=1e-6",
        "--xla_gpu_exhaustive_tiling_search=true",
        # "--xla_gpu_deterministic_ops=true",
        # "--xla_gpu_require_complete_aot_autotune_results=true",
    ]
)
# Single-host, multi-device computation on NVIDIA GPUs
os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)

import jax
import jax.numpy as jnp
# TODO: Disable x64 when clauses are short enough - find this limit.
jax.config.update("jax_platform_name", "gpu")  # gpu/cpu/tpu
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_memory_fitting_level", "O3")
jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_compiler_enable_remat_pass", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
# # DEBUGGING BLOCK
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_no_tracing", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_check_tracer_leaks", True)

#import jax_array_info as jai
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from sparklines import sparklines
from tqdm.auto import tqdm
from utils import get_gpu_l2_cache_size, LOG_LEVELS

from boolean_whf import Objective
from sat_loader import PBSATFormula
from solvers import Optimiser, build_eval_verify, seq_eval_verify

logger = logging.getLogger(__name__)
# fh = logging.FileHandler(filename="jax.log")
# fh.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# jaxlog = logging.getLogger("jax")
# # Remove any existing handlers
# for handler in jaxlog.handlers[:]:
#     jaxlog.removeHandler(handler)
# jaxlog.setLevel(logging.DEBUG)
# jaxlog.addHandler(fh)
# jaxlog.propagate = False

# ruff: enable[E402]

if jax.__version_info__[1] < 7:
    jax.P = jax.sharding.PartitionSpec
print = functools.partial(print, flush=True)
ShardSpec: TypeAlias = tuple[NamedSharding, tuple[NamedSharding, ...]]


def x0_guesses(
    rng_key: Array, batch: int, n_vars: int, method: str = "bias", prefixes: Array | None = None
) -> tuple[Array, Array]:
    """
    Generates initial guesses for variable assignments in SAT problems using different randomization methods.

    Args:
        rng_key (Array): JAX PRNG key for random number generation.
        batch (int): Number of guess vectors to generate.
        n_vars (int): Number of variables in each guess vector.
        method (str, optional): Method for generating guesses. Options are:
            - "bias" (default): Generates values biased towards False from a uniform distribution
            - "coin": Generates using a biased (70% tending False) coin flip (Bernoulli).
            - "uniform": Generates values uniformly between True and False
        prefix_vectors (Array, optional): Shape (N, n_vars) where 0=no fix, ±1=fix to that value.
                                        Will be replicated to fill batch size B, so each vector appears B//N times.

    Returns:
        Array: An array of shape (batch, n_vars) containing the generated initial guesses.

    Raises:
        ValueError: If an unsupported method is specified.
    """

    if method == "bias":
        # Generate values biased towards 1 (False) while maintaining full [-1, 1] range
        u = jax.random.uniform(rng_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        bias_strength = 0.5  # Lower value = stronger bias towards False
        x0 = 2 * u**bias_strength - 1

    elif method == "coin":
        false_prob = 0.7  # Y% probability of generating values tending towards False
        coin_key, rng_key = jax.random.split(rng_key)
        biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(batch, n_vars))
        signs = 2 * biased_coins - 1
        magnitudes = jax.random.uniform(rng_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        x0 = signs * magnitudes

    elif method == "uniform":
        x0 = jax.random.uniform(rng_key, minval=-1, maxval=1, shape=(batch, n_vars))

    elif method == "trunc":
        x0 = jax.random.truncated_normal(rng_key, lower=-1, upper=1, shape=(batch, n_vars))

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Fix positions of supplied prefixes.
    fixed_mask = jnp.full((batch, 1), fill_value=False, dtype=bool)
    if prefixes is not None:
        N = prefixes.shape[0]
        # Batch is already correctly sized equal points for each prefix.
        replicated_prefixes = jnp.repeat(prefixes, batch // N, axis=0)

        # Non-zero points are fixed, so adjust batch and disable gradients there.
        fixed_mask = replicated_prefixes != 0
        x0 = jnp.where(fixed_mask, replicated_prefixes, x0)

    return x0, fixed_mask


@overload
def shard_tree(target: tuple[Objective, ...], sharding: NamedSharding) -> tuple[Objective, ...]: ...


@overload
def shard_tree(target: tuple[Array, ...], sharding: NamedSharding) -> tuple[Array, ...]: ...


def shard_tree(target, sharding) -> tuple[object, ...]:
    mesh = sharding.mesh
    replication = NamedSharding(mesh, jax.P())

    def shard_leaf(leaf):
        if isinstance(leaf, jax.Array):
            # Replicate if scalar or if the first dimension has size 1
            if leaf.ndim == 0 or (leaf.ndim > 0 and leaf.shape[0] == 1):
                return jax.device_put(leaf, replication)
            # Otherwise, shard along the first dimension
            return jax.device_put(leaf, sharding)
        return leaf

    return jax.tree.map(shard_leaf, target)


def get_mesh(devices: list) -> tuple[Mesh, NamedSharding, NamedSharding]:
    """
    By default we prefer more points, so always a (1, n_gpu) mesh. For truly enormous objectives we can always
    do (n_gpu, 1) and only treat a single batch of points, otherwise if the objectives are still quite big, for
    nice grid sizes other splits can be considered (heuristically?) e.g:
    4 GPU - 50/50 at (2,2), 6 GPU - (3,2) or (2,3), 8 GPU - (4,2) or (2,4), etc...

    NB SEPT-25: Actually, I think we should just always maximize points per GPU and never shard the objectives.
    We have the capability to here, of course, but unless the problem is truly enormous, I think it's probably likely
    that higher throughput will still be achieved only sharding in the batch.
    """
    # The reshape here would be adjusted if we ever wanted to shard over objectives as well.
    mesh = Mesh(np.array(devices).reshape((len(devices), 1)), ("batch", "objective"))
    jax.sharding.set_mesh(mesh)

    objective_spec = jax.P("objective")
    obj_sharding = NamedSharding(mesh, objective_spec)

    batch_spec = jax.P("batch")
    batch_sharding = NamedSharding(mesh, batch_spec)

    return mesh, obj_sharding, batch_sharding


def adjust_batch(devices: list, batch: int, target: int, est_mem_per_point: int, n_prefix: int = 1) -> int:
    n_device = len(devices)
    # max_gpu_mem = devices[0].memory_stats()["bytes_limit"]
    # max_batch = int((max_gpu_mem * 0.9) // est_mem_per_point)
    # opt_batch = int((max_gpu_mem * 0.01) // est_mem_per_point)
    max_batch = int((target) // est_mem_per_point)
    opt_batch = int((target * 0.9) // est_mem_per_point)

    if batch == -1 or batch > max_batch:
        logger.info("Adjusting per-device batch size (either none specified to batch too large):")
        logger.info(f"Set to {opt_batch} p.d. (tot. {opt_batch * n_device}) from {batch} (theoretical max {max_batch})")
        batch = opt_batch * n_device
    else:
        batch = batch * n_device

    # Adjust alignment.
    alignment = math.lcm(n_prefix, n_device)
    opt_batch = (batch // alignment) * alignment

    # Ensure we have at least one batch element per prefix and device combination
    if opt_batch < alignment:
        opt_batch = alignment
        logger.warning(f"Warning: Batch size set to min {opt_batch} to fit {n_prefix} prefixes and {n_device} devices")

    if n_prefix > 1:
        points_per_prefix = opt_batch // n_prefix
        logger.info(f"Batch distribution: {opt_batch} total = {n_prefix} prefixes x {points_per_prefix} points each")

    return opt_batch


def run_solver(
    timeout: int,
    n_vars: int,
    n_clause: int,
    batch: int,
    restart_thresh: int,
    fuzz_limit: int,
    objs: tuple[Objective, ...],
    n_devices: int = 1,
    sample_method: str = "bias",
    optimiser: str = "pgd",
    warmup: bool = False,
    benchmark: bool = False,
    counting: int = 0,
    rand_seed: bool = False,
    prefix_vectors: Array | None = None,
    maxiters: int = 100,
    weight_decay: float = 0.9, #alpha,
    unsat_h: int = 0,
    solver_tol: float = 1e-3,
    binary_v: bool = False
) -> float:
    devices = jax.devices("gpu")[:n_devices]
    n_prefix = len(prefix_vectors) if prefix_vectors is not None else 1

    mesh, obj_sharding, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    # Construct weights, and shard both weights and objectives.
    objs = shard_tree(objs, obj_sharding)
    weights = tuple(jnp.full((obj.clauses.lits.shape[0],), 1.0, dtype=float) for obj in objs)
    weights = shard_tree(weights, obj_sharding)

    # Construct pure JAX functions (closures) and build solver.
    obj_eval_fns, obj_verify_fns = build_eval_verify(objs, optimiser == "unbounded")
    seq_evaluator, seq_verifier = seq_eval_verify(obj_eval_fns, obj_verify_fns)
    solver = Optimiser(seq_evaluator, seq_verifier, algorithm=optimiser, maxiter=maxiters, tol=solver_tol)

    seed = int(time()) if rand_seed else 0
    logger.info(f"seed={seed}, rand_seed={rand_seed}")
    # logger.debug(f"seed={seed}, rand_seed={rand_seed}")
    key = jax.random.PRNGKey(np.array(seed))
    f_key = jax.random.PRNGKey(np.array(seed + 1))

    guess_batch = 0
    if batch == -1:
        logger.info("Guessing optimal batch size")
        # Optimal throughput is achieved when working set fits in GPU on-chip cache.
        # Fall back to 1% of VRAM if cache size is unavailable.
        l2_cache_size = get_gpu_l2_cache_size(devices[0])
        if l2_cache_size is not None:
            # Target ~95% of detected cache budget to leave room for other data
            gpu_mem_target = int(l2_cache_size * 0.95) * n_devices * 2
            logger.info(f"Targeting total cache: {l2_cache_size / (1024*1024):.1f} MB per GPU")

        else:
            ### Dead branch for now - the above call builds in a sensible default.
            # Fallback: 1% of VRAM heuristic
            gpu_mem_target = devices[0].memory_stats()["bytes_limit"] * 0.01
            logger.info("Cache size unknown, using 1% VRAM heuristic")
        dtype_sz = jnp.dtype(objs[0].ffts.dft.dtype).itemsize
        all_obj_sz = sum([np.prod([max(o.clauses.lits.shape), max(o.ffts.dft.shape) ** 2, dtype_sz]) for o in objs])
        guess_batch = int(np.floor(gpu_mem_target / (all_obj_sz))) * n_devices
        guess_batch -= guess_batch % n_devices
        guess_batch = max(guess_batch, n_devices)  # Ensure at least 1 per device
        logger.info(f"Initial batch size guess: {guess_batch}")

        x_guess = jax.device_put(
            jax.random.uniform(key, minval=0.99 - (5e-2), maxval=0.99, shape=(guess_batch, n_vars)), batch_sharding
        )
        empty_prefix = jax.device_put(jnp.full((guess_batch, n_prefix), fill_value=False, dtype=bool), batch_sharding)
        w_guess = tuple((w - 1e-4) for w in weights)
        peak_mem = solver.peak_memory_estimation(x_guess, empty_prefix, w_guess)
        mem_est_per_point = peak_mem // guess_batch
        logger.info(f"Initial batch size guess mem/point: {mem_est_per_point}")
    else:
        mem_est_per_point = 1
        gpu_mem_target = devices[0].memory_stats()["bytes_limit"]

    batch = adjust_batch(devices, batch, gpu_mem_target, mem_est_per_point, n_prefix)

    if warmup:
        if guess_batch != batch:
            # Size changed, so we need new arrays for warmup
            x_guess = jax.device_put(
                jax.random.uniform(f_key, minval=0.99 - (5e-2), maxval=0.99, shape=(batch, n_vars)), batch_sharding
            )
            empty_prefix = jax.device_put(jnp.full((batch, 1), fill_value=False, dtype=bool), batch_sharding)

        if not benchmark and logger.isEnabledFor(logging.INFO):
            if mesh.shape["batch"] > 1:
                logger.info("Batch sharding:")
                jax.debug.visualize_array_sharding(x_guess)
            if mesh.shape["objective"] > 1:
                logger.info("Objective sharding:")
                jax.debug.visualize_array_sharding(objs[0].clauses.lits)

        peak_mem = solver.peak_memory_estimation(x_guess, empty_prefix, weights)
        mem_est_per_point = peak_mem // batch
        logger.info(f"Warmup: shape - {x_guess.shape[0]}, peak memory - {peak_mem}, peak/point - {mem_est_per_point}")

        warm_start = time()
        solver.warmup((x_guess, empty_prefix, weights), bool(counting))
        warm_end = time()
        if not counting and solver.warmup_sol:
            # Found a solution during warmup which we have printed. Exit now.
            logger.info(f"W-TTFS {warm_end - warm_start}")
            logger.info(f"W-XT {warm_end - warm_start}")
            return warm_end - warm_start

    all_sols: dict[tuple, int] = defaultdict(int)
    first_sol: tuple | None = None
    best_x = np.zeros((n_vars))
    ttfs = 0
    best_unsat = jnp.inf
    best_unsat_clauses_idx = np.array([0])
    batches_done = 0
    restart_ct = 0
    restart_unsats = []
    all_unsats = []
    all_iters = []
    all_evals = []
    all_flips = []
    timeout_m, timeout_s = divmod(timeout, 60)

    if not benchmark:
        sparkline_height = 5
        hist_width = min(shutil.get_terminal_size().columns, solver.maxiter)
        iters_histo = sparklines({x: 0 for x in range(1, hist_width)}.values(), num_lines=sparkline_height)  # type: ignore
        histbars = [tqdm(desc=" ", position=x, bar_format="{desc}", leave=True) for x in range(len(iters_histo))]
        infobars = [tqdm(desc=" ", position=x + len(iters_histo), bar_format="{desc}", leave=True) for x in range(2)]
        if restart_thresh > 1:
            desc = f"\n{restart_ct} restarts (next: {batches_done % restart_thresh}/{restart_thresh} batches)"
        else:
            desc = f"\n{batches_done} batches"
        pbar = tqdm(
            total=timeout,
            leave=True,
            position=len(infobars) + len(histbars),
            desc=f"{desc} [MAX-SAT cost: {best_unsat}]",
            bar_format="{l_bar}{bar}|{elapsed}/" + f"{str(timeout_m).zfill(2)}:{str(timeout_s).zfill(2)}" + "{postfix}",
            postfix=f"{0:.2f}s/it",
        )
    accum_time_descent = 0

    t0 = time()
    while (time() - t0 < timeout) and (not solver.warmup_sol or counting):
        start_batch = time()
        tloop = time()

        # Randomisation & Init
        key, s_key = jax.random.split(key)
        f_key, s_f_key = jax.random.split(f_key)
        x0, fixed_vars = x0_guesses(s_key, batch, n_vars, sample_method, prefix_vectors)
        x0_dev = jax.device_put(x0.copy(), batch_sharding)
        fixed_vars = jax.device_put(fixed_vars, batch_sharding)

        if logger.isEnabledFor(logging.WARNING) and batches_done < 5:
            print(f"c Initial Assigment (batch {batches_done}, point 0)", np.asarray(x0[0,:].copy().tolist()))

        # Run solver.
        opt_x0, opt_unsat, opt_iters, opt_unsat_ct, aux_info = solver.run(x0_dev, fixed_vars, weights)
        accum_time_descent += time() - tloop

        if logger.isEnabledFor(logging.WARNING) and batches_done < 5:
            print(f"c Final Assigment (batch {batches_done}, point 0)", np.asarray(opt_x0[0,:].copy().tolist()))
            print("c DIFFERENT?", not all((x0[0,:] == opt_x0[0,:]).tolist()))
            # print(aux_info)

        # Flag and bail if we encounter anomalous behaviour
        if logger.isEnabledFor(logging.WARNING):
            eval_last = np.abs(np.asarray(aux_info[-1]))
            eval_oob = (eval_last > n_clause) & (~np.isclose(eval_last, float(n_clause)))
            if np.any(eval_oob):
                exceed = np.argwhere(eval_oob).flatten()
                logger.warning(
                    f"[{optimiser}] Detected numerical instability! \n Abs(eval) > {n_clause} outside tolerance!\n"
                    + f"At indices {exceed} in the most recent batch, we found:\n"
                    + f"Energy/Eval values of: {np.asarray(aux_info[-1])[exceed]}\n"
                    + f"due to an input of: \n{np.asarray(opt_x0)[exceed, :]}"
                )
                x_opt_abs = np.abs(np.asarray(opt_x0))
                x_opt_oob = (x_opt_abs > 1.0) & (~np.isclose(x_opt_abs, 1.0))
                if np.any(x_opt_oob):
                    escaped = np.argwhere(x_opt_oob)
                    logger.warning(f"[{optimiser}] Optimizer returned out-of-bounds points at: {escaped}\n"
                                   + f"Points: {opt_x0[escaped]}")

                aux_x_abs = np.abs(np.asarray(aux_info[0]))
                aux_x_oob = (aux_x_abs > 1.0) & (~np.isclose(aux_x_abs, 1.0))
                if np.any(aux_x_oob):
                    escaped_aux = np.argwhere(aux_x_oob)
                    logger.warning(f"[{optimiser}] Auxiliary evaluation points left bounds at: {escaped_aux}\n"
                                   + f"Points: {aux_info[0][escaped_aux]}")
                if QUIT_ON_ANOMALY:
                    return 0.0

        flips = (opt_x0 > 0).sum(axis=1) - (x0 > 0).sum(axis=1)

        # first argument is the second to last value x had in the descent
        _, eval_scores = aux_info
        eval_scores = jnp.array(eval_scores).squeeze().T

        batch_best_loc = jnp.argmin(opt_unsat_ct)
        batch_best_unsat = jnp.take(opt_unsat_ct, batch_best_loc)
        batch_best_x = opt_x0[batch_best_loc]
        batch_best_unsat_clauses_idx = jnp.nonzero(opt_unsat[batch_best_loc])
        if batch_best_unsat < best_unsat:
            best_x = np.asarray(batch_best_x).copy()
            best_unsat = np.asarray(batch_best_unsat).copy()
            best_unsat_clauses_idx = np.asarray(batch_best_unsat_clauses_idx).copy()

        tbatch = time()
        found_sol = False

        if unsat_h and batch_best_unsat <= unsat_h:
            ttfs = time() - t0
            found_sol = True
            opt_iters_local = np.array(opt_iters.flatten()).tolist()
            batches_done += 1
            end_batch = time()
            break

        if batch_best_unsat == 0:
            best_x = np.asarray(batch_best_x).copy()
            if ttfs == 0:
                ttfs = tbatch - t0
            found_sol = True

        if fuzz_limit and (counting or not found_sol):
            # Knock current batch in attempt to find more solutions.
            fuzz_attempt = 0
            F_x = opt_x0[:]
            f_key, s_f_key = jax.random.split(f_key)
            fuzz = jax.random.uniform(s_f_key, minval=1e-7, maxval=1e-2, shape=x0.shape)
            fuzz_mag = 1
            while fuzz_attempt < fuzz_limit:
                fuzz_mask = np.ones(F_x.shape, dtype=bool)
                if found_sol:
                    sol_locs = jnp.argwhere(jnp.where(opt_unsat_ct < 1, 1, 0)).flatten().tolist()
                    fuzz_mask[sol_locs, :] = False  # Do not add fuzz to these rows

                fuzz_attempt += 1
                if fuzz_mag != 1:
                    fuzz_adj = jnp.sign(fuzz) * jnp.abs(fuzz) ** (1 / fuzz_mag)
                else:
                    fuzz_adj = fuzz

                # Project back on to hypercube.
                F_x = jnp.clip(jnp.where(fuzz_mask, F_x + fuzz_adj, F_x), -1, 1)

                F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct, (_, F_eval_scores) = solver.run(
                    F_x, fixed_vars, weights
                )

                F_batch_best_loc = jnp.argmin(F_opt_unsat_ct)
                F_batch_best_unsat = jnp.take(F_opt_unsat_ct, F_batch_best_loc)
                F_batch_best_x = opt_x0[F_batch_best_loc]
                F_batch_best_unsat_clauses_idx = jnp.nonzero(opt_unsat[F_batch_best_loc])
                if F_batch_best_unsat < best_unsat:
                    # TODO: We make a (spurious?) assumption that bumping a solution will find that solution again
                    # So this check needs to be adjusted to also check the number the solutions and replace only if
                    # find more - this covers both the counting and not counting case.
                    # TODO: This would also subsume the next check somewhat - since found_sol would already be true.
                    # We beat the unfuzzed convergence, so keep this result instead.
                    best_x = np.asarray(F_batch_best_x).copy()
                    best_unsat = np.asarray(F_batch_best_unsat).copy()
                    best_unsat_clauses_idx = np.asarray(F_batch_best_unsat_clauses_idx).copy()
                    opt_x0, opt_unsat, opt_iters, opt_unsat_ct = F_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct

                tfuzz = time()
                if F_batch_best_unsat == 0:
                    if not found_sol:
                        if ttfs == 0:
                            ttfs = tfuzz - t0
                        found_sol = True
                        logger.info(f"Fuzz {fuzz_attempt} found a solution!")
                    elif len(jnp.argwhere(jnp.where(F_opt_unsat_ct < 1, 1, 0)).flatten().tolist()) > len(sol_locs):
                        logger.debug(f"Fuzz {fuzz_attempt} found more solutions")

                    if not counting:
                        break

                if (jnp.sign(F_x) == jnp.sign(F_opt_x)).all():
                    # If no points ended up changing signs after convergence, we didn't move at all. Increase magnitude
                    fuzz_mag += 1
                else:
                    fuzz_mag = 1

                F_x = F_opt_x

        if found_sol:
            first_sol = tuple(np.sign(best_x).astype(int).tolist())
            sol_locs = jnp.argwhere(jnp.where(opt_unsat_ct < 1, 1, 0)).flatten().tolist()
            batch_sols = [tuple(row.astype(int).tolist()) for row in np.sign(np.asarray(opt_x0[sol_locs, :]))]
            if counting:
                for sol in batch_sols:
                    all_sols[sol] += 1
            else:
                all_sols[tuple(np.sign(opt_x0[batch_best_loc, :]).astype(int).tolist())] += 1
                if not benchmark:
                    # Close tqdm. 
                    #TODO: We probably don't need this at all if we change tqdm usage (progress) to context manager.
                    #TODO: we should change benchmark to track "progress instead" as the following is much clearer:
                    # # if progress and not counting:
                    for x in range(len(histbars)):
                        histbars[x].close()
                    for x in range(len(infobars)):
                        infobars[x].close()
                    pbar.close()
                    logger.info("SAT! at sample {}".format(max(batches_done, 0) * batch + batch_best_loc))
                opt_iters_local = np.array(opt_iters.flatten()).tolist()
                batches_done += 1
                end_batch = time()
                break

        opt_iters_local = np.array(opt_iters.flatten()).tolist()
        batches_done += 1
        end_batch = time()

        if restart_thresh:
            restart_unsats.append(np.asarray(opt_unsat))
            all_unsats.extend(np.array(opt_unsat_ct.flatten()).tolist())
            all_evals.extend(np.array(eval_scores.flatten()).tolist())
            all_iters.extend(opt_iters_local)
            all_flips.extend(np.array(flips.flatten()).tolist())
            # TODO: There is probably some way to calculate how many restarts are needed given the batch size to get
            # enough of a sample to ensure the reweighting is meaningful. If there's 500 clauses and we are only running
            # batches of 100, we probably need restart_thresh to be more than 1, at the very least. I think? I don't
            # know if this reasoning has mathematical basis or is just my intuition.
            ## LLM response to the above todo:
            # Statistical confidence for reweighting: We want enough samples per clause to make penalties meaningful.
            # With n_clause clauses and batch size B, after T batches we have B*T samples. For uniform sampling,
            # we'd expect ~(B*T)/n_clause samples per clause on average. A reasonable heuristic is to ensure at least
            # 10-20 samples per clause for stable weight updates, suggesting restart_thresh ≥ max(1, ceil(10*n_clause/batch)).
            # However, this assumes uniform distribution of unsat clauses, which isn't true - hard clauses will be
            # oversampled. So in practice, fewer batches may suffice. Consider: restart_thresh = max(1, n_clause // batch)
            # as a starting point, ensuring we see roughly batch-size coverage of the clause space.

            if not (batches_done % restart_thresh):
                # Gather unsat counts by clause
                penalty = jnp.atleast_1d(jnp.concatenate(restart_unsats, axis=1).sum(axis=0))
                if jnp.any(penalty):
                    worst = penalty.max() 

                    logger.debug(f"# Restart: {restart_ct} | Best MAX-SAT cost (#unsat): {best_unsat}")
                    logger.debug(f"Unsat counts: {penalty}, \nBest: {best_unsat_clauses_idx}")

                    pen_start = 0
                    new_weights: list[Array] = []
                    for weight in weights:
                        obj_clauses = len(weight)
                        pen_end = pen_start + obj_clauses
                        w_pens = penalty[pen_start:pen_end]
                        new_weight = weight_decay * weight + (1 - weight_decay) * w_pens / worst
                        new_weights.append(new_weight)
                        pen_start += obj_clauses

                    weights = shard_tree(tuple(new_weights), obj_sharding)
                restart_unsats = []
                restart_ct += 1

        if not benchmark:
            # Update tqdm info/histogram bars
            # TODO: This should be our own tqdm wrapper, with the histogram display as a param, and we can
            # remove this update code and the init code since it pollutes our flow here.
            # would allow for easier paraming of the "height" of the histo display, maybe even split display
            # e.g. left histo is current batch, right histo is aggregate.
            # TODO: since we are sharding as well, this might be a wrapper over jax-tqdm instead!
            opt_iters_counts = Counter(opt_iters_local)
            bin_width = solver.maxiter / hist_width if solver.maxiter > hist_width else 1
            iters_histo = [0] * hist_width
            for k, v in opt_iters_counts.items():
                bin_idx = int(k / bin_width)
                if bin_idx == hist_width:
                    bin_idx -= 1
                iters_histo[bin_idx] += v
            iters_histo_tq = sparklines(iters_histo, num_lines=5)  # type: ignore

            max_iter_str = str(solver.maxiter)
            max_iter_len = len(max_iter_str)
            if solver.maxiter >= hist_width:
                pad_bar = " " * (hist_width - 1 - max_iter_len)
                infobars[0].set_description_str("0" + pad_bar + max_iter_str)
            else:
                end_label = str(hist_width - 1)
                end_label_len = len(end_label)
                pad_left = " " * (solver.maxiter - max_iter_len)
                pad_right = " " * (hist_width - 1 - solver.maxiter - end_label_len)
                infobars[0].set_description_str("0" + pad_left + max_iter_str + pad_right + end_label)

            for x in range(len(histbars)):
                histbars[x].set_description_str(iters_histo_tq[x])

            infobars[-1].set_description_str(
                f"Optim Iters: min: {jnp.min(opt_iters)}, "
                + f"max: {jnp.max(opt_iters)} ({opt_iters_counts[solver.maxiter]}), "
                + f"median: {int(jnp.median(opt_iters))}"
            )

            if restart_thresh > 1:
                desc = f"{restart_ct} restarts (next: {batches_done % restart_thresh}/{restart_thresh} batches)"
            else:
                desc = f"{batches_done} batches"
            pbelapse = pbar.format_dict["elapsed"]
            pbn = pbar.format_dict["n"]
            batch_elapsed = end_batch - start_batch
            pbar.set_description(f"{desc} [MAX-SAT cost: {best_unsat}]")
            if pbn + batch_elapsed > timeout:
                pbar.update(timeout - pbelapse)
            else:
                pbar.update(end_batch - start_batch)  # update *adds* the input to the counter.
            pbar.set_postfix_str(f"({(pbelapse / batches_done):.2f}s/it)")

    tsolve = time() - t0
    if not benchmark:
        # TODO: The above todo re tqdm context would remove the need to close these.
        if len(histbars):
            for x in range(len(histbars)):
                histbars[x].close()
        if len(infobars):
            for x in range(len(infobars)):
                infobars[x].close()
        pbar.close()

    sol_info = f"Accelerated Fourier SAT ({optimiser})"
    print(f"c {"-"*len(sol_info)}")
    print(f"c {sol_info}")
    print(f"c {"-"*len(sol_info)}")

    if len(all_sols):
        print("s SATISFIABLE")
        for sol_i, sol in enumerate(all_sols.keys()):
            if not counting and first_sol != sol:
                continue
            sol_str_d = []
            sol_str_b = []
            for var, assigned in enumerate(np.sign(sol).astype(int)):
                lit = (var + 1) * assigned # assigned=-1 means "True"
                sol_str_d.append(f"{-lit}")
                sol_str_b.append("0" if assigned > 0 else "1")
            sol_str = "".join(sol_str_b) if binary_v else " ".join(sol_str_d)
            if counting:
                logger.info(f"{sol_i+1}: v {sol_str} 0")
                if not sol_i:
                    print("c ENUMERATING - example solution (of possibly many):")
                    print(f"v {sol_str} 0")
            elif first_sol == sol:
                print(f"v {sol_str} 0")
                break
    else:
        assign_str_d = []
        assign_str_b = []
        assignment = []
        pr_signs = ["?", "-", ""]
        for var, assigned in enumerate(np.sign(best_x).astype(int)):
            lit = var + 1
            assign_str_d.append(f"{pr_signs[int(assigned)]}{lit}")
            assign_str_b.append("0" if assigned > 0 else ("1" if assigned < 0 else "-"))
            if assigned:
                assignment.append(lit * -assigned)
        assign_str = "".join(assign_str_b) if binary_v else " ".join(assign_str_d)

        print("s UNKNOWN")
        print("c Best found MAX-SAT assignment (zero energy variables omitted or -):")
        print(f"o {best_unsat}")
        print(f"v {assign_str} 0")

        assignment = set(assignment)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("UNSATS (set literals-clause literals):\n")
            for i, cl_idx in enumerate(best_unsat_clauses_idx.flatten()):
                find_idx = cl_idx
                for obj in objs:
                    obj_len = obj.clauses.lits.shape[0]
                    if find_idx < obj_len:
                        if len(obj.clauses.sign.shape) > 1:
                            clause = set(((obj.clauses.lits[find_idx] + 1) * obj.clauses.sign[find_idx]).tolist())
                        else:
                            clause = set(((obj.clauses.lits[find_idx] + 1) * obj.clauses.sign).tolist())
                        logger.debug(f"{(sorted(clause.intersection(assignment)), clause)}")
                        break
                    else:
                        find_idx -= obj_len
            for i, w in enumerate(weights):
                logger.debug(f"\t{set((np.array(objs[i].clauses.types)).flatten().tolist())},{objs[i].clauses.lits.shape} \n{w}")
    print(f"c {"-"*len(sol_info)}")

    if logger.isEnabledFor(logging.INFO):
        if ttfs:
            logger.info(f"X-TTFS {ttfs}")
        logger.info("START EXP")
        logger.info(f"X-GPU {n_devices}")
        logger.info(f"X-PPBATCH {batch}")
        logger.info(f"X-PPGPUBATCH {batch // n_devices} PPBATCH/GPU")
        if warmup:
            logger.info(f"X-PEAK {peak_mem}")
            logger.info(f"X-PEAKPP {mem_est_per_point} PEAK/PPBATCH")
        logger.info(f"X-BATCHES {batches_done}")
        logger.info(f"X-PTOTAL {batches_done * batch} BATCHES*PPBATCH")
        logger.info(f"X-LOOP {tsolve}")
        logger.info(f"X-DESC {accum_time_descent}")
        if len(all_sols):
            logger.info(f"X-SOLS {sum(all_sols.values())}")
            logger.info(f"X-UQSOLS {len(all_sols.keys())}")
        else:
            logger.info("X-SOLS 0")
            logger.info("X-UQSOLS 0")
        logger.info(f"X-RATIODESC {(batches_done * batch) / accum_time_descent} POINTS/DESCTIME")
        logger.info(f"X-RATIOLOOP {(batches_done * batch) / tsolve} POINTS/LOOPTIME")
        logger.info(f"X-ITERHISTO {dict(Counter(all_iters))}")
        logger.info(f"X-RAWEVAL {all_evals}")
        logger.info(f"X-RAWUNSAT {all_unsats}")
        logger.info(f"X-RAWITER {all_iters}")
        logger.info(f"X-RAWFLIPS {all_flips}")
        logger.info("END EXP")
    return tsolve


def main(
    file: str,
    timeout: int,
    batch: int = -1,
    restart_thresh: int = 0,
    fuzz: int = 0,
    n_devices: int = 1,
    disk_cache: str = "",
    benchmark: bool = False,
    counting: int = 0,
    warmup: bool = False,
    rand_seed: bool = False,
    prefix_file: str = "",
    maxiters: int = 100,
    unsat_h: float = 0,
    sample_method: str = "bias",
    solver_tol: float = 1e-3,
    optimiser: str = "pgd",
    binary_v: bool = False
) -> None:
    if not file:
        raise ValueError("No problem file specified")
    stamp1 = time()
    sat_parser = PBSATFormula(workers=4, n_devices=n_devices, disk_cache=disk_cache, file=file, compactify=False)
    stamp2 = time()
    read_time = stamp2 - stamp1

    objectives = sat_parser.process_clauses_to_array()
    n_var = sat_parser.n_var
    n_clause = sat_parser.n_clause
    prefixes = sat_parser.process_prefix(prefix_file)
    prefixes = jnp.array(prefixes) if prefixes is not None else None
    stamp1 = time()
    process_time = stamp1 - stamp2

    t_solve = run_solver(
        timeout,
        n_var,
        n_clause,
        batch,
        restart_thresh,
        fuzz,
        objectives,
        n_devices=n_devices,
        counting=counting,
        rand_seed=rand_seed,
        benchmark=benchmark,
        warmup=warmup,
        prefix_vectors=prefixes,
        maxiters=maxiters,
        unsat_h=int(unsat_h * 2 * n_var) if unsat_h else 0,
        sample_method=sample_method,
        solver_tol=solver_tol,
        optimiser=optimiser,
        binary_v=binary_v,
    )

    logger.info(f"Time reading input: {read_time}")
    logger.info(f"Time processing to Arrays: {process_time}")
    logger.info(f"Time spent solving: {t_solve}")


if __name__ == "__main__":
    n_devices = len(jax.devices("gpu"))

    ap = ArgParse(
        description="Process a file with optional parameters",
        epilog="Some debug options:"
        + "JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=[X]"
        + "JAX_LOGGING_LEVEL=DEBUG TF_CPP_MIN_LOG_LEVEL=[X] TF_CPP_MAX_VLOG_LEVEL=[X]"
        + "JAX_TRACEBACK_FILTERING=off",
    )
    ap.add_argument("file", help="The file to process")
    ap.add_argument("-y", "--profile", action="store_true", help="Enable profiling")
    ap.add_argument("-t", "--timeout", type=int, default=300, help="Maximum runtime (timeout seconds)")
    ap.add_argument("-b", "--batch", type=int, default=-1, help="Batch size. -1 computes heuristic maximum")
    ap.add_argument("-f", "--fuzz", type=int, default=0, help="Number of times to attempt fuzzing per batch")
    ap.add_argument("-r", "--restart_thresh", type=int, default=1, help="Batches before reweighting (never if 0)")
    ap.add_argument("-n", "--n_devices", type=int, default=n_devices, help="Devices (eg. GPUs) to use. 0 uses all")
    ap.add_argument("-e", "--benchmark", action="store_true", default=True, help="Benchmark mode (reduce output)")
    ap.add_argument("--progress", action="store_false", dest="benchmark", help="Display progress stats (equiv to -e False)")
    ap.add_argument("-c", "--counting", action="store_true", help="Counting mode. Count solns until timeout")
    ap.add_argument("-w", "--warmup", action="store_true", help="Perform a warmup run before starting timer")
    ap.add_argument("-d", "--debug", choices=LOG_LEVELS, default="ERROR", help=f"Set logging level ({LOG_LEVELS})")
    ap.add_argument("-s", "--rand_seed", action="store_true", help="Randomise seed")
    ap.add_argument("-p", "--prefix", type=str, default="", help="Fixed assignments file, each a line of assigned vars")
    ap.add_argument("-i", "--iters_desc", type=int, default=100, help="Solver maximum iterations")
    ap.add_argument("-u", "--unsat_thresh", type=float, default=0, help="Stop when #UNSAT drops below threshold (PLE)")
    ap.add_argument("-m", "--sample_meth", type=str, default="bias", help="Starting point sampler (bias,coin,uniform)")
    ap.add_argument("-q", "--solver_tol", type=float, default=1e-3, help="Optimiser convergence tolerance (if supported)")
    ap.add_argument("-o", "--optimiser", "--optimizer", dest="optimiser", type=str, default="pgd", help="Optimiser")
    ap.add_argument("--stdout_log", action="store_true", help="Send log output to stdout instead of stderr")
    ap.add_argument("--anomaly_quit", action="store_true", default=False)
    ap.add_argument("--log_propagate", action="store_true", default=False)
    ap.add_argument("--binary_v", action="store_true", default=False, help="Short form solution string")

    arg = ap.parse_args()
    logger.info(f"{arg._get_args()}")

    # TODO: This should not be a psuedo-global like this???
    QUIT_ON_ANOMALY: bool = arg.anomaly_quit


    if not arg.n_devices:
        arg.n_devices = n_devices

    logger.propagate = True #arg.log_propagate
    logging.basicConfig(
        level=getattr(logging, arg.debug.upper()),
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        format="c %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout if arg.stdout_log else sys.stderr),
            # Optional: logging.FileHandler('sat_loader.log')  # Also log to file
        ],
    )

    # Run with or without profiler based on the flag
    profiler = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False) if arg.profile else nullcontext()
    with profiler:
        main(
            arg.file,
            arg.timeout,
            arg.batch,
            arg.restart_thresh,
            arg.fuzz,
            arg.n_devices,
            benchmark=arg.benchmark,
            counting=arg.counting,
            warmup=arg.warmup,
            rand_seed=arg.rand_seed,
            prefix_file=arg.prefix,
            maxiters=arg.iters_desc,
            unsat_h=arg.unsat_thresh,
            sample_method=arg.sample_meth,
            solver_tol=arg.solver_tol,
            optimiser=arg.optimiser,
            binary_v=arg.binary_v
        )
        if arg.profile:
            jax.profiler.save_device_memory_profile("memory.prof")
    logger.debug(f"Running with configuration: {arg}")
