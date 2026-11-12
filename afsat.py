# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import functools
import logging
import math
import os
import shutil
import sys
from argparse import SUPPRESS
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter as time
from typing import Any, TypeAlias, overload

from jsonargparse import ArgumentParser as ArgParse

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"  # Use full memory allocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Only the scheduler and stream flags are kept. An A/B on card, sudoku, PLE and SHA-1 r20 with RREF (JAX 0.11.2)
# showed no change in compile time, time per batch or results from the others: the GEMM flags (triton_gemm_any,
# cublaslt, autotune_gemm_rtol, exhaustive_tiling_search) have nothing to act on, since the compiled kernels contain
# no matrix multiplications, and fast_min_max (min/max that do not propagate NaN) and fast_math gave nothing
# measurable, while NaN-propagating min/max keep the anomaly checks honest. The two kept flags matter only with
# collectives (several GPUs). To A/B a flag, append to XLA_FLAGS after importing this module.
os.environ["XLA_FLAGS"] = " ".join(  # noqa: FLY002
    [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
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
# The evaluators close over their clause data on purpose, and JAX is moving to new handling of closed-over constants
# that hoists every one above this size into executable arguments (32 bytes in 0.11.2;
# docs.jax.dev/en/latest/internals/constants.html). Keep everything embedded. Measured on JAX 0.11.2: the new handling
# makes no difference on card, sudoku, Ramsey or SHA, but slows the all-XOR PLE kernel by 14-32% whatever this
# threshold is. The option exists from about JAX 0.11.
if hasattr(jax.config, "jax_embedded_constants_max_bytes"):
    jax.config.update("jax_embedded_constants_max_bytes", 2**31 - 1)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
# # DEBUGGING BLOCK
# jax_debug_nans is set from --debug_nans (off by default: it checks every computation's outputs on the host).
# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_no_tracing", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_check_tracer_leaks", True)

# import jax_array_info as jai
import numpy as np
from jax import Array
from numpy.typing import NDArray
from jax.sharding import Mesh, NamedSharding
from sparklines import sparklines
from tqdm.auto import tqdm

from boolean_whf import Objective
from samplers import SAMPLERS, sample_assignments
from sat_loader import PBSATFormula, UnsatError
from solvers import Optimiser, build_eval_verify, drop_projected_xor_evaluators, seq_eval_verify
from utils import (
    LOG_LEVELS,
    AFSATConfig,
)
from propagation import propagate_units
from var_mapper import VarMapper
from xor_rref import (
    FoldLayout,
    XorRow,
    XorRREFMetadata,
    fold_prefix,
    literals_from_unit_bits,
    make_fold_layout,
    pack_rows,
    preprocess_xor_system,
    rref_rows,
    unit_bits_from_literals,
)

logger = logging.getLogger(__name__)
QUIT_ON_ANOMALY = False
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


if jax.__version_info__[1] < 7:
    jax.P = jax.sharding.PartitionSpec
print = functools.partial(print, flush=True)
ShardSpec: TypeAlias = tuple[NamedSharding, tuple[NamedSharding, ...]]


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


def random_seed() -> int:
    """The seed when none is given (-s -1, the default): 31 bits from the OS, so independent runs, on one node or
    many, start from independent points. (The clock here is perf_counter, seconds since boot: too coarse a seed.)"""
    return int.from_bytes(os.urandom(4), "little") >> 1


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


# Batch sizing. Throughput is set by giving the GPU enough work, not by fitting a cache: on a V100
# and an RTX 2000 Ada (SHA-1 r20/r80, scripts/batch_sweep.py) per-point cost kept falling well
# past L2. Small problems need many points to fill the GPU, so they are sized by a share of device
# memory -- the 0.1-1% band where the AFSAT paper measured peak throughput. Large problems fill it
# with few points, and bigger batches then mostly pay for the vmapped line search waiting on the
# batch's worst point, so they get a fixed floor. An opt-in measured calibration is a possible
# future upgrade (claude_scratch/14_phase1_prefix_rref_plan.md).
BATCH_FILL_FRAC = 0.0025  # small problems: enough points to fill 0.25% of device memory
BATCH_FLOOR = 16  # large problems: points per device
BATCH_MEM_CAP_FRAC = 0.5  # never plan on more than half of device memory


def heuristic_batch_per_device(device, mem_per_point: int) -> tuple[int, int | None]:
    """(chosen, memory cap) points per device, given the estimated peak memory of one point."""
    mem_limit = (device.memory_stats() or {}).get("bytes_limit")
    if not mem_limit:
        return BATCH_FLOOR, None  # no memory stats (e.g. CPU): nothing to scale by
    mem_per_point = max(int(mem_per_point), 1)
    cap = max(int(BATCH_MEM_CAP_FRAC * mem_limit) // mem_per_point, 1)
    fill = math.ceil(BATCH_FILL_FRAC * mem_limit / mem_per_point)
    return min(max(BATCH_FLOOR, fill), cap), cap


def adjust_batch(devices: list, batch: int, max_batch: int | None = None, n_prefix: int = 1) -> int:
    """Total batch from a per-device size: capped by memory if known, aligned to devices and prefixes."""
    n_device = len(devices)
    if max_batch is not None and batch > max_batch:
        logger.warning(f"Batch of {batch} p.d. exceeds the memory cap of {max_batch} p.d.; using the cap")
        batch = max_batch
    logger.info(f"Set to {batch} p.d. (tot. {batch * n_device})")
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


def make_batch_programs(
    batch: int,
    n_vars: int,
    sample_method: str,
    ffsat_seed: bool,
    reweight: bool,
    weight_decay: float,
    weights: tuple[Array, ...],
    batch_sharding: NamedSharding,
) -> tuple[Callable, Callable, Callable]:
    """The main loop's per-batch work around the descent, as compiled programs: (draw, summarise, reweight).

    draw(state, prefixes) -> (state, x0, fixed, positives): the next batch's starting points, sharded over the batch
        like the kernel's inputs, and the sampler state for the batch after. --ffseed seeds batch k with PRNGKey(k), as
        FastFourierSAT does; otherwise a split chain. jax's partitionable threefry gives the same points whatever
        the sharding.
    summarise(x_opt, unsat, unsat_ct, penalty) -> (count, index, x, unsatisfied clauses, penalty): the batch's best
        point, and the reweighting penalty with this batch's unsatisfied counts per clause added. Reductions over the
        sharded batch axis become cross-device reductions; the outputs are replicated, so every device (and, with
        multiple hosts, every process) holds the whole result and can read it.
    reweight(weights, penalty) -> (weights, penalty): each clause weight moved towards its share of the penalty
        (worst clause = 1), with the weights' shardings unchanged, so the kernel never sees a new argument type; the
        penalty reset.
    """
    replicated = NamedSharding(batch_sharding.mesh, jax.P())
    w_sizes = tuple(int(w.shape[0]) for w in weights)
    w_shardings = tuple(w.sharding for w in weights)
    bounds = np.cumsum((0,) + w_sizes)

    def _draw(state, prefixes):
        if ffsat_seed:
            s_key, state = jax.random.PRNGKey(state), state + 1
        else:
            state, s_key = jax.random.split(state)
        x0, fixed = sample_assignments(s_key, batch, n_vars, sample_method, prefixes)
        return state, x0, fixed, jnp.sum(x0 > 0, axis=1)

    def _summarise(x_opt, unsat, unsat_ct, penalty):
        loc = jnp.argmin(unsat_ct)
        if reweight:
            penalty = penalty + jnp.sum(unsat, axis=0)
        return unsat_ct[loc], loc, x_opt[loc], unsat[loc], penalty

    def _reweight(weights, penalty):
        worst = jnp.max(penalty)
        safe = jnp.where(worst > 0, worst, 1)
        new = tuple(
            jnp.where(worst > 0, weight_decay * w + (1 - weight_decay) * penalty[a:b] / safe, w)
            for w, a, b in zip(weights, bounds[:-1], bounds[1:])
        )
        return new, jnp.zeros_like(penalty)

    draw = jax.jit(_draw, out_shardings=(replicated, batch_sharding, batch_sharding, batch_sharding))
    summarise = jax.jit(_summarise, out_shardings=replicated)
    reweight_weights = jax.jit(_reweight, out_shardings=(w_shardings, replicated))
    return draw, summarise, reweight_weights


def build_solver(
    config: AFSATConfig,
    objs: tuple[Objective, ...],
    xor_rref_meta: XorRREFMetadata | None,
    prefix_vectors: Array | None,
    *,
    keep_xor_eval: bool = False,
    folded_xor: bool = False,
) -> Optimiser:
    """The objective and the batch solver, shared by the CLI loop and the Dagster worker.

    objs are the objectives, placed on the mesh. Applies the XOR-evaluator drop under RREF projection (unless
    keep_xor_eval).

    folded_xor (Dagster worker): the projection reads per-prefix folded RREF rows from the kernel's extra argument
    (xor_rref.fold_prefix / pack_rows). The fold keeps every XOR clause satisfied by construction whatever the prefix
    fixes, so the XOR evaluator is always dropped."""
    runtime_afsat, optimiser_cfg = config.runtime_afsat, config.optimiser
    optimiser = optimiser_cfg.name
    obj_eval_fns, obj_verify_fns = build_eval_verify(objs, optimiser == "unbounded")
    if runtime_afsat.drop_xor_eval and xor_rref_meta is not None and optimiser != "unbounded" and not keep_xor_eval:
        # The projector satisfies every XOR clause by construction -- unless a prefix fixes an
        # RREF dependent, in which case that row is only enforced by the objective, so keep it.
        # (A folded RREF has no fixed dependents: the fold re-pivots their rows.)
        dep_idx = np.asarray(xor_rref_meta.dependent_indices)
        fixed_dep = 0 if prefix_vectors is None or folded_xor else int(
            np.any(np.asarray(prefix_vectors)[:, dep_idx] != 0, axis=0).sum()
        )
        if fixed_dep:
            logger.info(f"Keeping XOR objective: prefixes fix {fixed_dep} RREF dependent variables")
        else:
            obj_eval_fns, n_dropped = drop_projected_xor_evaluators(objs, obj_eval_fns)
            logger.info(f"XOR objective handled by RREF projection; dropped {n_dropped} evaluator(s)")
    seq_evaluator, seq_verifier = seq_eval_verify(
        obj_eval_fns, obj_verify_fns, xor_rref_meta=xor_rref_meta, folded_xor=folded_xor
    )
    return Optimiser(
        seq_evaluator, seq_verifier, algorithm=optimiser, maxiter=optimiser_cfg.max_iters,
        tol=optimiser_cfg.tolerance,
    )


def pin_warmup_inputs(x: Array, prefix_vectors: Array | None) -> tuple[Array, Array]:
    """Apply the run's prefixes to warmup data, exactly as sample_assignments does for real batches.

    Warmup must see the same fixed variables as the real run, for two reasons. Correctness: unit
    clauses exist only as the prefix (the loader strips them from the clause arrays), so an
    unpinned warmup can "solve" the instance while violating them. And cost: sample_assignments
    returns a (batch, n_vars) mask whenever prefixes exist, so a (batch, 1) warmup mask compiles a
    different program and the first real batch recompiles.
    """
    if prefix_vectors is None:
        return x, jnp.full((x.shape[0], 1), fill_value=False, dtype=bool)
    # Warmup batch sizes need not divide by the prefix count; cycle the rows instead.
    replicated = jnp.resize(jnp.asarray(prefix_vectors), (x.shape[0], x.shape[1])).astype(x.dtype)
    fixed = replicated != 0
    return jnp.where(fixed, replicated, x), fixed


@dataclass
class AFSATProblem:
    problem_file: str
    sat_parser: PBSATFormula
    objectives: tuple[Objective, ...]
    xor_rref_meta: XorRREFMetadata | None
    n_var: int
    n_clause: int
    var_mapper: VarMapper
    n_never_sat: int = 0  # MaxSAT mode: soft clauses no assignment satisfies (not in the objectives)


@dataclass
class AFSATWorkerSession:
    """A persistent worker's search (afsat_worker.py), built and compiled once by create_worker_session; each request
    then runs one batch for its prefix (run_worker_single_batch).

    Under the RREF projection (pgd / lbfgsb), each request's prefix is folded into the RREF on the host
    (xor_rref.fold_prefix): fixed dependents hand their rows to free variables, dependents the prefix determines join
    the prefix, and a contradicting prefix is refuted without a descent. The folded rows go to the kernel as an
    argument in a fixed slot layout (fold_layout), so a new prefix costs a device_put, not a recompile, and the XOR
    evaluator stays dropped. If the folded rows overflow the layout (fill-in), the request falls back to a kernel
    with the unfolded RREF built in that keeps the XOR evaluator (solvers[True], compiled on first need), as it does
    for any RREF instance with another optimiser (dep_idx: when a prefix fixes a dependent)."""

    prepared: AFSATProblem
    config: AFSATConfig
    objs: tuple[Objective, ...]
    weights: tuple[Array, ...]
    batch: int
    solvers: dict[bool, Optimiser]
    dep_idx: NDArray | None
    fold_rows: list[XorRow] | None
    fold_layout: FoldLayout | None
    replicated: NamedSharding
    draw: Callable
    summarise: Callable
    sampler_state: Array
    penalty: Array
    f_key: Array
    unit_vector: Array
    x_spec: Any
    fixed_spec: Any
    unsat_h: int
    n_never_sat: int
    fuzz_limit: int
    counting: bool
    stats: bool
    binary_v: bool
    warmup_done: bool


@dataclass
class AFSATBatchResult:
    best_unsat: int
    sat: bool
    threshold_hit: bool
    best_assignment_signed: tuple[int, ...]
    best_assignment_str: str
    best_unsat_clause_indices: list[int]
    iters: list[int]
    unsat_per_point: list[int]
    eval_per_point: list[float]
    flips_per_point: list[int]
    elapsed_sec: float


def prepare_problem(
    problem_file: str,
    config: AFSATConfig,
    *,
    prefix_file: str | None = None,
    trace_hook: Callable[[str], None] | None = None,
) -> tuple[AFSATProblem, Array | None, float, float]:
    def trace(message: str) -> None:
        if trace_hook is None:
            return
        trace_hook(message)

    if not problem_file:
        raise ValueError("No problem file specified")

    # MaxSAT mode (-u): every input clause is soft (the input formats cannot mark a clause hard), so the only literals
    # that may be pinned are prefix literals, which are ground truth. Nothing is derived from the formula: no unit
    # pins from parsing, no unit propagation, no XOR preprocessing (its derived literals and RREF projection both
    # treat the XOR clauses as hard). Counting mode keeps hard semantics.
    soft = bool(config.runtime_common.unsat_thresh)
    stamp1 = time()
    sat_parser = PBSATFormula(
        workers=4,
        n_devices=config.runtime_common.n_devices,
        disk_cache=config.invocation.disk_cache,
        file=problem_file,
        compactify=False,
        soft=soft,
    )
    stamp2 = time()
    read_time = stamp2 - stamp1

    maxsatish_mode = bool(config.runtime_common.counting or config.runtime_common.unsat_thresh)
    objectives = sat_parser.process_clauses_to_array()
    xor_rref_meta: XorRREFMetadata | None = None
    if soft:
        logger.info(
            "MaxSAT mode: all clauses soft; no unit propagation, XOR preprocessing or XOR RREF projection "
            "(they would enforce soft clauses as hard)"
        )
    if config.runtime_afsat.propagate and not soft:
        prop1 = time()
        propagated = propagate_units(sat_parser.clause_sets, sat_parser.unit_prefix)
        logger.info(f"X-PROPTIME {time() - prop1}")
        if propagated.unsat:
            if not maxsatish_mode:
                print("s UNSATISFIABLE")
                raise UnsatError("Formula is inconsistent under unit propagation")
            logger.warning("Unit propagation found a conflict; continuing with the unpropagated prefix")
        elif propagated.derived:
            # Implied literals only need to reach fixed_vars/x0: no clause array is rewritten, a
            # clause the propagation satisfied simply evaluates as satisfied under the pin.
            sat_parser.unit_prefix |= propagated.derived
            logger.info(
                f"Unit propagation implied {len(propagated.derived)} further literals; "
                f"unit prefix is now {len(sat_parser.unit_prefix)} of {sat_parser.n_var} variables"
            )

    # Unit propagation over the XOR system is problem simplification, not projection, so it
    # runs whenever there are XOR clauses. Only the second tier -- dependents the reduction
    # forces outright, which need a linear combination of clauses to expose -- is gated on
    # the RREF actually being built.
    if sat_parser.xor_clause_sets and not soft:
        xor1 = time()
        xor_result = preprocess_xor_system(
            sat_parser.xor_clause_sets,
            unit_bits_from_literals(sat_parser.unit_prefix),
            build_rref=config.runtime_afsat.xor_rref,
        )
        xor_rref_meta = xor_result.meta
        xor2 = time()
        logger.info(f"X-XORPROCTIME {xor2 - xor1}")
        # print(f"X-XORPROCTIME {xor2 - xor1}", flush=True)
        if xor_result.unsat:
            if not maxsatish_mode:
                print("s UNSATISFIABLE")
                raise UnsatError("XOR subsystem is inconsistent under preprocessing")
            logger.warning("MaxSAT mode: continuing without XOR projection despite the inconsistent XOR system")
        else:
            if xor_result.derived:
                # Propagation implied further literals. They must reach fixed_vars/x0, and
                # both process_prefix (just below) and process_prefix_line merge unit_prefix
                # into every prefix, so extending it here is all that is needed. This must
                # stay ahead of the process_prefix call.
                sat_parser.unit_prefix |= set(literals_from_unit_bits(xor_result.derived))
                logger.info(
                    f"XOR unit propagation implied {len(xor_result.derived)} further literals; "
                    f"unit prefix is now {len(sat_parser.unit_prefix)}"
                )
            if xor_rref_meta is None:
                if config.runtime_afsat.xor_rref:
                    logger.info("XOR system fully determined by unit propagation; no projection needed")
            else:
                logger.info(f"XOR RREF sucessful - received {xor_rref_meta}")


    pf = config.invocation.prefix_file if prefix_file is None else prefix_file
    prefixes = sat_parser.process_prefix(pf)
    prefixes = jnp.array(prefixes) if prefixes is not None else None
    stamp1 = time()
    process_time = stamp1 - stamp2

    prepared = AFSATProblem(
        problem_file=problem_file,
        sat_parser=sat_parser,
        objectives=objectives,
        xor_rref_meta=xor_rref_meta,
        n_var=sat_parser.n_var,
        n_clause=sat_parser.n_clause,
        var_mapper=sat_parser.var_mapper,
        n_never_sat=sat_parser.n_never_sat,
    )
    return prepared, prefixes, read_time, process_time


def create_worker_session(
    prepared: AFSATProblem,
    config: AFSATConfig,
    *,
    prefix_count: int = 1,
    trace_hook: Callable[[str], None] | None = None,
) -> AFSATWorkerSession:
    def trace(message: str) -> None:
        if trace_hook is None:
            return
        trace_hook(message)

    """Build the worker's search once: mesh, objectives, solver and the per-batch programs (make_batch_programs), all
    compiled ahead of time before the worker reports ready, whatever -w says, so no request pays for compilation."""
    trace("enter")
    runtime_common = config.runtime_common
    runtime_afsat = config.runtime_afsat
    optimiser_cfg = config.optimiser
    output_cfg = config.output_logging

    n_vars = prepared.n_var
    xor_rref_meta = prepared.xor_rref_meta
    n_devices = runtime_common.n_devices
    trace(f"config n_devices={n_devices} prefix_count={prefix_count}")
    devices = jax.devices("gpu")[:n_devices]
    trace(f"resolved devices={len(devices)}")
    if not devices:
        raise RuntimeError("No GPU devices were selected")

    trace("building mesh")
    mesh, obj_sharding, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    trace("sharding objectives")
    sharded_objs = shard_tree(prepared.objectives, obj_sharding)
    trace("allocating objective weights")
    weights = tuple(jnp.full((obj.clauses.lits.shape[0],), 1.0, dtype=float) for obj in sharded_objs)
    weights = shard_tree(weights, obj_sharding)

    trace("building solver")
    replicated = NamedSharding(mesh, jax.P())
    xor_dropped = runtime_afsat.drop_xor_eval and xor_rref_meta is not None and optimiser_cfg.name != "unbounded"
    dep_idx = np.asarray(xor_rref_meta.dependent_indices) if xor_dropped and xor_rref_meta is not None else None
    fold_rows = fold_layout = None
    xor_rows: tuple = ()
    if xor_dropped and xor_rref_meta is not None and optimiser_cfg.name in ("pgd", "lbfgsb"):
        fold_rows = rref_rows(xor_rref_meta)
        fold_layout = make_fold_layout(fold_rows, n_vars)
        xor_rows = (jax.device_put(pack_rows(fold_layout, fold_rows), replicated),)  # the unfolded rows always fit
        logger.info(f"Per-prefix RREF folding: {len(fold_rows)} rows in buckets of width {fold_layout.widths}, "
                    f"capacities {fold_layout.capacities}")
    solver = build_solver(config, sharded_objs, xor_rref_meta, None, folded_xor=fold_layout is not None)

    seed = random_seed() if runtime_common.rand_seed == -1 else runtime_common.rand_seed
    logger.info(f"seed={seed}, rand_seed={runtime_common.rand_seed}")
    trace(f"seed initialized={seed}")
    key = jax.random.PRNGKey(np.array(seed))
    f_key = jax.random.PRNGKey(np.array(seed + 1))

    # Every worker request goes through process_prefix_line, which always yields a (n_vars,)
    # vector with the unit prefix merged in; compile for exactly that (zeros if there are no units).
    unit_vector = jnp.asarray(prepared.sat_parser.process_prefix_line([]))[None, :]

    batch = runtime_afsat.batch_per_device
    guess_batch = 0
    mem_cap = None
    if batch == -1:
        trace("batch=-1 entering heuristic sizing")
        logger.info("Guessing optimal batch size")
        # Estimate per-point memory by compiling at the floor size (analysed, never executed).
        guess_batch = BATCH_FLOOR * n_devices
        trace("materializing x_guess")
        x_guess, empty_prefix = pin_warmup_inputs(
            jax.random.uniform(key, minval=0.99 - (5e-2), maxval=0.99, shape=(guess_batch, n_vars)),
            unit_vector,
        )
        x_guess = jax.device_put(x_guess, batch_sharding)
        empty_prefix = jax.device_put(empty_prefix, batch_sharding)
        w_guess = tuple((w - 1e-4) for w in weights)
        trace("estimating peak memory")
        # memory_analysis() describes the per-device program, so divide by points per device.
        peak_mem = solver.peak_memory_estimation(x_guess, empty_prefix, w_guess)
        mem_est_per_point = peak_mem // BATCH_FLOOR
        batch, mem_cap = heuristic_batch_per_device(devices[0], mem_est_per_point)
        trace(f"peak memory estimated per_point={mem_est_per_point} cap={mem_cap}")
    else:
        trace(f"fixed batch mode batch={batch}")

    trace("adjusting final batch")
    batch = adjust_batch(devices, batch, mem_cap, prefix_count)
    trace(f"adjusted batch={batch}")

    # The per-batch programs, compiled ahead of time from each other's output shapes and shardings, as run_solver's
    # warm-up does: requests then compile nothing (a worker answers many requests, so this is never skipped).
    draw, summarise, _ = make_batch_programs(
        batch, n_vars, runtime_common.pt_sampler, False, False, runtime_common.weight_decay, weights, batch_sharding,
    )
    sampler_state = jax.device_put(key, replicated)
    penalty = jax.device_put(jnp.zeros((sum(int(w.shape[0]) for w in weights),), dtype=jnp.int64), replicated)
    trace("compiling")
    compile_start = time()
    _, x_spec, fixed_spec, _ = draw.lower(sampler_state, unit_vector).compile().out_info
    kernel = solver.compile(x_spec, fixed_spec, weights, *xor_rows)
    out = kernel.out_info  # x_opt, unsat, iters, unsat_ct, aux
    summarise.lower(out[0], out[1], out[3], penalty).compile()
    peak_mem = solver.peak_bytes(kernel)
    mem_limit = (devices[0].memory_stats() or {}).get("bytes_limit") or 1
    logger.info(
        f"Compiled in {time() - compile_start:.2f} s: batch {x_spec.shape[0]}, peak memory p.d. {peak_mem}, "
        f"peak/point {peak_mem // max(batch // n_devices, 1)}, frac of device memory {peak_mem / mem_limit:.4f}"
    )
    trace(f"compiled elapsed={time() - compile_start:.3f}s")

    # As run_solver: never-satisfiable soft clauses are not in the kernel's counts, so the threshold drops by them.
    unsat_h = int(runtime_common.unsat_thresh * 2 * n_vars) if runtime_common.unsat_thresh else 0
    if prepared.n_never_sat:
        unsat_h = max(unsat_h - prepared.n_never_sat, 0)

    trace("returning worker session")
    return AFSATWorkerSession(
        prepared=prepared,
        config=config,
        objs=sharded_objs,
        weights=weights,
        batch=batch,
        solvers={False: solver},
        dep_idx=dep_idx,
        fold_rows=fold_rows,
        fold_layout=fold_layout,
        replicated=replicated,
        draw=draw,
        summarise=summarise,
        sampler_state=sampler_state,
        penalty=penalty,
        f_key=f_key,
        unit_vector=unit_vector,
        x_spec=x_spec,
        fixed_spec=fixed_spec,
        unsat_h=unsat_h,
        n_never_sat=prepared.n_never_sat,
        fuzz_limit=runtime_afsat.fuzz,
        counting=bool(runtime_common.counting),
        stats=output_cfg.stats,
        binary_v=output_cfg.binary_v,
        warmup_done=True,
    )


def _worker_request(session: AFSATWorkerSession, prefix_vectors: Array) -> tuple[Optimiser, tuple, Array]:
    """The solver, its extra kernel arguments and the prefix for one request (AFSATWorkerSession).

    With per-prefix folding: fold the prefix into the RREF, add the dependents it determines to the prefix, and pass
    the folded rows; a prefix the fold refutes raises UnsatError (the worker logs it and runs no descent). Otherwise,
    or if the folded rows overflow the layout: the kernel that keeps the XOR evaluator when the prefix fixes an RREF
    dependent, built and compiled the first time a prefix needs it."""
    if session.fold_layout is not None and session.fold_rows is not None:
        prefix = np.asarray(prefix_vectors)[0]
        folded = fold_prefix(session.fold_rows, prefix)
        if folded is None:
            raise UnsatError("cube contradicts the XOR system (refuted by the RREF fold)")
        if folded.forced:
            prefix = prefix.copy()
            for var, bit in folded.forced.items():
                prefix[var] = -1 if bit else 1  # -1 is TRUE
            prefix_vectors = jnp.asarray(prefix)[None, :]
        packed = pack_rows(session.fold_layout, folded.rows)
        if packed is not None:
            return session.solvers[False], (jax.device_put(packed, session.replicated),), prefix_vectors
        logger.warning("Folded RREF rows overflow the slot layout; this request keeps the XOR evaluator instead")
        keep = True
    else:
        keep = session.dep_idx is not None and bool(np.any(np.asarray(prefix_vectors)[:, session.dep_idx] != 0))
    if keep not in session.solvers:
        logger.info("Prefix fixes an RREF dependent variable: compiling the kernel that keeps the XOR evaluator")
        solver = build_solver(session.config, session.objs, session.prepared.xor_rref_meta, None, keep_xor_eval=True)
        solver.compile(session.x_spec, session.fixed_spec, session.weights)
        session.solvers[keep] = solver
    return session.solvers[keep], (), prefix_vectors


def run_worker_single_batch(session: AFSATWorkerSession, prefix_vectors: Array | None = None) -> AFSATBatchResult:
    """One batch for one prefix, on the compiled programs of create_worker_session: draw the starting points (prefix
    applied), descend, and summarise on the device; the best point's count comes back, and with --stats the
    per-descent statistics as well. Fuzzing, if enabled, is as in run_solver."""
    start_batch = time()
    if prefix_vectors is None:
        prefix_vectors = session.unit_vector
    solver, extra, prefix_vectors = _worker_request(session, prefix_vectors)
    weights, summarise = session.weights, session.summarise

    session.sampler_state, x0, fixed_vars, x0_positive = session.draw(session.sampler_state, prefix_vectors)
    opt_x, opt_unsat, opt_iters, opt_unsat_ct, aux_info = solver.run(x0, fixed_vars, weights, *extra)  # x0 donated
    b_unsat, _, best_x_dev, best_row_dev, _ = summarise(opt_x, opt_unsat, opt_unsat_ct, session.penalty)
    best_unsat = int(jax.device_get(b_unsat))

    threshold_hit = bool(session.unsat_h and best_unsat <= session.unsat_h) or bool(
        session.n_never_sat and best_unsat == 0
    )
    found_sol = threshold_hit or best_unsat == 0

    if session.fuzz_limit and (session.counting or not found_sol):
        fuzz_attempt = 0
        F_x = opt_x
        session.f_key, s_f_key = jax.random.split(session.f_key)
        fuzz = jax.random.uniform(s_f_key, minval=1e-7, maxval=1e-2, shape=opt_x.shape)
        fuzz_mag = 1
        while fuzz_attempt < session.fuzz_limit:
            fuzz_mask = np.ones(F_x.shape, dtype=bool)
            if found_sol:
                sol_locs = np.flatnonzero(np.asarray(opt_unsat_ct) < 1)
                fuzz_mask[sol_locs, :] = False  # do not fuzz the solutions

            fuzz_attempt += 1
            fuzz_adj = jnp.sign(fuzz) * jnp.abs(fuzz) ** (1 / fuzz_mag) if fuzz_mag != 1 else fuzz

            # Back onto the hypercube; a new array, so donating it leaves opt_x intact.
            F_x = jnp.clip(jnp.where(fuzz_mask, F_x + fuzz_adj, F_x), -1, 1)
            F_x_sign = jnp.sign(F_x)  # F_x is donated to the solver below

            F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct, _ = solver.run(F_x, fixed_vars, weights, *extra)
            F_unsat, _, F_x_best, F_row, _ = summarise(F_opt_x, F_opt_unsat, F_opt_unsat_ct, session.penalty)
            F_best = int(jax.device_get(F_unsat))
            if F_best < best_unsat:
                # The fuzzed descent beat the batch: keep its descended points and their best point.
                best_unsat, best_x_dev, best_row_dev = F_best, F_x_best, F_row
                opt_x, opt_unsat, opt_iters, opt_unsat_ct = F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct

            if F_best == 0:
                found_sol = True
                if not session.counting:
                    break

            # No point changed sign during the descent: the fuzz was too small to move anything, so strengthen it.
            fuzz_mag = fuzz_mag + 1 if bool((F_x_sign == jnp.sign(F_opt_x)).all()) else 1
            F_x = F_opt_x

    iters = unsat_per_point = flips_per_point = []
    eval_per_point: list[float] = []
    if session.stats:
        n_clause = session.prepared.n_clause
        eval_last = np.abs(np.asarray(aux_info[-1]))
        eval_oob = (eval_last > n_clause) & (~np.isclose(eval_last, float(n_clause)))
        if np.any(eval_oob):
            exceed = np.argwhere(eval_oob).flatten()
            logger.warning(
                f"Detected numerical instability! Abs(eval) > {n_clause} outside tolerance at indices {exceed}: "
                f"{np.asarray(aux_info[-1])[exceed]}"
            )
            if QUIT_ON_ANOMALY:
                raise FloatingPointError("Numerical instability encountered and anomaly_quit is enabled")
        iters = np.asarray(opt_iters).flatten().astype(int).tolist()
        unsat_per_point = np.asarray(opt_unsat_ct).flatten().astype(int).tolist()
        eval_per_point = np.asarray(aux_info[1]).squeeze().T.flatten().astype(float).tolist()
        flips_per_point = (np.asarray(opt_x > 0).sum(axis=1) - np.asarray(x0_positive)).astype(int).tolist()

    signed_best = tuple(np.sign(np.asarray(best_x_dev)).astype(int).tolist())
    best_str = session.prepared.var_mapper.assn_str(signed_best, session.binary_v, inc_zero=True)

    return AFSATBatchResult(
        best_unsat=best_unsat + session.n_never_sat,
        # With never-satisfiable soft clauses, satisfying every other clause is the MaxSAT optimum, not SAT.
        sat=best_unsat == 0 and not session.n_never_sat,
        threshold_hit=threshold_hit,
        best_assignment_signed=signed_best,
        best_assignment_str=best_str,
        best_unsat_clause_indices=np.flatnonzero(np.asarray(best_row_dev)).astype(int).tolist(),
        iters=iters,
        unsat_per_point=unsat_per_point,
        eval_per_point=eval_per_point,
        flips_per_point=flips_per_point,
        elapsed_sec=time() - start_batch,
    )


def run_solver(
    config: AFSATConfig,
    n_vars: int,
    n_clause: int,
    objs: tuple[Objective, ...],
    xor_rref_meta: XorRREFMetadata | None = None,
    prefix_vectors: Array | None = None,
    *,
    var_mapper: VarMapper,
    n_never_sat: int = 0,
) -> float:
    runtime_common = config.runtime_common
    runtime_afsat = config.runtime_afsat
    optimiser_cfg = config.optimiser
    output_cfg = config.output_logging

    timeout = runtime_common.timeout_sec
    batch = runtime_afsat.batch_per_device
    restart_thresh = runtime_common.restart_f
    fuzz_limit = runtime_afsat.fuzz
    n_devices = runtime_common.n_devices
    sample_method = runtime_common.pt_sampler
    optimiser = optimiser_cfg.name
    warmup = runtime_afsat.warmup
    benchmark = runtime_common.benchmark
    counting = int(runtime_common.counting)
    rand_seed = runtime_common.rand_seed
    ffsat_seed = runtime_common.ffsat_seed
    weight_decay = runtime_common.weight_decay
    unsat_h = int(runtime_common.unsat_thresh * 2 * n_vars) if runtime_common.unsat_thresh else 0
    # MaxSAT mode: soft clauses no assignment satisfies are not in the objectives, so the kernel's counts omit them.
    # The threshold applies to the full cost, every reported cost adds them back, and with any present a kernel count
    # of 0 is the optimum (cost n_never_sat) rather than a satisfying assignment.
    if n_never_sat:
        unsat_h = max(unsat_h - n_never_sat, 0)
        logger.warning(f"MaxSAT mode: {n_never_sat} clauses can never be satisfied; the optimum cost is at least that")
    binary_v = output_cfg.binary_v

    devices = jax.devices("gpu")[:n_devices]
    n_prefix = len(prefix_vectors) if prefix_vectors is not None else 1

    mesh, obj_sharding, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    # Construct weights, and shard both weights and objectives.
    objs = shard_tree(objs, obj_sharding)
    weights = tuple(jnp.full((obj.clauses.lits.shape[0],), 1.0, dtype=float) for obj in objs)
    weights = shard_tree(weights, obj_sharding)

    # Construct pure JAX functions (closures) and build solver.
    solver = build_solver(config, objs, xor_rref_meta, prefix_vectors)

    seed = random_seed() if rand_seed == -1 else rand_seed
    logger.info(f"seed={seed}, rand_seed={rand_seed}")
    # logger.debug(f"seed={seed}, rand_seed={rand_seed}")
    key = jax.random.PRNGKey(np.array(seed))
    f_key = jax.random.PRNGKey(np.array(seed + 1))

    guess_batch = 0
    mem_cap = None
    if batch == -1:
        logger.info("Guessing optimal batch size")
        # Estimate per-point memory by compiling at the floor size (analysed, never executed).
        guess_batch = BATCH_FLOOR * n_devices
        x_guess, empty_prefix = pin_warmup_inputs(
            jax.random.uniform(key, minval=0.99 - (5e-2), maxval=0.99, shape=(guess_batch, n_vars)), prefix_vectors
        )
        x_guess = jax.device_put(x_guess, batch_sharding)
        empty_prefix = jax.device_put(empty_prefix, batch_sharding)
        w_guess = tuple((w - 1e-4) for w in weights)
        # memory_analysis() describes the per-device program, so divide by points per device.
        peak_mem = solver.peak_memory_estimation(x_guess, empty_prefix, w_guess)
        mem_est_per_point = peak_mem // BATCH_FLOOR
        batch, mem_cap = heuristic_batch_per_device(devices[0], mem_est_per_point)
        logger.info(f"Estimated peak memory per point: {mem_est_per_point}; memory cap {mem_cap} p.d.")

    batch = adjust_batch(devices, batch, mem_cap, n_prefix)

    # The per-batch work around the descent is one compiled program each (make_batch_programs), compiled at warm-up
    # below. The main loop then runs no one-off array operations (each would compile on first use, inside the timer)
    # and moves two scalars to the host per batch. --stats adds the per-descent statistics and anomaly checks; the
    # progress display draws on them.
    stats = output_cfg.stats or not benchmark
    reweight = bool(restart_thresh)
    draw, summarise, reweight_weights = make_batch_programs(
        batch, n_vars, sample_method, ffsat_seed, reweight, weight_decay, weights, batch_sharding
    )
    # Placed as the programs return them (replicated), so the first call and every later one share one argument type.
    replicated = NamedSharding(mesh, jax.P())
    sampler_state = jax.device_put(jnp.asarray(0, dtype=jnp.int64) if ffsat_seed else key, replicated)
    penalty = jax.device_put(jnp.zeros((sum(int(w.shape[0]) for w in weights),), dtype=jnp.int64), replicated)

    if warmup:
        # Warm-up is compilation ahead of time, with no dummy descent: the sampler, the batch kernel and the
        # bookkeeping, each lowered on its own jitted object for the real batches' shapes and shardings (taken from
        # the output of the program before it), so the first real batch reuses all three and the timer holds no
        # compilation. The kernel's compile also gives X-PEAK. Without -w they compile in the first batch instead.
        logger.info("Warmup (ahead-of-time compilation)")
        warm_start = time()
        _, x_spec, fixed_spec, _ = draw.lower(sampler_state, prefix_vectors).compile().out_info
        kernel = solver.compile(x_spec, fixed_spec, weights)
        out = kernel.out_info  # x_opt, unsat, iters, unsat_ct, aux
        if len(out) == 5:
            penalty_spec = summarise.lower(out[0], out[1], out[3], penalty).compile().out_info[-1]
            if reweight:
                reweight_weights.lower(weights, penalty_spec).compile()
        logger.info(f"Warmup Complete {time() - warm_start}")

        if not benchmark and logger.isEnabledFor(logging.INFO):
            if mesh.shape["batch"] > 1:
                logger.info("Batch sharding:")
                jax.debug.visualize_sharding(x_spec.shape, x_spec.sharding)
            if mesh.shape["objective"] > 1:
                logger.info("Objective sharding:")
                jax.debug.visualize_array_sharding(objs[0].clauses.lits)

        peak_mem = solver.peak_bytes(kernel)
        mem_est_per_point = peak_mem // batch  # X-PEAKPP keeps its historical definition
        mem_limit = (devices[0].memory_stats() or {}).get("bytes_limit") or 1
        logger.info(
            f"Warmup: shape - {x_spec.shape[0]}, peak memory p.d. - {peak_mem}, "
            + f"peak/point - {peak_mem // (batch // n_devices)}, frac of device memory - {peak_mem / mem_limit:.4f}"
        )

    all_sols: dict[tuple[int, ...], int] = defaultdict(int)
    first_sol: tuple[int, ...] | None = None
    best_x_dev = best_row_dev = None  # the best point so far and its unsatisfied clauses, kept on the device
    ttfs = 0
    best_unsat = math.inf
    batches_done = 0
    restart_ct = 0
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
            desc=f"{desc} [MAX-SAT cost: {best_unsat + n_never_sat}]",
            bar_format="{l_bar}{bar}|{elapsed}/" + f"{str(timeout_m).zfill(2)}:{str(timeout_s).zfill(2)}" + "{postfix}",
            postfix=f"{0:.2f}s/it",
        )
    accum_time_descent = 0

    t0 = time()
    while (time() - t0 < timeout) and (not solver.warmup_sol or counting):
        start_batch = time()
        tloop = time()

        # Sample, descend (x0 is donated to the solver), and pick the batch's best point; the only transfer to the
        # host is the best point's unsatisfied count and index.
        sampler_state, x0, fixed_vars, x0_positive = draw(sampler_state, prefix_vectors)
        opt_x0, opt_unsat, opt_iters, opt_unsat_ct, aux_info = solver.run(x0, fixed_vars, weights)
        b_unsat, b_loc, b_x, b_row, penalty = summarise(opt_x0, opt_unsat, opt_unsat_ct, penalty)
        batch_best_unsat, batch_best_loc = (int(v) for v in jax.device_get((b_unsat, b_loc)))
        accum_time_descent += time() - tloop

        # Flag and bail if we encounter anomalous behaviour
        if stats and logger.isEnabledFor(logging.WARNING):
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
                    logger.warning(
                        f"[{optimiser}] Optimizer returned out-of-bounds points at: {escaped}\n"
                        + f"Points: {opt_x0[escaped]}"
                    )

                aux_x_abs = np.abs(np.asarray(aux_info[0]))
                aux_x_oob = (aux_x_abs > 1.0) & (~np.isclose(aux_x_abs, 1.0))
                if np.any(aux_x_oob):
                    escaped_aux = np.argwhere(aux_x_oob)
                    logger.warning(
                        f"[{optimiser}] Auxiliary evaluation points left bounds at: {escaped_aux}\n"
                        + f"Points: {aux_info[0][escaped_aux]}"
                    )
                if QUIT_ON_ANOMALY:
                    return 0.0

        if batch_best_unsat < best_unsat:
            best_unsat, best_x_dev, best_row_dev = batch_best_unsat, b_x, b_row

        tbatch = time()
        found_sol = False

        if (unsat_h and batch_best_unsat <= unsat_h) or (n_never_sat and batch_best_unsat == 0):
            ttfs = time() - t0
            found_sol = True
            batches_done += 1
            end_batch = time()
            break

        if batch_best_unsat == 0:
            if ttfs == 0:
                ttfs = tbatch - t0
            found_sol = True

        if fuzz_limit and (counting or not found_sol):
            # Knock current batch in attempt to find more solutions.
            fuzz_attempt = 0
            F_x = opt_x0[:]
            f_key, s_f_key = jax.random.split(f_key)
            fuzz = jax.random.uniform(s_f_key, minval=1e-7, maxval=1e-2, shape=opt_x0.shape)
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
                F_x_sign = jnp.sign(F_x)  # F_x is donated to the solver below

                F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct, (_, F_eval_scores) = solver.run(
                    F_x, fixed_vars, weights
                )

                F_unsat, F_loc, F_x_best, F_row, _ = summarise(F_opt_x, F_opt_unsat, F_opt_unsat_ct, penalty)
                F_batch_best_unsat, F_batch_best_loc = (int(v) for v in jax.device_get((F_unsat, F_loc)))
                if F_batch_best_unsat < best_unsat:
                    # TODO: We make a (spurious?) assumption that bumping a solution will find that solution again
                    # So this check needs to be adjusted to also check the number the solutions and replace only if
                    # find more - this covers both the counting and not counting case.
                    # TODO: This would also subsume the next check somewhat - since found_sol would already be true.
                    # We beat the unfuzzed convergence, so keep this result instead (the descended points, and their
                    # best point; the earlier code kept the pre-descent points and indexed the unfuzzed batch).
                    best_unsat, best_x_dev, best_row_dev = F_batch_best_unsat, F_x_best, F_row
                    batch_best_unsat, batch_best_loc, b_x = F_batch_best_unsat, F_batch_best_loc, F_x_best
                    opt_x0, opt_unsat, opt_iters, opt_unsat_ct = F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct

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

                if (F_x_sign == jnp.sign(F_opt_x)).all():
                    # If no points ended up changing signs after convergence, we didn't move at all. Increase magnitude
                    fuzz_mag += 1
                else:
                    fuzz_mag = 1

                F_x = F_opt_x

        batches_done += 1
        end_batch = time()

        if stats:
            opt_iters_local = np.asarray(opt_iters).flatten().tolist()
            # first argument is the second to last value x had in the descent
            eval_scores = jnp.array(aux_info[1]).squeeze().T
            flips = jnp.sum(opt_x0 > 0, axis=1) - x0_positive
            all_unsats.extend(np.asarray(opt_unsat_ct).flatten().tolist())
            all_evals.extend(np.asarray(eval_scores).flatten().tolist())
            all_iters.extend(opt_iters_local)
            all_flips.extend(np.asarray(flips).flatten().tolist())

        if found_sol:
            first_sol = tuple(np.sign(np.asarray(b_x)).astype(int).tolist())
            if counting:
                sol_locs = np.flatnonzero(np.asarray(opt_unsat_ct) < 1)
                for row in np.sign(np.asarray(opt_x0)[sol_locs, :]):
                    all_sols[tuple(row.astype(int).tolist())] += 1
            else:
                all_sols[first_sol] += 1
                if not benchmark:
                    # Close tqdm.
                    # TODO: We probably don't need this at all if we change tqdm usage (progress) to context manager.
                    # TODO: we should change benchmark to track "progress instead" as the following is much clearer:
                    # # if progress and not counting:
                    for x in range(len(histbars)):
                        histbars[x].close()
                    for x in range(len(infobars)):
                        infobars[x].close()
                    pbar.close()
                    logger.info(f"SAT! at sample {max(batches_done, 0) * batch + batch_best_loc}")
                break

        if reweight:
            # summarise() has added this batch's unsatisfied counts per clause to the penalty.
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
                # The penalty sums the unsatisfied counts over the restart_thresh batches since the last update (the
                # earlier code concatenated the batches along the clause axis, so above 1 it used the first batch's).
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"# Restart: {restart_ct} | Best MAX-SAT cost (#unsat): {best_unsat + n_never_sat}")
                    logger.debug(f"Unsat counts: {penalty}, \nBest: {np.flatnonzero(np.asarray(best_row_dev))}")
                weights, penalty = reweight_weights(weights, penalty)
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
            pbar.set_description(f"{desc} [MAX-SAT cost: {best_unsat + n_never_sat}]")
            if pbn + batch_elapsed > timeout:
                pbar.update(timeout - pbelapse)
            else:
                pbar.update(end_batch - start_batch)  # update *adds* the input to the counter.
            pbar.set_postfix_str(f"({(pbelapse / batches_done):.2f}s/it)")

    tsolve = time() - t0
    best_x = np.asarray(best_x_dev) if best_x_dev is not None else np.zeros(n_vars)
    best_unsat_clauses_idx = np.flatnonzero(np.asarray(best_row_dev)) if best_row_dev is not None else np.array([0])
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
    print(f"c {'-' * len(sol_info)}")
    print(f"c {sol_info}")
    print(f"c {'-' * len(sol_info)}")

    if len(all_sols):
        print("s SATISFIABLE")
        for sol_i, sol in enumerate(all_sols.keys()):
            if not counting and first_sol != sol:
                continue
            sol_str = var_mapper.assn_str(sol, binary_v)
            # var_mapper.bin_str(sol) if binary_v else var_mapper.lits_str(sol)
            if counting:
                logger.info(f"{sol_i + 1}: v {sol_str} 0")
                if not sol_i:
                    print("c ENUMERATING - example solution (of possibly many):")
                    print(f"v {sol_str} 0")
            elif first_sol == sol:
                print(f"v {sol_str} 0")
                break
    else:
        signed_best = tuple(np.sign(best_x).astype(int).tolist())
        assign_str = var_mapper.assn_str(signed_best, binary_v, inc_zero=True)
        # var_mapper.bin_str(signed_best) if binary_v else var_mapper.lits_str(signed_best, include_unknown=True)

        # Every satisfiable clause satisfied, with some never satisfiable: the MaxSAT optimum.
        print("s OPTIMUM FOUND" if n_never_sat and best_unsat == 0 else "s UNKNOWN")
        print("c Best found MAX-SAT assignment (zero energy variables omitted or -):")
        print(f"o {best_unsat + n_never_sat}")
        print(f"v {assign_str} 0")

        assignment = {
            var_mapper.map_to_input(var_idx_0 + 1) * -assigned
            for var_idx_0, assigned in enumerate(signed_best)
            if assigned
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("UNSATS (set literals-clause literals):\n")
            for i, cl_idx in enumerate(best_unsat_clauses_idx.flatten()):
                find_idx = cl_idx
                for obj in objs:
                    obj_len = obj.clauses.lits.shape[0]
                    if find_idx < obj_len:
                        if len(obj.clauses.sign.shape) > 1:
                            dense_clause = ((obj.clauses.lits[find_idx] + 1) * obj.clauses.sign[find_idx]).tolist()
                        else:
                            dense_clause = ((obj.clauses.lits[find_idx] + 1) * obj.clauses.sign).tolist()
                        clause = {var_mapper.map_to_input(int(lit)) for lit in dense_clause if lit}
                        logger.debug(f"{(sorted(clause.intersection(assignment)), clause)}")
                        break
                    else:
                        find_idx -= obj_len
            for i, w in enumerate(weights):
                logger.debug(
                    f"\t{set((np.array(objs[i].clauses.types)).flatten().tolist())},{objs[i].clauses.lits.shape} \n{w}"
                )
    print(f"c {'-' * len(sol_info)}")

    if logger.isEnabledFor(logging.INFO):
        if ttfs:
            logger.info(f"X-TTFS {ttfs}")
        logger.info("START EXP")
        logger.info(f"X-CLAUSES {n_clause}")
        logger.info(f"X-VARS {n_vars}")
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
        if stats:
            logger.info(f"X-ITERHISTO {dict(Counter(all_iters))}")
            logger.info(f"X-RAWEVAL {all_evals}")
            logger.info(f"X-RAWUNSAT {all_unsats}")
            logger.info(f"X-RAWITER {all_iters}")
            logger.info(f"X-RAWFLIPS {all_flips}")
        logger.info("END EXP")
    return tsolve


def main(problem_file: str, config: AFSATConfig) -> None:
    prepared, prefixes, read_time, process_time = prepare_problem(problem_file, config)

    t_solve = run_solver(
        config,
        prepared.n_var,
        prepared.n_clause,
        prepared.objectives,
        xor_rref_meta=prepared.xor_rref_meta,
        prefix_vectors=prefixes,
        var_mapper=prepared.var_mapper,
        n_never_sat=prepared.n_never_sat,
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
    ap.add_argument("problem_file", help="The file to process")
    ap.add_class_arguments(AFSATConfig, nested_key="config")
    ap.set_defaults(**{"config.runtime_common.n_devices": n_devices})
    ap.link_arguments(
        "config.runtime_common.benchmark",
        "config.runtime_common.progress_enabled",
        compute_fn=lambda benchmark: not benchmark,
        apply_on="parse",
    )

    def make_option_group(title: str, section: str):
        group = ap.add_argument_group(title)

        def add_opt(*flags: str, field: str, **kwargs) -> None:
            kwargs.setdefault("default", SUPPRESS)
            group.add_argument(*flags, dest=f"config.{section}.{field}", **kwargs)

        return add_opt

    # Backward-compatible short aliases grouped by config sections.
    io_opts = make_option_group("IO Aliases", "invocation")
    io_opts("-y", "--profile", action="store_true", field="profile_enabled", help="Enable profiling")
    io_opts("--cache", type=str, field="disk_cache", help="Disk cache for FFT matrices")
    io_opts("-p", "--prefix", type=str, field="prefix_file", help="Fixed assignments file, each a line of assignments")

    runtime_common_opts = make_option_group("Runtime Common Aliases", "runtime_common")
    runtime_common_opts("-t", "--timeout", type=int, field="timeout_sec", help="Maximum runtime (timeout seconds)")
    runtime_common_opts("-r", "--restart_f", type=int, field="restart_f", help="Batches before reweighting (0 = never)")
    runtime_common_opts("-n", "--n_devices", type=int, field="n_devices", help="Devices (eg. GPUs) to use. 0 uses all")
    runtime_common_opts("-e", "--benchmark", action="store_true", field="benchmark", help="Benchmarking (less output)")
    runtime_common_opts("--progress", action="store_false", field="benchmark", help="Display progress (impl. -e False)")
    runtime_common_opts("-c", "--counting", action="store_true", field="counting", help="#SAT - Enum sols to timeout")
    runtime_common_opts("-s", "--seed", type=int, field="rand_seed", help="Force initialise seed (non-negative int)")
    runtime_common_opts("--ffseed", field="ffsat_seed", action="store_true", help="FFSAT Key increments")
    runtime_common_opts("-u", "--unsat_thresh", type=float, field="unsat_thresh", help="MAXSAT - #UNSAT stop threshold")
    runtime_common_opts("-m", "--sampler", type=str, field="pt_sampler", choices=SAMPLERS, help="Initial point sampler")

    runtime_afsat_opts = make_option_group("Runtime AFSAT Aliases", "runtime_afsat")
    runtime_afsat_opts("-b", "--batch", type=int, field="batch_per_device", help="Batch size. -1 = heuristic maximum")
    runtime_afsat_opts("-f", "--fuzz", type=int, field="fuzz", help="Number of times to attempt fuzzing per batch")
    runtime_afsat_opts(
        "-w", "--warmup", action="store_true", field="warmup", help="Compile the solver before the timer starts"
    )
    runtime_afsat_opts("--xor_rref", action="store_true", field="xor_rref", help="Enable XOR GJ Elim (RREF projection)")
    runtime_afsat_opts(
        "--no_propagate", action="store_false", field="propagate", help="Disable full-formula unit propagation"
    )
    runtime_afsat_opts(
        "--keep_xor_eval", action="store_false", field="drop_xor_eval", help="Keep the XOR objective under RREF"
    )

    optimiser_opts = make_option_group("Optimiser Aliases", "optimiser")
    optimiser_opts("-i", "--iters_desc", type=int, field="max_iters", help="Solver maximum iterations")
    optimiser_opts("-q", "--solver_tol", type=float, field="tolerance", help="Optimiser convergence tolerance")
    optimiser_opts("-o", "--optimiser", "--optimizer", type=str, field="name", help="Optimiser algorithm")

    output_opts = make_option_group("Output Aliases", "output_logging")
    output_opts("-d", "--debug", choices=LOG_LEVELS, field="debug_level", help=f"Set logging level ({LOG_LEVELS})")
    output_opts("--stdout_log", action="store_true", field="stdout_log", help="Logs output to stdout instead of stderr")
    output_opts("--anomaly_quit", action="store_true", field="anomaly_quit")
    output_opts("--log_propagate", action="store_true", field="log_propagate")
    output_opts("--binary_v", action="store_true", field="binary_v", help="Short form solution string")
    output_opts(
        "--stats", action="store_true", field="stats",
        help="Per-descent statistics (X-RAW*, X-ITERHISTO) and anomaly checks; off = production loop",
    )
    output_opts("--debug_nans", action="store_true", field="debug_nans", help="Enable jax_debug_nans (slow)")

    parsed = ap.parse_args()
    instantiated = ap.instantiate(parsed)
    config = instantiated.config
    config.runtime_common.n_devices = config.runtime_common.n_devices or n_devices
    jax.config.update("jax_debug_nans", config.output_logging.debug_nans)

    QUIT_ON_ANOMALY = config.output_logging.anomaly_quit
    # Keep module logs routed through root handlers configured below.
    logger.propagate = True
    logging.basicConfig(
        level=getattr(logging, config.output_logging.debug_level.upper()),
        # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        format="c %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout if config.output_logging.stdout_log else sys.stderr),
            # Optional: logging.FileHandler('sat_loader.log')  # Also log to file
        ],
    )

    # Run with or without profiler based on the flag
    profiler = (
        jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False)
        if config.invocation.profile_enabled
        else nullcontext()
    )
    with profiler:
        main(parsed.problem_file, config)
        if config.invocation.profile_enabled:
            jax.profiler.save_device_memory_profile("memory.prof")
    logger.debug(f"Running with configuration: {config}")
