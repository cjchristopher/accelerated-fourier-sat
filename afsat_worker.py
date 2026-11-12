# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# ruff: noqa: I001
from __future__ import annotations

from argparse import SUPPRESS
import atexit
import logging
import os
import sys
from time import perf_counter as time

import jax
import jax.numpy as jnp
import numpy as np
from jsonargparse import ArgumentParser as ArgParse

from afsat import (
    create_worker_session,
    prepare_problem,
    run_worker_single_batch,
)
from advice_pool import AdvicePool, LineReader
from sat_loader import UnsatError
from samplers import SAMPLERS
from utils import AFSATConfig, LOG_LEVELS

logger = logging.getLogger(__name__)
BRIDGE_TRACE_FH = None


class BridgeEmitHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            emit_log(f"pylog {msg}")
        except Exception as e:
            return


def _close_bridge_trace_file() -> None:
    global BRIDGE_TRACE_FH
    if BRIDGE_TRACE_FH is not None:
        try:
            BRIDGE_TRACE_FH.flush()
            BRIDGE_TRACE_FH.close()
        except Exception as e:
            pass
        BRIDGE_TRACE_FH = None


def init_bridge_trace_file(
    log_file: str | None,
    *,
    world_rank: int,
    helper_rank: int,
    phase: int,
) -> str | None:
    global BRIDGE_TRACE_FH

    chosen = (log_file or "").strip() or os.getenv("AFSAT_BRIDGE_LOG_FILE", "").strip()
    if not chosen:
        log_dir = os.getenv("AFSAT_BRIDGE_LOG_DIR", "/tmp").strip() or "/tmp"
        os.makedirs(log_dir, exist_ok=True)
        chosen = os.path.join(
            log_dir,
            f"afsat_bridge_w{world_rank}_h{helper_rank}_p{phase}_{os.getpid()}.log",
        )

    try:
        BRIDGE_TRACE_FH = open(chosen, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        BRIDGE_TRACE_FH.write("=== afsat bridge trace start ===\n")
        BRIDGE_TRACE_FH.flush()
        return chosen
    except OSError as err:
        print(f"LOG bridge_trace_file_open_failed path={chosen} error={err}", flush=True)
        BRIDGE_TRACE_FH = None
        return None


atexit.register(_close_bridge_trace_file)


def emit_line(line: str) -> None:
    print(line, flush=True)
    if BRIDGE_TRACE_FH is not None:
        BRIDGE_TRACE_FH.write(line + "\n")
        BRIDGE_TRACE_FH.flush()
    # print("wrote line", flush=True)


def emit_bridge_line(command: str, lits: np.ndarray) -> None:
    payload = " ".join(str(int(v)) for v in lits.tolist())
    if payload:
        line = f"{command} {lits.size} {payload}"
    else:
        line = f"{command} 0"
    emit_line(line)


def emit_log(message: str) -> None:
    emit_line(f"LOG {message}")


def parse_prefix_line(line: str) -> tuple[np.ndarray, bool]:
    stripped = line.strip()
    if not stripped:
        return np.empty(0, dtype=np.intc), False

    parts = stripped.split()
    command = parts[0].upper()

    if command == "STOP":
        emit_log("STOP RECEIVED")
        return np.empty(0, dtype=np.intc), True

    if command != "PREFIX":
        raise ValueError(f"unknown bridge command: {command}")

    if len(parts) < 2:
        raise ValueError("PREFIX message missing length")

    try:
        length = int(parts[1])
        # emit_log(f"got length {length}")
    except ValueError as exc:
        raise ValueError(f"invalid PREFIX length: {parts[1]}") from exc

    if length < 0:
        raise ValueError(f"invalid PREFIX length: {length}")

    lits_text = parts[2:]
    if len(lits_text) < length:
        raise ValueError(f"PREFIX expected {length} literals, got {len(lits_text)}")

    if length == 0:
        return np.empty(0, dtype=np.intc), True

    try:
        # emit_log(f"attemptings get lits {type(lits_text)}, {len(lits_text)}")
        # emit_log(f"attemptings get lits {" ".join(lits_text)[:100]} ...")
        lits = np.asarray([int(x) for x in lits_text[:length]], dtype=np.intc)
        # emit_log(f"got lits {lits}")
    except ValueError as exc:
        raise ValueError("PREFIX contains non-integer literal") from exc
    # emit_log(f"returning")
    return lits, False


def assignment_vector_to_lits(assign_vec: np.ndarray, var_mapper=None, order: np.ndarray | None = None) -> np.ndarray:
    """Map an AFSAT assignment vector in {-1,0,+1} (-1 is TRUE) back to signed literals in the input file's
    variable numbering, in index order or in the given order of variable indices."""
    nz = np.nonzero(assign_vec)[0] if order is None else order[assign_vec[order] != 0]
    if nz.size == 0:
        return np.empty(0, dtype=np.intc)

    signs = np.sign(assign_vec[nz]).astype(np.intc)
    lits = (nz.astype(np.intc) + 1) * (-signs)
    if var_mapper is not None and var_mapper.dense_to_input_var:  # the loader compacted sparse variable ids
        lits = np.asarray([var_mapper.dense_to_input(int(l)) for l in lits], dtype=np.intc)
    return lits.astype(np.intc)


ADVICE_POLICIES = ("index", "agree", "random")  # random: free variables and phases at random (a control)


def agreement_advice(
    points: np.ndarray, points_unsat: np.ndarray, free: np.ndarray, elite_frac: float, best_signed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Advice ordered by agreement among the batch's best descents.

    The elite is the best ceil(elite_frac * batch) descents by violated-clause count (at least 2). For each free
    variable the elite's mean sign gives the phase (the best descent's sign breaks a tie) and its magnitude the
    agreement. Variables are ordered by agreement, then by the elite's mean |x| (how firmly the descents hold the
    value). Returns the signed phase vector, the variable order and summary statistics for the log."""
    batch = points.shape[0]
    m = min(batch, max(2, int(np.ceil(elite_frac * batch))))
    elite = np.argsort(points_unsat, kind="stable")[:m]
    signs = np.sign(points[elite])
    mean = signs.mean(axis=0)
    agree = np.abs(mean)
    margin = np.abs(points[elite]).mean(axis=0)
    phase = np.where(mean != 0, np.sign(mean), best_signed).astype(np.int8)
    idx = np.flatnonzero(free)
    order = idx[np.lexsort((-margin[idx], -agree[idx]))]
    unanimous = int(np.count_nonzero(agree[idx] == 1.0))
    stats = {"elite": m, "elite_unsat_min": int(points_unsat[elite[0]]), "elite_unsat_max": int(points_unsat[elite[-1]]),
             "free": int(idx.size), "unanimous": unanimous,
             "agree_q50": float(np.median(agree[idx])) if idx.size else 0.0}
    return phase, order, stats


def run_worker(
    problem_file: str,
    config: AFSATConfig,
    *,
    phase: int,
    helper_rank: int,
    helper_count: int,
    world_rank: int,
    suggestion_size: int,
) -> None:
    emit_log(f"GPUS: {jax.devices('gpu')}")
    start_time = time()
    emit_log(f"start_time set {start_time}, preparing problem")
    prepared, _, read_time, process_time = prepare_problem(problem_file, config, prefix_file="")
    prep_time = time()
    emit_log(f"prep_time set {prep_time}, creating session")
    session = create_worker_session(
        prepared,
        config,
        prefix_count=1,
        trace_hook=lambda message: emit_log(f"create_worker_session {message}"),
    )
    worker_time = time()

    n_units = len(prepared.sat_parser.unit_prefix)
    var_mapper = prepared.sat_parser.var_mapper

    # Advice policy (Dagster's launch command passes no worker flags, so these are environment variables):
    #   AFSAT_ADVICE=index (default) | agree | random, AFSAT_ADVICE_ELITE=<fraction of the batch, default 0.5>,
    #   AFSAT_ADVICE_STATS=1 logs the agreement statistics under either policy.
    advice_policy = os.environ.get("AFSAT_ADVICE", "index")
    if advice_policy not in ADVICE_POLICIES:
        raise ValueError(f"AFSAT_ADVICE={advice_policy}: expected one of {ADVICE_POLICIES}")
    elite_frac = float(os.environ.get("AFSAT_ADVICE_ELITE", "0.5"))
    collect_points = advice_policy == "agree" or os.environ.get("AFSAT_ADVICE_STATS") == "1"
    rng = np.random.default_rng(config.runtime_common.rand_seed if config.runtime_common.rand_seed >= 0 else None)

    emit_log(
        f"Worker ready | world_rank={world_rank} helper_rank={helper_rank}/{helper_count} "
        f"vars={prepared.n_var} clauses={prepared.n_clause} batch={session.batch} "
        f"units={n_units} read={read_time:.3f}s process={process_time:.3f}s "
        f"warmup={int(session.warmup_done)}"
    )
    emit_log(
        f"Elapsed | prep={prep_time - start_time:.3f} "
        f"worker_step={worker_time - prep_time:.3f} "
        f"total={worker_time - start_time:.3f}"
    )
    emit_log(
        f"worker_ready world_rank={world_rank} helper_rank={helper_rank}/{helper_count} "
        f"vars={prepared.n_var} clauses={prepared.n_clause} batch={session.batch} phase={phase}"
    )
    emit_line("READY")

    if os.environ.get("AFSAT_WORKER_MODE", "oneshot") == "stream":
        run_stream(session, prepared, var_mapper, int(suggestion_size), elite_frac,
                   int(os.environ.get("AFSAT_ADVICE_POOL", "1024")))
        return

    while True:
        # emit_log("reading next line...")
        raw = sys.stdin.readline()
        # emit_log(f"read {raw[:100]} [...]")
        if raw == "":
            emit_log("Bridge stdin closed; exiting")
            return

        try:
            # emit_log("Attempt prefix parse")
            prefix_lits, stop = parse_prefix_line(raw)
            # emit_log("Received Prefix from Dagster")
            if stop:
                emit_log("Stop command received; shutting down worker loop")
                return

            if prefix_lits.size == 0:
                continue

            # emit_log("Prefix valid, running optimser")
            # Go through the loader rather than mapping the literals here. It merges the
            # problem-implied unit literals into the prefix (the loader strips unit clauses
            # from the clause arrays, so nothing else enforces them) and raises UnsatError
            # when the cube contradicts one of them or itself.
            prefix_vector = prepared.sat_parser.process_prefix_line([str(int(v)) for v in prefix_lits.tolist()])
            prefix_vectors = jnp.asarray(prefix_vector)[None, :]
            result = run_worker_single_batch(session, prefix_vectors, collect_points=collect_points)

            assign_vec_full = np.asarray(result.best_assignment_signed, dtype=np.int8)
            # Suggestions should not simply repeat fixed literals. That means every fixed
            # position, not just the received cube: the merged unit literals would otherwise
            # consume the whole suggestion budget with values the caller already knows.
            free = np.asarray(prefix_vector) == 0
            max_suggestions = max(int(suggestion_size), 0)
            if collect_points:
                phase, order, stats = agreement_advice(
                    result.points, result.points_unsat, free, elite_frac, assign_vec_full
                )
                emit_log("advice policy=%s batch_best=%d %s" % (
                    advice_policy, result.best_unsat, " ".join("%s=%s" % kv for kv in stats.items())))
            if advice_policy == "agree":
                lits = assignment_vector_to_lits(phase, var_mapper, order=order[:max_suggestions])
            elif advice_policy == "random":
                pick = rng.permutation(np.flatnonzero(free))[:max_suggestions]
                lits = assignment_vector_to_lits(rng.choice(np.array([-1, 1], dtype=np.int8), size=free.size),
                                                 var_mapper, order=pick)
            else:
                assign_vec_suggest = assign_vec_full.copy()
                assign_vec_suggest[~free] = 0
                lits = assignment_vector_to_lits(assign_vec_suggest, var_mapper)
            suggestion_lits = lits[:max_suggestions]
            emit_bridge_line("SUGGEST", suggestion_lits)

            if result.sat:
                solution_lits = assignment_vector_to_lits(assign_vec_full, var_mapper)
                emit_bridge_line("SOLUTION", solution_lits)

        except UnsatError as err:
            logger.warning("Skipping conflicting/unsat prefix: %s", err)
            emit_log(f"unsat_prefix {err}")

        except Exception as err:
            # Don't tear the worker down on a single bad request. The C bridge
            # would relaunch us and we'd likely re-hit the same failure, burning
            # the relaunch budget. Log and keep serving the next prefix, exactly
            # as we do for an UnsatError.
            logger.exception("Worker request failed; skipping prefix")
            emit_log(f"request_failed {err}")


def run_stream(session, prepared, var_mapper, suggestion_size: int, elite_frac: float, capacity: int) -> None:
    """Streaming worker (AFSAT_WORKER_MODE=stream): descend continuously under the newest prefix, pool the descents
    (advice_pool.AdvicePool keeps those consistent with each new prefix), and emit refreshed agreement advice after
    every batch. Input is read without blocking, and only the newest of the prefixes waiting is used.
    AFSAT_ADVICE_MIN_AGREE withholds literals whose elite agreement is below the threshold. With
    AFSAT_BRIDGE_LOG_PREFIX=1 each prefix used is written to the bridge trace file as `PREFIX_USED <n> <lits>`."""
    reader = LineReader(sys.stdin.fileno())
    pool = AdvicePool(prepared.n_var, capacity=capacity, elite_frac=elite_frac)
    log_prefix = os.environ.get("AFSAT_BRIDGE_LOG_PREFIX") == "1"
    min_agree = float(os.environ.get("AFSAT_ADVICE_MIN_AGREE", "0"))
    prefix_vector = prefix_vectors = None
    batches = prefixes = dropped_total = 0
    while True:
        newest = None
        for raw in reader.read_lines(block=prefix_vector is None):
            if not raw.strip():
                continue
            try:
                lits, stop = parse_prefix_line(raw)
            except ValueError as err:
                emit_log(f"request_failed {err}")
                continue
            if stop:
                emit_log(f"Stop command received; shutting down worker loop (batches={batches} prefixes={prefixes})")
                return
            if lits.size:
                newest = lits
        if newest is None and reader.eof:
            emit_log("Bridge stdin closed; exiting")
            return
        if newest is not None:
            try:
                pv = np.asarray(prepared.sat_parser.process_prefix_line([str(int(v)) for v in newest.tolist()]))
            except UnsatError as err:
                emit_log(f"unsat_prefix {err}")
                prefix_vector = None
                continue
            prefix_vector, prefix_vectors = pv, jnp.asarray(pv)[None, :]
            dropped_total += pool.set_prefix(prefix_vector)
            prefixes += 1
            if log_prefix and BRIDGE_TRACE_FH is not None:
                BRIDGE_TRACE_FH.write("PREFIX_USED %d %s\n" % (newest.size, " ".join(map(str, newest.tolist()))))
        if prefix_vector is None:
            continue
        try:
            result = run_worker_single_batch(session, prefix_vectors, collect_points=True)
        except UnsatError as err:
            emit_log(f"unsat_prefix {err}")
            prefix_vector = None
            continue
        batches += 1
        pool.add(result.points, result.points_unsat)
        phase, order, stats = pool.advice(suggestion_size, min_agree)
        emit_bridge_line("SUGGEST", assignment_vector_to_lits(phase, var_mapper, order=order))
        if result.sat:
            emit_bridge_line("SOLUTION", assignment_vector_to_lits(
                np.asarray(result.best_assignment_signed, dtype=np.int8), var_mapper))
        if batches % 50 == 0:
            emit_log("stream batches=%d prefixes=%d dropped=%d %s" % (
                batches, prefixes, dropped_total, " ".join("%s=%s" % kv for kv in stats.items())))


def main() -> None:
    try:
        n_devices = len(jax.devices("gpu"))
    except RuntimeError:
        n_devices = 0
    if n_devices == 0:
        n_devices = max(1, len(jax.devices()))

    ap = ArgParse(description="Persistent AFSAT worker for Dagster C bridge")
    ap.add_argument("problem_file", nargs="?", default=None, help="Problem instance path")
    ap.add_argument("--cnf", type=str, default=None, help="Problem instance path (bridge mode)")
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

    io_opts = make_option_group("IO Aliases", "invocation")
    io_opts("--cache", type=str, field="disk_cache", help="Disk cache for FFT matrices")

    runtime_common_opts = make_option_group("Runtime Common Aliases", "runtime_common")
    runtime_common_opts("-n", "--n_devices", type=int, field="n_devices", help="Number of devices")
    runtime_common_opts("-c", "--counting", action="store_true", field="counting", help="Counting mode")
    runtime_common_opts("-s", "--seed", default=156322, type=int, field="rand_seed", help="Force initialise seed (non-negative int)")
    runtime_common_opts("-u", "--unsat_thresh", type=float, field="unsat_thresh", help="Threshold for early stop")
    runtime_common_opts("-m", "--sample_meth", type=str, field="pt_sampler", choices=SAMPLERS, help="Sampler")

    runtime_afsat_opts = make_option_group("Runtime AFSAT Aliases", "runtime_afsat")
    runtime_afsat_opts("-b", "--batch", type=int, field="batch_per_device", help="Batch size per device")
    runtime_afsat_opts("-f", "--fuzz", type=int, field="fuzz", help="Fuzz attempts")
    runtime_afsat_opts(
        "-w", "--warmup", action="store_true", field="warmup",
        help="Accepted for compatibility: the worker always compiles before reporting READY",
    )
    runtime_afsat_opts("--xor_rref", action="store_true", field="xor_rref", help="Enable XOR RREF projection")
    runtime_afsat_opts(
        "--eo_simplex", action="store_true", field="eo_simplex",
        help="Exactly-one groups as simplices (pgd only; ignored with -u or with XOR RREF projection)",
    )

    optimiser_opts = make_option_group("Optimiser Aliases", "optimiser")
    optimiser_opts("-i", "--iters_desc", type=int, field="max_iters", help="Max iterations")
    optimiser_opts("-q", "--solver_tol", type=float, field="tolerance", help="Solver tolerance")
    optimiser_opts("-o", "--optimiser", "--optimizer", type=str, field="name", help="Optimiser")
    optimiser_opts("--pgd_rule", type=str, choices=["spg", "fista"], field="pgd_rule",
                   help="PGD update rule: spg (default) or fista")

    output_opts = make_option_group("Output Aliases", "output_logging")
    output_opts("-d", "--debug", choices=LOG_LEVELS, field="debug_level", help=f"Set logging level ({LOG_LEVELS})")
    output_opts("--binary_v", action="store_true", field="binary_v", help="Use binary assignment string")
    output_opts("--stats", action="store_true", field="stats", help="Per-descent statistics and anomaly checks")

    ap.add_argument("--phase", type=int, default=0, help="Current Dagster phase")
    ap.add_argument("--helper-rank", type=int, default=0, help="Helper rank within local group")
    ap.add_argument("--helper-count", type=int, default=1, help="Total number of helper ranks in local group")
    ap.add_argument("--world-rank", type=int, default=-1, help="Global Dagster MPI rank")
    ap.add_argument("--suggestion-size", type=int, default=30, help="Maximum number of suggestion literals to emit")
    ap.add_argument("--gpus-per-helper", type=int, default=0, help="Provided by bridge for logging and runtime policies")
    ap.add_argument("--bridge-log-file", type=str, default=None, help="Optional path for bridge trace output")

    parsed = ap.parse_args()
    instantiated = ap.instantiate(parsed)
    config = instantiated.config
    config.runtime_common.n_devices = config.runtime_common.n_devices or n_devices
    # Dagster's launch command passes no worker flags, so these two can also be set from the environment.
    if os.environ.get("AFSAT_WORKER_BATCH"):  # batch per device (as -b)
        config.runtime_afsat.batch_per_device = int(os.environ["AFSAT_WORKER_BATCH"])
    if os.environ.get("AFSAT_PGD_RULE"):  # PGD update rule (as --pgd_rule)
        rule = os.environ["AFSAT_PGD_RULE"].strip().lower()
        if rule not in ("spg", "fista"):
            raise ValueError(f"AFSAT_PGD_RULE={rule!r}: expected spg or fista")
        config.optimiser.pgd_rule = rule

    problem_file = parsed.cnf if parsed.cnf else parsed.problem_file
    if not problem_file:
        raise ValueError("A problem file must be supplied via --cnf or positional argument")

    logging.basicConfig(
        level=getattr(logging, config.output_logging.debug_level.upper()),
        format="c %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    trace_path = init_bridge_trace_file(
        parsed.bridge_log_file,
        world_rank=parsed.world_rank,
        helper_rank=parsed.helper_rank,
        phase=parsed.phase,
    )

    # Mirror AFSAT Python logging into bridge LOG lines so imported module logs are
    # visible in the bridge file even if stderr relay is delayed.
    bridge_handler = BridgeEmitHandler(level=logging.INFO)
    bridge_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logging.getLogger("afsat").addHandler(bridge_handler)
    logging.getLogger(__name__).addHandler(bridge_handler)

    if trace_path:
        emit_log(f"bridge_trace_file {trace_path} {problem_file} {config}")

    run_worker(
        problem_file,
        config,
        phase=parsed.phase,
        helper_rank=parsed.helper_rank,
        helper_count=parsed.helper_count,
        world_rank=parsed.world_rank,
        suggestion_size=parsed.suggestion_size,
    )


if __name__ == "__main__":
    main()
