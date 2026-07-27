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
        return np.empty(0, dtype=np.intc), True

    if command != "PREFIX":
        raise ValueError(f"unknown bridge command: {command}")

    if len(parts) < 2:
        raise ValueError("PREFIX message missing length")

    try:
        length = int(parts[1])
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
        lits = np.asarray([int(x) for x in lits_text[:length]], dtype=np.intc)
    except ValueError as exc:
        raise ValueError("PREFIX contains non-integer literal") from exc

    return lits, False


def prefix_lits_to_numpy_vector(prefix_lits: np.ndarray, vc: int) -> np.ndarray:
    """Map signed literals to AFSAT assignment convention in {-1, 0, +1}.

    AFSAT convention:
      +literal -> -1
      -literal -> +1
    """
    x = np.zeros(vc, dtype=np.int8)
    if prefix_lits.size == 0:
        return x

    idx = np.abs(prefix_lits) - 1
    x[idx] = np.where(prefix_lits > 0, -1, 1).astype(np.int8)
    return x


def assignment_vector_to_lits(assign_vec: np.ndarray) -> np.ndarray:
    """Map AFSAT assignment vector in {-1,0,+1} back to signed literals."""
    nz = np.nonzero(assign_vec)[0]
    if nz.size == 0:
        return np.empty(0, dtype=np.intc)

    signs = np.sign(assign_vec[nz]).astype(np.intc)
    lits = (nz.astype(np.intc) + 1) * (-signs)
    return lits.astype(np.intc)


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

    vc = prepared.n_var

    emit_log(
        f"Worker ready | world_rank={world_rank} helper_rank={helper_rank}/{helper_count} "
        f"vars={prepared.n_var} clauses={prepared.n_clause} batch={session.batch} "
        f"read={read_time:.3f}s process={process_time:.3f}s warmup={int(session.warmup_done)}"
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

    # while True:
    #     raw = sys.stdin.readline()
    #     if raw == "":
    #         emit_log("Bridge stdin closed; exiting")
    #         return

    #     try:
    #         prefix_lits, stop = parse_prefix_line(raw)
    #         emit_log("Received Prefix from Dagster")
    #         if stop:
    #             emit_log("Stop command received; shutting down worker loop")
    #             return

    #         if prefix_lits.size == 0:
    #             continue

    #         prefix_vector = prefix_lits_to_numpy_vector(prefix_lits, vc)
    #         prefix_vectors = jnp.asarray(prefix_vector)[None, :]
    #         result = run_worker_single_batch(session, prefix_vectors)

    #         assign_vec_full = np.asarray(result.best_assignment_signed, dtype=np.int8)
    #         # Suggestions should not simply repeat fixed prefix literals.
    #         assign_vec_suggest = assign_vec_full.copy()
    #         assign_vec_suggest[np.abs(prefix_lits) - 1] = 0

    #         lits = assignment_vector_to_lits(assign_vec_suggest)
    #         max_suggestions = max(int(suggestion_size), 0)
    #         suggestion_lits = lits[:max_suggestions]
    #         emit_bridge_line("SUGGEST", suggestion_lits)

    #         if result.sat:
    #             solution_lits = assignment_vector_to_lits(assign_vec_full)
    #             emit_bridge_line("SOLUTION", solution_lits)

    #     except UnsatError as err:
    #         logger.warning("Skipping conflicting/unsat prefix: %s", err)
    #         emit_log(f"unsat_prefix {err}")

    #     except Exception:
    #         logger.exception("Worker request failed")
    #         raise

    while True:
        raw = sys.stdin.readline()
        if raw == "":
            emit_log("Bridge stdin closed; exiting")
            return

        try:
            prefix_lits, stop = parse_prefix_line(raw)
            emit_log("Received Prefix from Dagster")
            if stop:
                emit_log("Stop command received; shutting down worker loop")
                return

            if prefix_lits.size == 0:
                continue

            prefix_vector = prefix_lits_to_numpy_vector(prefix_lits, vc)
            prefix_vectors = jnp.asarray(prefix_vector)[None, :]
            result = run_worker_single_batch(session, prefix_vectors)

            assign_vec = np.asarray(result.best_assignment_signed, dtype=np.int8)
            # Do not echo the fixed prefix literals back to the controller.
            assign_vec[np.abs(prefix_lits) - 1] = 0

            lits = assignment_vector_to_lits(assign_vec)
            max_suggestions = max(int(suggestion_size), 0)
            suggestion_lits = lits[:max_suggestions]
            emit_bridge_line("SUGGEST", suggestion_lits)

            if result.sat:
                emit_line(" ".join([str(l) for l in lits]))
                emit_bridge_line("SOLUTION", lits)

        except UnsatError as err:
            logger.warning("Skipping conflicting/unsat prefix: %s", err)
            emit_log(f"unsat_prefix {err}")

        except Exception:
            logger.exception("Worker request failed")
            raise


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
    runtime_common_opts("-s", "--seed", type=int, field="rand_seed", help="Force initialise seed (non-negative int)")
    runtime_common_opts("-u", "--unsat_thresh", type=float, field="unsat_thresh", help="Threshold for early stop")
    runtime_common_opts("-m", "--sample_meth", type=str, field="sample_method", choices=SAMPLERS, help="Sampler")

    runtime_afsat_opts = make_option_group("Runtime AFSAT Aliases", "runtime_afsat")
    runtime_afsat_opts("-b", "--batch", type=int, field="batch_per_device", help="Batch size per device")
    runtime_afsat_opts("-f", "--fuzz", type=int, field="fuzz", help="Fuzz attempts")
    runtime_afsat_opts("-w", "--warmup", action="store_true", field="warmup", help="Warmup before requests")
    runtime_afsat_opts("--xor_rref", action="store_true", field="xor_rref", help="Enable XOR RREF projection")

    optimiser_opts = make_option_group("Optimiser Aliases", "optimiser")
    optimiser_opts("-i", "--iters_desc", type=int, field="max_iters", help="Max iterations")
    optimiser_opts("-q", "--solver_tol", type=float, field="tolerance", help="Solver tolerance")
    optimiser_opts("-o", "--optimiser", "--optimizer", type=str, field="name", help="Optimiser")

    output_opts = make_option_group("Output Aliases", "output_logging")
    output_opts("-d", "--debug", choices=LOG_LEVELS, field="debug_level", help=f"Set logging level ({LOG_LEVELS})")
    output_opts("--binary_v", action="store_true", field="binary_v", help="Use binary assignment string")

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
