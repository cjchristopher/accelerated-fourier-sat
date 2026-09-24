# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import ctypes
import functools
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


class LogThrottle:
    """Throttles repeated logger.debug messages. After `limit` calls with the same key,
    further messages are suppressed. Call flush() to emit suppressed counts."""

    def __init__(self, logger: logging.Logger, limit: int = 5):
        self.logger = logger
        self.limit = limit
        self._counts: dict[str, int] = {}

    def debug(self, key: str, msg: str) -> None:
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count <= self.limit:
            self.logger.debug(msg)

    def flush(self) -> None:
        for key, count in self._counts.items():
            suppressed = count - self.limit
            if suppressed > 0:
                self.logger.debug(f"... suppressed {suppressed} further '{key}' messages")
        self._counts.clear()

@dataclass
class InvocationIOConfig:
    """Invocation-level and IO inputs for a solve run."""

    prefix_file: str = ""
    disk_cache: str = ""
    profile_enabled: bool = False


@dataclass
class RuntimeCommonConfig:
    """Runtime options shared by AFSAT and novelty routines."""

    timeout_sec: int = 300
    n_devices: int = 1
    counting: bool = False
    benchmark: bool = True
    progress_enabled: bool = False
    rand_seed: int = -1
    ffsat_seed: bool = False
    unsat_thresh: float = 0.0
    pt_sampler: str = "bias"
    restart_f: int = 1
    weight_decay: float = 0.9


@dataclass
class RuntimeAFSATConfig:
    """AFSAT-only runtime options."""

    batch_per_device: int = -1
    fuzz: int = 0
    warmup: bool = True
    xor_rref: bool = True
    propagate: bool = True  # full-formula unit propagation before the JAX stage
    drop_xor_eval: bool = True  # skip the XOR objective when the RREF projector satisfies it by construction


@dataclass
class RuntimeNoveltyConfig:
    """Novelty-only runtime options."""

    beam_per_device: int = -1
    top_m: int = 1
    beta: float = 0.0


@dataclass
class OptimiserConfig:
    """Optimiser and stopping controls."""

    name: str = "pgd"
    max_iters: int = 100
    tolerance: float = 1e-3
    projection_type: str = "box"
    projection_bounds: tuple[float, float] = (-1.0, 1.0)


@dataclass
class OutputLoggingConfig:
    """Output and logging controls."""

    debug_level: str = "ERROR"
    stdout_log: bool = False
    log_propagate: bool = True
    binary_v: bool = False
    anomaly_quit: bool = False


@dataclass
class AFSATConfig:
    """Top-level config for AFSAT runs."""

    invocation: InvocationIOConfig = field(default_factory=InvocationIOConfig)
    runtime_common: RuntimeCommonConfig = field(default_factory=RuntimeCommonConfig)
    runtime_afsat: RuntimeAFSATConfig = field(default_factory=RuntimeAFSATConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    output_logging: OutputLoggingConfig = field(default_factory=OutputLoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AFSATConfig:
        """Load nested AFSAT config from a dictionary."""
        return cls(
            invocation=InvocationIOConfig(**data.get("invocation", {})),
            runtime_common=RuntimeCommonConfig(**data.get("runtime_common", {})),
            runtime_afsat=RuntimeAFSATConfig(**data.get("runtime_afsat", {})),
            optimiser=OptimiserConfig(**data.get("optimiser", {})),
            output_logging=OutputLoggingConfig(**data.get("output_logging", {})),
        )

    def to_file(self, filepath: str) -> None:
        """Save nested AFSAT config to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class NoveltyConfig:
    """Top-level config for novelty runs."""

    invocation: InvocationIOConfig = field(default_factory=InvocationIOConfig)
    runtime_common: RuntimeCommonConfig = field(default_factory=RuntimeCommonConfig)
    runtime_novelty: RuntimeNoveltyConfig = field(default_factory=RuntimeNoveltyConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    output_logging: OutputLoggingConfig = field(default_factory=OutputLoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoveltyConfig:
        """Load nested novelty config from a dictionary."""
        return cls(
            invocation=InvocationIOConfig(**data.get("invocation", {})),
            runtime_common=RuntimeCommonConfig(**data.get("runtime_common", {})),
            runtime_novelty=RuntimeNoveltyConfig(**data.get("runtime_novelty", {})),
            optimiser=OptimiserConfig(**data.get("optimiser", {})),
            output_logging=OutputLoggingConfig(**data.get("output_logging", {})),
        )

    def to_file(self, filepath: str) -> None:
        """Save nested novelty config to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)


# CUdevice_attribute value from cuda.h; stable across CUDA versions.
_CU_ATTR_L2_CACHE_SIZE = 38


@functools.cache
def _query_cuda_l2_bytes(ordinal: int) -> int | None:
    """L2 cache size of CUDA device `ordinal`, read from the driver; None if unavailable.

    Uses the driver API (libcuda ships with every NVIDIA driver), which needs only cuInit: no
    context is created and no device memory is touched, so it is safe alongside JAX. The
    ordinal is relative to CUDA_VISIBLE_DEVICES, as JAX's `local_hardware_id` is.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    device, value = ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
        return None
    if cuda.cuDeviceGetAttribute(ctypes.byref(value), _CU_ATTR_L2_CACHE_SIZE, device) != 0:
        return None
    return value.value or None


def get_gpu_l2_cache_size(device) -> int | None:
    """
    L2 cache capacity of a GPU in bytes: queried from the CUDA driver, else from the table below.

    L2 alone, not L1 + L2. L2 is the one chip-wide cache: every global-memory access passes
    through it, and the per-SM L1s are filled from it, so they mostly hold copies of L2 lines
    rather than adding capacity; data read by many SMs is duplicated across them; and L1 is not
    coherent across kernel launches, so the hand-off between XLA's fused kernels -- the reuse a
    "working set fits in cache" batch size relies on -- can only hit in L2.
    """
    ordinal = getattr(device, "local_hardware_id", None)
    if device.platform == "gpu" and ordinal is not None:
        queried = _query_cuda_l2_bytes(ordinal)
        if queried is not None:
            logger.info(f"Queried {device.device_kind} L2 from the CUDA driver: {queried / 2**20:.1f} MB")
            return queried

    # Fallback by GPU name, for when the driver query fails. L2 sizes in bytes.
    MB = 1024 * 1024
    L2_TABLE = {
        "V100": 6 * MB,
        # Ampere
        "A100": 40 * MB,
        "RTX A2000": 4 * MB,
        "RTX A4000": 4 * MB,
        "RTX A5000": 6 * MB,
        "RTX A6000": 6 * MB,
        "A4000": 4 * MB,
        "A5000": 6 * MB,
        "A6000": 6 * MB,
        "RTX 3090": 6 * MB,
        "RTX 3080": 5 * MB,
        "RTX 3070": 4 * MB,
        # Hopper
        "H100": 50 * MB,
        "H200": 50 * MB,
        # Ada Lovelace
        "RTX 4090": 72 * MB,
        "RTX 4080": 64 * MB,
        "RTX 4070": 36 * MB,
        "L40": 96 * MB,  # AD102
    }
    gpu_name = device.device_kind
    for key, size in L2_TABLE.items():
        if key in gpu_name:
            return size
    return 32 * MB  # Conservative default

