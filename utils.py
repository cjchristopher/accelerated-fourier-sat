# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

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
    rand_seed: bool = False
    unsat_thresh: float = 0.0
    sample_method: str = "bias"
    restart_interval: int = 1
    weight_decay: float = 0.9


@dataclass
class RuntimeAFSATConfig:
    """AFSAT-only runtime options."""

    batch_per_device: int = -1
    fuzz: int = 0
    warmup: bool = False
    xor_rref: bool = False


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
    def from_dict(cls, data: dict[str, Any]) -> "AFSATConfig":
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
    def from_dict(cls, data: dict[str, Any]) -> "NoveltyConfig":
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


def get_gpu_l2_cache_size(device) -> int | None:
    """
    Query total on-chip cache capacity of a GPU in bytes.
    Returns None if unable to determine.

    Returns a model-based estimate of L1 + L2 (+ L3 if present) cache capacity.
    Function name is kept for backward compatibility with existing call sites.
    """
    # Lookup table by GPU name.
    # Values target total cache budget (L1 + L2 + L3 where present), in bytes.
    # For NVIDIA parts listed here, totals are L1+L2 (no dedicated L3 on these models).
    CACHE_TABLE = {
        "V100": int(16 * 1024 * 1024),  # ~6MB L2 + ~10MB aggregate L1
        # Ampere
        "A100": int(60.25 * 1024 * 1024),  # 40MB L2 + 20.25MB aggregate L1
        "A6000": int(16.5 * 1024 * 1024),  # 6MB L2 + 10.5MB aggregate L1
        "A5000": int(14 * 1024 * 1024),  # 6MB L2 + 8MB aggregate L1
        "A4000": int(10 * 1024 * 1024),  # 4MB L2 + 6MB aggregate L1
        "RTX 3090": int(16.25 * 1024 * 1024),  # 6MB L2 + 10.25MB aggregate L1
        "RTX 3080": int(13.5 * 1024 * 1024),  # 5MB L2 + 8.5MB aggregate L1
        "RTX 3070": int(9.75 * 1024 * 1024),  # 4MB L2 + 5.75MB aggregate L1
        # Hopper
        "H100": int(80 * 1024 * 1024),  # 50MB L2 + ~30MB aggregate L1
        "H200": int(83 * 1024 * 1024),  # 50MB L2 + ~33MB aggregate L1
        # Ada Lovelace
        "RTX 4090": int(88 * 1024 * 1024),  # 72MB L2 + 16MB aggregate L1
        "RTX 4080": int(73.5 * 1024 * 1024),  # 64MB L2 + 9.5MB aggregate L1
        "RTX 4070": int(41.75 * 1024 * 1024),  # 36MB L2 + 5.75MB aggregate L1
        "RTX A2000": int(7.25 * 1024 * 1024),  # 4MB L2 + 3.25MB aggregate L1
        "RTX A4000": int(10 * 1024 * 1024),  # 4MB L2 + 6MB aggregate L1
        "RTX A5000": int(14 * 1024 * 1024),  # 6MB L2 + 8MB aggregate L1
        "RTX A6000": int(16.5 * 1024 * 1024),  # 6MB L2 + 10.5MB aggregate L1
        "L40": int(65.75 * 1024 * 1024),  # 48MB L2 + ~17.75MB aggregate L1
        # Blackwell
        "B100": int(128 * 1024 * 1024),  # Estimated total cache
        "B200": int(160 * 1024 * 1024),  # Estimated total cache
    }
    gpu_name = device.device_kind
    for key, size in CACHE_TABLE.items():
        if key in gpu_name:
            return size
    return 32 * 1024 * 1024  # Conservative default

