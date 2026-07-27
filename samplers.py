# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

SAMPLERS: tuple[str, ...] = ("bias", "coin", "uniform", "trunc")


def _check_method(method: str) -> None:
    if method not in SAMPLERS:
        supported = ",".join(SAMPLERS)
        raise ValueError(f"Unsupported method: {method}. Supported methods: {supported}")


def sample_assignments(
    rng_key: Array,
    batch: int,
    n_vars: int,
    method: str = "bias",
    prefixes: Array | None = None,
 ) -> tuple[Array, Array]:
    """Sample assignments in [-1, 1], optionally applying prefixes.

    When prefixes are provided (shape N x n_vars), each prefix is replicated
    batch//N times and fixed positions are applied to the sampled points.

    Returns a tuple of (samples, fixed_mask), where fixed_mask marks coordinates
    constrained by prefixes. If no prefixes are provided, fixed_mask is all False.
    """
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
    _check_method(method)

    if method == "bias":
        # Bias toward false (+1) while preserving full [-1, 1] support.
        u = jax.random.uniform(rng_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        bias_strength = 0.5
        x0 = 2 * u**bias_strength - 1
    elif method == "coin":
        false_prob = 0.7
        coin_key, mag_key = jax.random.split(rng_key)
        biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(batch, n_vars))
        signs = 2 * biased_coins - 1
        magnitudes = jax.random.uniform(mag_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        x0 = signs * magnitudes
    elif method == "uniform":
        x0 = jax.random.uniform(rng_key, minval=-1, maxval=1, shape=(batch, n_vars))
    elif method == "trunc":
        x0 = jax.random.truncated_normal(rng_key, lower=-1, upper=1, shape=(batch, n_vars))
    else:
        # Unreachable: guarded by _check_method.
        raise ValueError(f"Unsupported method: {method}")

    fixed_mask = jnp.full((batch, 1), fill_value=False, dtype=bool)
    if prefixes is not None:
        n_prefix = prefixes.shape[0]
        if n_prefix <= 0 or batch % n_prefix:
            raise ValueError(f"Batch size {batch} must be divisible by number of prefixes {n_prefix}")
        replicated_prefixes = jnp.repeat(prefixes, batch // n_prefix, axis=0)
        fixed_mask = replicated_prefixes != 0
        x0 = jnp.where(fixed_mask, replicated_prefixes, x0)

    # # Fix positions of supplied prefixes.
    # fixed_mask = jnp.full((batch, 1), fill_value=False, dtype=bool)
    # if prefixes is not None:
    #     N = prefixes.shape[0]
    #     # Batch is already correctly sized equal points for each prefix.
    #     replicated_prefixes = jnp.repeat(prefixes, batch // N, axis=0)

    #     # Non-zero points are fixed, so adjust batch and disable gradients there.
    #     fixed_mask = replicated_prefixes != 0
    #     x0 = jnp.where(fixed_mask, replicated_prefixes, x0)

    return x0, fixed_mask
