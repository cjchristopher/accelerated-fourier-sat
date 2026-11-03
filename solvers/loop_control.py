# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""Separate unroll flags for the jaxopt solvers AFSAT batches with vmap.

jaxopt applies one `unroll` flag to both the iteration loop and the line search nested in it. Under vmap a rolled
loop (lax.while_loop) runs until the batch's slowest point is done and pays a batch-wide condition check on every
trip; an unrolled loop (a fixed-length lax.scan) skips the checks but always runs to its limit. The subclasses
below give the line search its own flag (`ls_unroll`); their iterates are the same as stock jaxopt's
(scripts/check_loop_control.py).

The strategy is static, from V100 measurements (16-2048 points per device, SHA-1 r20-r50, random 3-SAT, sudoku,
Ramsey, cardinality; throughput relative to everything rolled):

  outer loop       always rolled. Unrolling it never paid off: PGD 0.91x (geometric mean), down to 0.37x;
                   L-BFGS-B 1.01x.
  L-BFGS-B         line search rolled: its zoom search accepts the first trial on ~99% of steps.
  PGD line search  unrolled from PGD_LS_UNROLL_MIN_BATCH points per device. A rolled search runs until the batch's
                   slowest point accepts; with ~1000+ points nearly every step has a point that needs most of the
                   trials, so the checks are pure overhead: 1.35-2.7x faster there (worst 0.96x, Ramsey). Below
                   that the early exit wins (down to 0.50x unrolled, sudoku at 256).

A per-batch adaptive switch between these kernels was tried and dropped: it averaged 1.03x and chose the wrong
kernel at large batches.

The PGD subclass also replaces jaxopt's stopping measure (AFSAT_STOP below). jaxopt's accelerated update stops on
||x_{k+1} - x_k|| / s_next, where s_next is the step for the *next* iteration: twice the step used after a
successful line search (decrease_factor 0.5), and 1.0 once the used step has fallen to 1e-6 or below. Two faults
follow. The momentum part of x_{k+1} - x_k does not vanish at a stationary point, so a descent drifting along an
exactly flat direction never stops; and a collapsed line search divides a tiny move by 1.0 and reports
convergence. The replacement is the gradient mapping at the point the step was taken from,
||x_{k+1} - y_k|| / s_used = ||y_k - P(y_k - s_used grad f(y_k))|| / s_used, which is zero exactly at stationary
points of f over the box (Beck, First-Order Methods in Optimization, SIAM 2017, ch. 10); jaxopt's non-accelerated
update and its optimality_fun already measure this quantity. A line search that used all its trials without
meeting sufficient decrease reports an infinite error, so it never counts as convergence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB, ProjectedGradient, ProximalGradient
from jaxopt._src import base, prox
from jaxopt._src.linesearch_util import _setup_linesearch
from jaxopt._src.proximal_gradient import ProxGradState, fista_line_search
from jaxopt._src.tree_util import tree_add_scalar_mul, tree_l2_norm, tree_single_dtype, tree_sub

PGD_LS_UNROLL_MIN_BATCH = 1024

# AFSAT_STOP selects the PGD stopping measure: "gradmap" (default) is the gradient mapping described above,
# "legacy" is jaxopt's ||x_{k+1} - x_k|| / s_next, kept to reproduce earlier runs.
STOP_ENV = "AFSAT_STOP"


def stop_measure() -> str:
    raw = os.environ.get(STOP_ENV, "gradmap").strip().lower()
    if raw not in ("gradmap", "legacy"):
        raise ValueError(f"{STOP_ENV}={raw!r}: expected 'gradmap' or 'legacy'")
    return raw

# AFSAT_LOOPS overrides the rule for benchmarking: two letters, outer loop then line search, R = rolled and
# U = unrolled (so "UU" is jaxopt's unroll=True and "RR" its unroll=False). "auto" (default) is the rule above.
LOOPS_ENV = "AFSAT_LOOPS"


def forced_loops() -> Optional[tuple[bool, bool]]:
    raw = os.environ.get(LOOPS_ENV, "auto").strip()
    if raw.lower() == "auto":
        return None
    if len(raw) != 2 or set(raw.upper()) - {"R", "U"}:
        raise ValueError(f"{LOOPS_ENV}={raw!r}: expected 'auto' or two letters from R/U, e.g. RR, RU, UR, UU")
    return raw[0].upper() == "U", raw[1].upper() == "U"


def choose_loops(algorithm: str, points_per_device: int) -> tuple[bool, bool]:
    """(outer loop unrolled, line search unrolled) for a batch of this size."""
    forced = forced_loops()
    if forced is not None:
        return forced
    return False, algorithm == "pgd" and points_per_device >= PGD_LS_UNROLL_MIN_BATCH


@dataclass(eq=False)
class _LoopAwareProximalGradient(ProximalGradient):
    """ProximalGradient whose FISTA line search has its own unroll flag, and which stops on the gradient mapping
    (module docstring) unless `legacy_stop` is set."""

    ls_unroll: bool = False
    legacy_stop: bool = False

    def __post_init__(self):
        super().__post_init__()
        fun_with_aux, _, _ = base._make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )
        fun_without_aux = lambda *a, **kw: fun_with_aux(*a, **kw)[0]
        ls = partial(fista_line_search, fun_without_aux, self._prox_grad, self.jit, self.ls_unroll)
        self._fista_line_search = jax.jit(ls, static_argnums=(0,)) if self.jit else ls

    def _update_accel(self, x, state, hyperparams_prox, args, kwargs):
        line_search = not callable(self.stepsize) and self.stepsize <= 0
        if self.legacy_stop or not line_search:
            return super()._update_accel(x, state, hyperparams_prox, args, kwargs)
        # jaxopt's _update_accel and _iter, with the step policy unchanged and only the error replaced.
        dtype = tree_single_dtype(x)
        y, t, stepsize = state.velocity, state.t, state.stepsize
        (y_fun_val, aux), y_fun_grad = self._value_and_grad_with_aux(y, *args, **kwargs)
        next_x, used = self._fista_line_search(
            self.maxls, y, y_fun_val, y_fun_grad, stepsize, self.decrease_factor, hyperparams_prox, args, kwargs
        )
        next_stepsize = jnp.where(used <= 1e-6, 1.0, used / self.decrease_factor)
        next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
        next_y = tree_add_scalar_mul(next_x, (t - 1) / next_t, tree_sub(next_x, x))
        # The search shrinks the step by decrease_factor per rejected trial, so the used step is
        # stepsize * decrease_factor**k; k == maxls means the first maxls trials were rejected and the last was
        # returned unchecked. Half a factor of slack absorbs rounding for any decrease_factor.
        exhausted = used < stepsize * self.decrease_factor ** (self.maxls - 0.5)
        error = jnp.where(exhausted, jnp.inf, tree_l2_norm(tree_sub(next_x, y)) / used)
        next_state = ProxGradState(
            iter_num=state.iter_num + 1, velocity=next_y, t=next_t,
            stepsize=jnp.asarray(next_stepsize, dtype=dtype), error=jnp.asarray(error, dtype=dtype), aux=aux,
        )
        return base.OptStep(params=next_x, state=next_state)


@dataclass(eq=False)
class LoopAwareProjectedGradient(ProjectedGradient):
    """ProjectedGradient with independent outer (`unroll`) and line-search (`ls_unroll`) loop strategies."""

    ls_unroll: bool = False

    def __post_init__(self):
        if getattr(self, "check_projection", False):
            raise ValueError("check_projection wraps the inner solver's update and is not supported here")
        super().__post_init__()
        self._pg = _LoopAwareProximalGradient(
            fun=self.fun,
            prox=prox.make_prox_from_projection(self.projection),
            value_and_grad=self.value_and_grad,
            has_aux=self.has_aux,
            stepsize=self.stepsize,
            maxiter=self.maxiter,
            maxls=self.maxls,
            tol=self.tol,
            acceleration=self.acceleration,
            decrease_factor=self.decrease_factor,
            verbose=self.verbose,
            implicit_diff=self.implicit_diff,
            jit=self.jit,
            unroll=self.unroll,
            ls_unroll=self.ls_unroll,
            legacy_stop=stop_measure() == "legacy",
        )


@dataclass(eq=False)
class LoopAwareLBFGSB(LBFGSB):
    """LBFGSB with independent outer (`unroll`) and line-search (`ls_unroll`) loop strategies. The rule never
    unrolls its line search; `ls_unroll=True` exists to reproduce jaxopt's coupled behaviour in benchmarks."""

    ls_unroll: bool = False

    def __post_init__(self):
        super().__post_init__()
        fun_with_aux, _, value_and_grad_with_aux = base._make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )
        self.run_ls = _setup_linesearch(
            linesearch=self.linesearch,
            fun=fun_with_aux,
            value_and_grad=value_and_grad_with_aux,
            has_aux=True,
            maxlsiter=self.maxls,
            max_stepsize=self.max_stepsize,
            jit=self.jit,
            unroll=self.ls_unroll,
            verbose=max(int(self.verbose) - 1, 0),
        ).run
