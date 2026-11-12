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

The PGD subclass also replaces jaxopt's FISTA line search (carried_fista_line_search). jaxopt evaluates the
sufficient-decrease test, and with it the objective, in the loop condition. Under vmap, JAX evaluates a batched
loop predicate twice per trip: in the condition, reduced over the batch, and in the body, to select the points that
advance. XLA compiles the two copies of the objective independently, and nothing makes them round alike; on JAX
0.11 its fusion autotuner can emit one with Triton and the other natively, with different summation orders. Where
the projected step is zero, as at converged vertices, the test compares f(x) with itself against an absolute eps
of 2.2e-16, so one ulp decides it. When the condition says "continue" and the body says "done", that point never
advances and the loop never exits (PLE at tol 1e-12 hung this way, on JAX 0.11.2 and 0.12.0.1). The replacement
evaluates the test once per trial in the body and carries it, so the condition only reads a stored flag. Trial
points and accepted steps are unchanged, and a rolled search now evaluates the objective once per trial instead of
twice.

PGD has two update rules (`method`, from the optimiser config's `pgd_rule`):

  spg    spectral projected gradient (_LoopAwareSPG below), the default. Every iterate is feasible, it has
         stationarity guarantees for smooth non-convex objectives on convex sets, and on SHA-1 r20 preimages
         (16 instances, 5 random-seed repetitions of 300 s) it reached a lower best unsatisfied count than FISTA on
         every instance; on PLE and cardinality benchmarks it matched FISTA within noise.
  fista  FISTA with jaxopt's backtracking, the previous default, kept to reproduce earlier runs and for problems
         where its extrapolated points outside the box help (on PLE they solve more descents than FISTA with the
         extrapolated point clipped to the box).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB, ProjectedGradient, ProximalGradient
from jaxopt._src import base, loop, prox
from jaxopt._src.linesearch_util import _setup_linesearch
from jaxopt._src.proximal_gradient import ProxGradState
from jaxopt._src.tree_util import (
    tree_add_scalar_mul, tree_l2_norm, tree_single_dtype, tree_sub, tree_vdot, tree_zeros_like,
)

PGD_LS_UNROLL_MIN_BATCH = 1024
PGD_RULES = ("spg", "fista")

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


def carried_fista_line_search(fun, prox_grad, jit, unroll, maxls, x, x_fun_val, x_fun_grad, stepsize,
                              decrease_factor, hyperparams_prox, args, kwargs):
    """jaxopt's fista_line_search (same signature, trial points and accepted step) with the sufficient-decrease
    test evaluated once per trial in the loop body and carried in the loop state; the loop condition only reads it
    (module docstring)."""
    eps = jnp.finfo(x_fun_val.dtype).eps

    def insufficient(next_x, step):
        diff_x = tree_sub(next_x, x)
        fun_decrease = step * (fun(next_x, *args, **kwargs) - x_fun_val)
        condition = step * tree_vdot(diff_x, x_fun_grad) + 0.5 * tree_l2_norm(diff_x, squared=True)
        return fun_decrease > condition + eps

    def body(state):
        next_step = state[1] * decrease_factor
        next_x = prox_grad(x, x_fun_grad, next_step, hyperparams_prox)
        return next_x, next_step, insufficient(next_x, next_step)

    init_x = prox_grad(x, x_fun_grad, stepsize, hyperparams_prox)
    next_x, used, _ = loop.while_loop(
        cond_fun=lambda state: state[2], body_fun=body, init_val=(init_x, stepsize, insufficient(init_x, stepsize)),
        maxiter=maxls, unroll=unroll, jit=jit,
    )
    return next_x, used


@dataclass(eq=False)
class _LoopAwareProximalGradient(ProximalGradient):
    """ProximalGradient whose FISTA line search carries its test and has its own unroll flag, and which stops on
    the gradient mapping (module docstring) unless `legacy_stop` is set."""

    ls_unroll: bool = False
    legacy_stop: bool = False

    def __post_init__(self):
        super().__post_init__()
        fun_with_aux, _, _ = base._make_funs_with_aux(
            fun=self.fun, value_and_grad=self.value_and_grad, has_aux=self.has_aux
        )
        self._fun_without_aux = lambda *a, **kw: fun_with_aux(*a, **kw)[0]
        ls = partial(carried_fista_line_search, self._fun_without_aux, self._prox_grad, self.jit, self.ls_unroll)
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


class SPGState(NamedTuple):
    """Spectral projected gradient state; `iter_num` and `error` drive jaxopt's run loop as for ProxGradState."""
    iter_num: Any
    stepsize: Any  # the spectral step used by the last iteration
    error: Any
    aux: Any
    x_prev: Any  # the iterate and gradient the last iteration started from (for the Barzilai-Borwein step)
    g_prev: Any
    f_hist: Any  # objective at the last `memory` iterates, newest first (-inf where unfilled)


def spg_line_search(fun, jit, unroll, maxls, x, d, f_ref, slope_gd, f_tol, decrease_factor, args, kwargs):
    """Backtracking along the segment x + lam * d (lam = 1, decrease_factor, ...) until
    f(x + lam d) <= f_ref + lam * slope_gd + f_tol, with the test carried in the loop state as in
    carried_fista_line_search. Returns (lam, still_insufficient); the second is True only when all maxls
    reductions were rejected, in which case the last trial is returned unchecked."""

    def insufficient(lam):
        return fun(tree_add_scalar_mul(x, lam, d), *args, **kwargs) > f_ref + lam * slope_gd + f_tol

    def body(state):
        lam = state[0] * decrease_factor
        return lam, insufficient(lam)

    one = jnp.ones_like(f_ref)
    lam, bad = loop.while_loop(cond_fun=lambda state: state[1], body_fun=body, init_val=(one, insufficient(one)),
                               maxiter=maxls, unroll=unroll, jit=jit)
    return lam, bad


@dataclass(eq=False)
class _LoopAwareSPG(_LoopAwareProximalGradient):
    """Spectral projected gradient ("spg"), algorithm SPG2 of Birgin, Martinez and Raydan, "Nonmonotone spectral
    projected gradient methods on convex sets", SIAM J. Optim. 10 (2000).

    Each iteration takes the projected direction d = P(x - a g) - x for the Barzilai-Borwein step
    a = (s.s)/(s.y), s and y being the last change in iterate and gradient, clipped to [step_min, step_max]
    (step_max when s.y <= 0, i.e. no positive curvature along the last step; the first iteration uses
    1 / ||P(x - g) - x||_inf). It then backtracks along the segment x + lam d, which stays feasible, until the
    non-monotone Armijo test f(x + lam d) <= max(last `memory` values of f) + slope lam g.d holds. The
    objective may rise between iterations but never above the recent maximum; this converges to stationary
    points for smooth non-convex objectives on closed convex sets, which is what the boxes and the XOR projection
    give. A rounding allowance of f_tol_ulps ulp of f is added to the test (the trial's f is evaluated by a
    different kernel from the one that computed f(x)). There is no momentum, so every trial lies in the feasible
    set.

    Stopping, as for FISTA here: the gradient mapping at unit step, ||P(x - g) - x||, which is zero exactly at
    stationary points and, unlike a mapping at the spectral step, cannot be made small by a long step; an
    exhausted search reports an infinite error (the last trial, at lam = decrease_factor**maxls, is kept).
    Each iteration costs one value-and-gradient and one objective evaluation per trial, as FISTA's does.
    """

    memory: int = 10
    slope: float = 1e-4
    step_min: float = 1e-10
    step_max: float = 1e10
    f_tol_ulps: float = 8.0

    def __post_init__(self):
        super().__post_init__()
        ls = partial(spg_line_search, self._fun_without_aux, self.jit, self.ls_unroll)
        self._spg_line_search = jax.jit(ls, static_argnums=(0,)) if self.jit else ls

    def init_state(self, init_params, hyperparams_prox, *args, **kwargs) -> SPGState:
        del hyperparams_prox
        aux = self.fun(init_params, *args, **kwargs)[1] if self.has_aux else None
        dtype = tree_single_dtype(init_params)
        return SPGState(
            iter_num=jnp.asarray(0), stepsize=jnp.asarray(1.0, dtype=dtype), error=jnp.asarray(jnp.inf, dtype=dtype),
            aux=aux, x_prev=init_params, g_prev=tree_zeros_like(init_params),
            f_hist=jnp.full((self.memory,), -jnp.inf, dtype=dtype),
        )

    def update(self, params, state, hyperparams_prox, *args, **kwargs) -> base.OptStep:
        x = params
        dtype = tree_single_dtype(x)
        (fx, aux), g = self._value_and_grad_with_aux(x, *args, **kwargs)
        unit = tree_sub(self._prox_grad(x, g, 1.0, hyperparams_prox), x)  # projected gradient at unit step
        s, y = tree_sub(x, state.x_prev), tree_sub(g, state.g_prev)
        sy, ss = tree_vdot(s, y), tree_vdot(s, s)
        spectral = jnp.where(sy > 0, jnp.clip(ss / jnp.where(sy > 0, sy, 1.0), self.step_min, self.step_max),
                             self.step_max)
        unit_inf = jnp.max(jnp.abs(unit))
        first = jnp.clip(1.0 / jnp.where(unit_inf > 0, unit_inf, 1.0), self.step_min, self.step_max)
        step = jnp.where(state.iter_num == 0, first, spectral)
        d = tree_sub(self._prox_grad(x, g, step, hyperparams_prox), x)
        f_hist = jnp.roll(state.f_hist, 1).at[0].set(fx)
        f_tol = self.f_tol_ulps * jnp.finfo(dtype).eps * jnp.maximum(1.0, jnp.abs(fx))
        lam, exhausted = self._spg_line_search(
            self.maxls, x, d, jnp.max(f_hist), self.slope * tree_vdot(g, d), f_tol, self.decrease_factor, args, kwargs
        )
        next_x = tree_add_scalar_mul(x, lam, d)
        error = jnp.where(exhausted, jnp.inf, tree_l2_norm(unit))
        next_state = SPGState(
            iter_num=state.iter_num + 1, stepsize=jnp.asarray(step, dtype=dtype),
            error=jnp.asarray(error, dtype=dtype), aux=aux, x_prev=x, g_prev=g, f_hist=f_hist,
        )
        return base.OptStep(params=next_x, state=next_state)


@dataclass(eq=False)
class LoopAwareProjectedGradient(ProjectedGradient):
    """ProjectedGradient with independent outer (`unroll`) and line-search (`ls_unroll`) loop strategies, and the
    update rule `method` (one of PGD_RULES; module docstring)."""

    ls_unroll: bool = False
    method: str = "spg"

    def __post_init__(self):
        if getattr(self, "check_projection", False):
            raise ValueError("check_projection wraps the inner solver's update and is not supported here")
        if self.method not in PGD_RULES:
            raise ValueError(f"PGD rule {self.method!r}: expected one of {', '.join(PGD_RULES)}")
        super().__post_init__()
        rule = _LoopAwareSPG if self.method == "spg" else _LoopAwareProximalGradient
        self._pg = rule(
            fun=self.fun,
            prox=prox.make_prox_from_projection(self.projection),
            value_and_grad=self.value_and_grad,
            has_aux=self.has_aux,
            stepsize=self.stepsize,
            maxiter=self.maxiter,
            maxls=self.maxls,
            tol=self.tol,
            acceleration=self.acceleration and self.method == "fista",
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
