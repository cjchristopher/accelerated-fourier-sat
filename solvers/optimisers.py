# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import abc
import functools
import logging
from collections.abc import Callable
from time import perf_counter as time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

# import optimistix as optx
from jax import Array
from jaxopt import (
    LBFGS,
    LBFGSB,
    GradientDescent,
    NonlinearCG,
    ProjectedGradient,
    ProximalGradient,
    ScipyBoundedMinimize,
)
from jaxopt._src import base as job
from jaxopt.projection import projection_box as box
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize import minimize as ScipyMinimize

from boolean_whf import Objective, clause_type_ids
from xor_rref import XorRREFMetadata

logger = logging.getLogger(__name__)

# TODO: Import specific solver modules when needed
# from . import hj_mad
# from . import langevin_annealing
# from . import pgd
print = functools.partial(print, flush=True)

EvalFn: TypeAlias = Callable[[Array, Array, Array], Array]
SeqEvalFn: TypeAlias = Callable[[Array, Array, tuple[Array, ...]], Array | tuple[Array, Array | tuple[Array, Array]]]
VerifyFn: TypeAlias = Callable[[Array], Array]
UnsatRule: TypeAlias = Callable[[Array, Array, Array], Array]

# fmt: off
UNSAT_RULES: dict[str, UnsatRule] = {
    "xor":  lambda x, mask, _    :  jnp.sum(x < 0, axis=1, where=mask) % 2 == 0,
    "cnf":  lambda x, mask, _    :  jnp.min(x, axis=1, where=mask, initial=jnp.finfo(x.dtype).max) > 0,
    "eo":   lambda x, mask, _    :  jnp.sum(x < 0, axis=1, where=mask) != 1,
    "amo":  lambda x, mask, _    :  jnp.sum(x < 0, axis=1, where=mask) > 1,
    "nae":  lambda x, mask, _    :  jnp.logical_not(
                                        jnp.logical_and(
                                            (jnp.min(x, axis=1, where=mask, initial=jnp.finfo(x.dtype).max) < 0),
                                            (jnp.max(x, axis=1, where=mask, initial=jnp.finfo(x.dtype).min) > 0),
                                        )
                                    ),
    "card": lambda x, mask, cards:  jnp.where(
                                        cards < 0,
                                        jnp.sum(x < 0, axis=1, where=mask) >= jnp.abs(cards),
                                        jnp.sum(x < 0, axis=1, where=mask) < cards,
                                    ),
    "ek":   lambda x, mask, cards:  jnp.sum(x < 0, axis=1, where=mask) != cards,
}
# fmt: on


def _bind_unsat_rule(template: UnsatRule, mask: Array, cards: Array) -> Callable[[Array], Array]:
    return lambda x, _t=template, _m=mask, _c=cards: _t(x, _m, _c)


def _make_xor_projector(meta: XorRREFMetadata) -> Callable[[Array], Array]:
    """Overwrite each RREF dependent with the value its row derives from the free variables.

    Row i asserts `x_dep = (1 - 2 b_i) * prod_{j in row} x_j`. That single product carries both
    the sign (an odd number of negative factors is exactly odd GF(2) parity) and the multilinear
    magnitude, and its gradient is the exact product rule -- defined everywhere, including at 0,
    so no log/exp detour or magnitude clip is needed.

    The one thing the bare product gets wrong is its *sign* when a factor is exactly 0: the
    product is then 0, which the verifier reads as FALSE whatever the parity says. The
    `parity_sign * tiny` term fixes that. Whenever the product is non-zero it has the same sign
    as the product (so it only perturbs the magnitude by `tiny`), when the product is 0 it
    supplies the parity sign alone, and it is piecewise constant, so gradients are untouched.
    """
    row_vars = meta.row_vars
    row_mask = meta.row_mask
    row_sign = meta.row_sign
    dep_idx = meta.dependent_indices

    def project(x: Array) -> Array:
        dtype = x.dtype
        sign = row_sign.astype(dtype)
        xs = x[row_vars]  # (n_dep, K)
        product = jnp.prod(jnp.where(row_mask, xs, 1.0), axis=-1)
        negatives = jnp.sum((xs < 0) & row_mask, axis=-1)
        parity_sign = sign * (1.0 - 2.0 * (negatives % 2)).astype(dtype)
        x_dep = sign * product + parity_sign * jnp.finfo(dtype).tiny
        return x.at[dep_idx].set(x_dep)

    return project


def build_eval_verify(objs: tuple[Objective, ...], unbounded: bool) -> tuple[tuple[EvalFn, ...], tuple[VerifyFn, ...]]:
    """
    Constructs JAX-based evaluators and verifiers for a set of objectives.
    This function generates callable evaluators and verifiers for the given objectives by closing over their constants.
    Evaluators compute the cost of assignments and verifiers check the satisfaction of clauses based on the assignment.

    Uses the convolution theorem to compute many ESP evaluations in parallel via the fourier domain.
    O(n^2) before parallelisation - tree based convolution is O(nlog^2n), but is ordered thus can't be parallelised.

    Args:
        objs (tuple[Objective, ...]): The objectives to generate evaluators and verifiers for.

    Returns:
        tuple: A tuple containing two tuples of Callables:
            - evaluators (tuple[Evaluator, ...]): Given assignment and weights, evaluate the cost of an objective
            - verifiers (tuple[Verifier, ...]): Given assignment, check satisfaction of an objective.
    """

    def single_eval_verify(obj: Objective) -> tuple[EvalFn, VerifyFn]:
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types
        is_xor_obj = bool(np.all(np.asarray(types).reshape(-1) == clause_type_ids["xor"]))

        dft, idft = obj.ffts
        forward_mask = True if jnp.all(obj.forward_mask) else obj.forward_mask
        clause_count = lits.shape[0]
        # Most objectives are homogeneous; prefilter to the relevant clause rules once.
        type_ids_present = set(np.asarray(types).reshape(-1).tolist())
        # print("binding rules:", [ctype for ctype in UNSAT_RULES.keys() if clause_type_ids[ctype] in type_ids_present])
        relevant_rules = [
            (clause_type_ids[clause_type], _bind_unsat_rule(template, mask, cards))
            for clause_type, template in UNSAT_RULES.items()
            if clause_type_ids[clause_type] in type_ids_present
        ]

        # fmt: off
        def evaluate_xor(x: Array, fixed_vars: Array, weight: Array) -> Array:
            x = jnp.where(fixed_vars, jax.lax.stop_gradient(x), x)
            assignment = sign * x[lits]                                           # (N,K) * (N,K) = (N,K)
            clause_eval = jnp.prod(assignment, axis=-1)                           # (N,)
            weighted_eval = weight * clause_eval                                  # (N,) * (N,) = (N,)
            x_eval = jnp.sum(weighted_eval, axis=-1)                              # (1,)
            if not unbounded:
                return jnp.atleast_1d(x_eval)
            else:
            # Affine shift to [0,1]-cube and add error term for unbounded optimisation.
                return ((jnp.atleast_1d(x_eval)+1)/2)**2 + (x**2 - 1)**(lits.shape[-1])

        def evaluate(x: Array, fixed_vars: Array, weight: Array) -> Array:
            x = jnp.where(fixed_vars, jax.lax.stop_gradient(x), x)
            assignment = sign * x[lits]                                           # (N,K) * (N,K) = (N,K)
            # Add dim to capture K+1 shifted roots for K terms of the clause.
            fourier_domain = dft + assignment[:, None, :]                         # (N,(K+1),1) + (N,_,K) = (N,(K+1),K)
            esp_freq = jnp.prod(fourier_domain, axis=-1, where=forward_mask)      # (N,(K+1))
            esp_eval = idft * esp_freq                                            # (1,(K+1)) * (N,(K+1)) = (N,(K+1))
            clause_eval = jnp.sum(esp_eval.real, axis=-1)                         # (N,)
            weighted_eval = weight * clause_eval                                  # (N,) * (N,) = (N,)
            x_eval = jnp.sum(weighted_eval, axis=-1)                              # (1,)
            if not unbounded:
                return jnp.atleast_1d(x_eval)
            else:
            # Affine shift to [0,1]-cube and add error term for unbounded optimisation.
                return ((jnp.atleast_1d(x_eval)+1)/2)**2 + (x**2 - 1)**(lits.shape[-1])

        # fmt: on
        def verify(x: Array) -> Array:
            assignment = sign * x[lits]
            unsat = jnp.zeros(clause_count, dtype=bool)

            for type_id, rule in relevant_rules:
                type_mask = types == type_id
                unsat_clauses = rule(assignment)
                unsat = unsat | jnp.where(type_mask, unsat_clauses, False)
            return unsat

        if is_xor_obj:
            eval_f = evaluate_xor
        else:
            eval_f = evaluate
        return eval_f, verify

    eval_fns: tuple[EvalFn]
    verify_fns: tuple[VerifyFn]
    eval_fns, verify_fns = zip(*[single_eval_verify(obj) for obj in objs])
    return eval_fns, verify_fns


def drop_projected_xor_evaluators(
    objs: tuple[Objective, ...], eval_fns: tuple[EvalFn, ...]
) -> tuple[tuple[EvalFn, ...], int]:
    """Replace the evaluator of every all-XOR objective with a zero cost.

    Only valid while the RREF projector owns those clauses, i.e. every XOR clause is satisfied
    by construction -- which fails if a prefix fixes an RREF dependent, so the caller must check
    that first. The verifiers are untouched, and each objective keeps its slot, so `weights`
    stays index-aligned with the reweighting loop.

    Returns the new evaluators and how many objectives were dropped.
    """
    xor_id = clause_type_ids["xor"]

    def zero_cost(x: Array, fixed_vars: Array, weight: Array) -> Array:
        return jnp.zeros((1,), dtype=x.dtype)

    new_fns, dropped = [], 0
    for obj, fn in zip(objs, eval_fns):
        if bool(np.all(np.asarray(obj.clauses.types).reshape(-1) == xor_id)):
            new_fns.append(zero_cost)
            dropped += 1
        else:
            new_fns.append(fn)
    return tuple(new_fns), dropped


def seq_eval_verify(
    eval_fns: tuple[EvalFn, ...],
    verify_fns: tuple[VerifyFn, ...],
    xor_rref_meta: XorRREFMetadata | None = None,
) -> tuple[SeqEvalFn, VerifyFn]:
    """
    Groups a collection (usually all) of Evaluation functions & Verifier functions into a sequence.
    """

    xor_projector: Callable[[Array], Array] | None = (
        _make_xor_projector(xor_rref_meta) if xor_rref_meta is not None else None
    )

    def _apply_fixed_and_projection(x: Array, fixed_vars: Array) -> Array:
        x_eval = jnp.where(fixed_vars, jax.lax.stop_gradient(x), x)
        if xor_projector is None:
            return x_eval

        x_proj = xor_projector(x_eval)
        return jnp.where(fixed_vars, x_eval, x_proj)

    def seq_evals(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array | tuple[Array, Array]]:
        x_eval = _apply_fixed_and_projection(x, fixed_vars)
        costs = [evaluate(x_eval, fixed_vars, weight) for (evaluate, weight) in zip(eval_fns, weights)]
        cost = jnp.sum(jnp.array(costs))
        return cost, (x_eval, cost)  
        # returns costs in aux for breakdown by objective. aux info - remove when consolidating

    def seq_verifies(x: Array) -> Array:
        all_res = [verify(x) for verify in verify_fns]
        res = jnp.concat(all_res, axis=-1)
        return res

    return seq_evals, seq_verifies


# jaxopt offers two jittable loop strategies and they trade off in opposite
# directions under vmap. lax.while_loop stops once every start in the batch is
# done, but pays a dynamic trip count and a predicate reduction each iteration.
# The unrolled path is a fixed-length lax.scan: cheaper per iteration, but its
# early-exit cond degenerates to a select under vmap, so it always runs the full
# maxiter. Measured on GPU the while_loop iteration costs ~1.5x the scan
# iteration, so the scan wins exactly when the while_loop would run past this
# fraction of maxiter.
SCAN_BREAKEVEN_FRAC = 0.65


def _unroll_decision(iters: Array, maxiter: int, batch: int) -> tuple[bool, dict[str, float]]:
    """Pick a loop strategy from the warmup iteration counts.

    Under vmap the while_loop runs until *every* start is done, so its cost is
    set by max(iters), not by the mean -- one straggler costs as much as a fully
    saturated batch. The question is therefore not whether a lone deep start was
    an outlier, but whether a future batch is likely to contain one: if a
    fraction q of starts run deep, a batch of B holds at least one with
    probability 1 - (1 - q)**B, which for q = 1/1000 and B = 1000 is already
    63%. Rare stragglers are a reason to prefer the scan, not to discount.
    """
    flat = np.asarray(iters).reshape(-1).astype(np.float64)
    threshold = SCAN_BREAKEVEN_FRAC * maxiter
    deep = flat >= threshold
    q = float(deep.mean()) if flat.size else 0.0
    # Probability that a future batch of this size contains at least one deep start.
    p_deep = 1.0 - (1.0 - q) ** max(batch, 1)
    shallow = flat[~deep]
    shallow_max = float(shallow.max()) if shallow.size else 0.0
    # Expected while_loop trip count = max over the batch, which is maxiter
    # whenever any start runs deep and the shallow maximum otherwise.
    expected_trips = p_deep * maxiter + (1.0 - p_deep) * shallow_max

    stats = {
        "mean": float(flat.mean()) if flat.size else 0.0,
        "median": float(np.median(flat)) if flat.size else 0.0,
        "stdev": float(flat.std()) if flat.size else 0.0,
        "p99": float(np.percentile(flat, 99)) if flat.size else 0.0,
        "max": float(flat.max()) if flat.size else 0.0,
        "frac_deep": q,
        "p_deep_batch": p_deep,
        "expected_trips": expected_trips,
        "breakeven": threshold,
    }
    return expected_trips > threshold, stats


class Optimiser(abc.ABC):
    def __init__(
        self,
        evaluator: SeqEvalFn,
        verifier: VerifyFn,
        algorithm: str = "lbfgsb",
        maxiter: int = 100,
        tol: float = 1e-3,
        unroll: bool = False,
    ) -> None:
        self.algo = algorithm
        self.maxiter = maxiter
        self.warmup_sol = False
        self.warmup_x: NDArray | None = None
        self.unroll = unroll
        # Kept so warmup can rebuild the solver once it has seen real iteration counts.
        self._build_args = (evaluator, verifier, algorithm, maxiter, tol)

        # TODO: Change to optimistix when mature.
        # if self.sol_name in ["optim"]:
        #     self.solver = BoundedBFGS()
        #     self.solver = optimistix.minimise(evaluator, BoundedBFGS, args={'weights': weights}, has_aux=True)

        if self.algo in ["lbfgsb", "pgd", "josp-lbfgsb", "unbounded", "unbounded2"]:
            # probably change to if sol_name in JAX_OPTIMS
            if self.algo in ["unbounded2"]:
                lbfgs = LBFGS(fun=evaluator, maxiter=self.maxiter, has_aux=True, tol=tol)
                logger.info("Setting up JAXOPT Squared L-BFGS:")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    x_opt, state = lbfgs.run(init_params=x, fixed_vars=fixed_vars, weights=weights)
                    final_cost, eval_aux = evaluator(x_opt, fixed_vars, weights)
                    x_eval = eval_aux[0]
                    final_aux: Any = (x_eval, final_cost)
                    unsat = jnp.squeeze(verifier(x_eval))
                    return x_eval, unsat, jnp.atleast_1d(state.iter_num), final_aux

            if self.algo in ["unbounded"]:
                gd = GradientDescent(fun=evaluator, maxiter=self.maxiter, has_aux=True, tol=tol)
                logger.info("Setting up JAXOPT Squared Gradient Descent:")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    x_opt, state = gd.run(init_params=x, fixed_vars=fixed_vars, weights=weights)
                    final_cost, eval_aux = evaluator(x_opt, fixed_vars, weights)
                    x_eval = eval_aux[0]
                    final_aux: Any = (x_eval, final_cost)
                    unsat = jnp.squeeze(verifier(x_eval))
                    return x_eval, unsat, jnp.atleast_1d(state.iter_num), final_aux

            if self.algo in ["lbfgsb"]:
                lbfgsb = LBFGSB(fun=evaluator, maxiter=self.maxiter, has_aux=True, tol=tol, unroll=self.unroll)
                logger.info("Setting up JAXOPT L-BFGS-B:")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = lbfgsb.run(init_params=x, fixed_vars=fixed_vars, weights=weights, bounds=bounds)
                    final_cost, eval_aux = evaluator(x_opt, fixed_vars, weights)
                    x_eval = eval_aux[0]
                    final_aux: Any = (x_eval, final_cost)
                    unsat = jnp.squeeze(verifier(x_eval))
                    return x_eval, unsat, jnp.atleast_1d(state.iter_num), final_aux

            elif self.algo in ["josp-lbfgsb"]:
                spminB = ScipyBoundedMinimize(
                    fun=evaluator, method="L-BFGS-B", maxiter=self.maxiter, has_aux=True, tol=tol
                )
                logger.info("Setting up JAXOPT ScipyBounded L-BFGS-B")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = spminB.run(x, fixed_vars=fixed_vars, weights=weights, bounds=bounds)
                    final_cost, eval_aux = evaluator(x_opt, fixed_vars, weights)
                    x_eval = eval_aux[0]
                    final_aux: Any = (x_eval, final_cost)
                    unsat = jnp.squeeze(verifier(x_eval))
                    return x_eval, unsat, jnp.atleast_1d(state.iter_num), final_aux

            elif self.algo in ["pgd"]:
                # Leave implicit_diff on: it is free (the custom_vjp only records a
                # differentiation rule, and the optimised HLO is identical either way),
                # and it is what makes the solve differentiable w.r.t. weights. What is
                # NOT free is that jaxopt derives unroll from implicit_diff when unroll is
                # "auto", so turning it off silently switches the solver loop to a
                # full-length lax.scan whose early-exit cond degenerates to a select under
                # vmap -- every start then pays all `maxiter` iterations. unroll is pinned
                # rather than left on "auto" so that coupling cannot bite; warmup retunes
                # it from observed iteration counts (see _unroll_decision).
                pgd = ProjectedGradient(
                    fun=evaluator, projection=box, maxiter=self.maxiter, has_aux=True, tol=tol, unroll=self.unroll
                )
                logger.info("Setting up JAXOPT Projected Gradient (Box)")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    x_opt, state = pgd.run(x, fixed_vars=fixed_vars, weights=weights, hyperparams_proj=(-1, 1))
                    final_cost, eval_aux = evaluator(x_opt, fixed_vars, weights)
                    x_eval = eval_aux[0]
                    final_aux: Any = (x_eval, final_cost)
                    unsat = jnp.squeeze(verifier(x_eval))
                    return x_eval, jnp.atleast_1d(unsat), jnp.atleast_1d(state.iter_num), final_aux

            else:
                pass

            def vectorise(
                xs: Array, fixed_vars: Array, weights: tuple[Array, ...]
            ) -> tuple[Array, Array, Array, Array, Array]:
                x_opt, unsat, iters, evals = jax.vmap(opt, in_axes=(0, 0, None))(xs, fixed_vars, weights)
                unsat_cl_count = jnp.sum(jnp.atleast_2d(unsat), axis=1)
                return x_opt, jnp.atleast_2d(unsat), iters, unsat_cl_count, evals

            self.run = jax.jit(vectorise, donate_argnums=(0))

        elif self.algo in ["prox"]:
            self.solver = ProximalGradient
            # self.solver = ProximalGradient(fun=evaluator, prox=hj_moreau, maxiter=self.maxiter)
            raise NotImplementedError("HJ Moreau Proximal Gradient not yet Implemented")

        elif self.algo in ["nlcg"]:
            self.solver = NonlinearCG
            raise NotImplementedError("Non-Linear Conjugate Gradient not yet Implemented")

        elif self.algo in ["langevin"]:
            raise NotImplementedError("Langevin Annealing not yet Implemented")

        elif self.algo in ["sp-lbfgsb"]:
            # We can only sensibly run this in combined multi-start since sp.minimize is max(CPU-thread) bound.
            # Here we manually VMAP and JIT the opt and ver steps
            self.solver = ScipyMinimize
            logger.info("Setting up ScipyBounded L-BFGS-B (CPU) with JIT'd eval+verify (GPU)")
            _, _, eval_fun = job._make_funs_without_aux(fun=evaluator, value_and_grad=False, has_aux=True)
            v_eval_fun = jax.jit(jax.vmap(eval_fun, in_axes=(0, 0, None)))
            v_verifier = jax.jit(jax.vmap(verifier, in_axes=(0,)))
            self.eval_fun = v_eval_fun

            def full_vectorise(
                x: Array, fixed_vars: Array, weights: tuple[Array, ...]
            ) -> tuple[Array, Array, Array, Array, Array]:
                def flat(np_x0: NDArray) -> tuple[float, NDArray]:
                    # Convert back to JAX arrays, run on GPU, return CPU results to minimize.
                    x0 = jnp.array(np_x0).reshape(x.shape)
                    v, g = v_eval_fun(x0, fixed_vars, weights)  # this is eval.
                    return float(jnp.sum(v)), np.array(g.flatten())

                x0 = np.array(x.flatten())
                bounds = Bounds(-1, 1, True)
                options = {"maxiter": self.maxiter}
                res: OptimizeResult = ScipyMinimize(flat, x0, bounds=bounds, jac=True, tol=tol, options=options)
                x_opt = jnp.array(res.x).reshape(x.shape)
                _, eval_aux = evaluator(x_opt, fixed_vars, weights)
                x_eval = eval_aux[0]
                unsat = jnp.squeeze(v_verifier(x_eval))
                unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
                return x_eval, unsat, jnp.array([res.nit]), unsat_cl_count, jnp.array(res.aux)

            self.run = full_vectorise

        else:
            pass

    def peak_memory_estimation(self, x0: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> int:
        runner = self.run if self.algo not in ["sp-lbfgsb"] else self.eval_fun
        traced = runner.trace(x0, fixed_vars, weights)
        lowered = traced.lower()
        compiled = lowered.compile()
        analysis = compiled.memory_analysis()

        peak_est = (
            analysis.temp_size_in_bytes  # type: ignore
            + analysis.argument_size_in_bytes  # type: ignore
            + analysis.output_size_in_bytes  # type: ignore
            - analysis.alias_size_in_bytes  # type: ignore
        )
        return int(peak_est)

    def _retune_unroll(
        self,
        iters: Array,
        x0_spec: tuple[tuple[int, ...], Any],
        fixed_vars: Array,
        weights: tuple[Array, ...],
    ) -> None:
        """Rebuild the solver if warmup says the other loop strategy is cheaper.

        Warmup already solves the real instance, so its iteration counts are a
        direct sample of this problem rather than a guess from its class.
        """
        if self.algo not in ["lbfgsb", "pgd"]:
            return

        shape, dtype = x0_spec
        batch = shape[0] if shape else 1
        want_unroll, stats = _unroll_decision(iters, self.maxiter, batch)
        logger.info(
            "Warmup iteration counts: mean=%.1f median=%.1f stdev=%.1f p99=%.1f max=%.0f "
            "(maxiter=%d); %.2f%% of starts run past the %.0f-iteration breakeven, so a "
            "batch of %d contains one with p=%.2f -> expected while_loop trips %.0f",
            stats["mean"], stats["median"], stats["stdev"], stats["p99"], stats["max"],
            self.maxiter, 100.0 * stats["frac_deep"], stats["breakeven"], batch,
            stats["p_deep_batch"], stats["expected_trips"],
        )
        if want_unroll == self.unroll:
            logger.info("Keeping unroll=%s", self.unroll)
            return

        logger.info("Switching unroll=%s -> %s and recompiling", self.unroll, want_unroll)
        t0 = time()
        evaluator, verifier, algorithm, maxiter, tol = self._build_args
        warmup_sol = self.warmup_sol
        self.__init__(evaluator, verifier, algorithm, maxiter, tol, unroll=want_unroll)  # type: ignore[misc]
        self.warmup_sol = warmup_sol
        # Precompile so the switch does not push the cost onto the first real batch.
        # x0 was donated by the warmup call above, hence a fresh array here.
        self.run.trace(jnp.zeros(shape, dtype), fixed_vars, weights).lower().compile()
        logger.info(f"Recompile Complete {time() - t0}")

    def warmup(self, warmup_data: tuple[Array, Array, tuple[Array, ...]], counting: bool = False) -> None:
        if warmup_data:
            runner = self.run if self.algo not in ["sp-lbfgsb"] else self.eval_fun
            x0, fixed_vars, weights = warmup_data
            # Captured before the call: donate_argnums=(0) invalidates x0.
            x0_spec = (x0.shape, x0.dtype)
            logger.info("Warmup Run (Dummy Data Compilation)")
            t0 = time()
            opt_x0, opt_unsat, opt_iters, _, _ = runner(x0, fixed_vars, weights)
            if not counting:
                batch_unsat_scores = jnp.sum(opt_unsat, axis=1)
                batch_best = jnp.min(batch_unsat_scores)
                loc = jnp.argmin(batch_unsat_scores)
                best_x = opt_x0[loc]
                if batch_best == 0:
                    # The caller prints it: only it knows the variable mapping and output format.
                    logger.info(f"Found a solution in warmup at index {loc}")
                    self.warmup_x = np.asarray(best_x)
                    self.warmup_sol = True

            logger.info(f"Warmup Complete {time() - t0}")

            # Warmup solved the real instance, so its iteration counts say which
            # loop strategy the rest of this job wants. Pointless if warmup already
            # found a solution.
            if not self.warmup_sol:
                self._retune_unroll(opt_iters, x0_spec, fixed_vars, weights)
