"""Batch-size throughput sweep with no convergence stragglers.

(a) bare objective + gradient, vmapped: no loops at all.
(b) 100 PGD iterations as a fixed-length scan (no early exit; line search still loops).
Production config: r20 hb10, propagation + RREF + XOR eval dropped, fp64.

Usage: STEP=0|0.05 PYTHONPATH=. python claude_scratch/batch_sweep.py <cnf> <B,B,...> <max B for (b)>
STEP=0 is jaxopt's backtracking line search (production); STEP>0 a fixed step (timing only).

Results, RTX 2000 Ada laptop (32 MB L2, 24 SMs), r20 hb10, 2026-09-25, with the 2 s warm-up --
us per point per PGD iteration:
    B            16    32    40    48    56    64   128
    line search 758   695  1292  1305  1479  1500  1655
    fixed step  193   196   198   191   192   187   199
(a) bare value_and_grad: flat to slightly falling (363 -> 314 us) from B=16 to 128, no knee at L2.
The batch penalty is the vmapped line search running every iteration to the batch's worst
backtrack count, not cache.

Measurement hazard: without the warm-up, a dense B=1..96 sweep showed ~2.2x spikes (e.g. B=48, 64)
that moved between repeat runs of the same compiled program; they tracked the laptop GPU's memory
clock dropping to 810 MHz after idle (compiles). With the warm-up, B=40..66 was flat in two runs.
A spike is only B-specific if it reproduces at the same B across runs.
"""
import logging, sys, time, numpy as np
logging.basicConfig(level=logging.ERROR)
import afsat  # repo jax config (x64, precision)
import jax, jax.numpy as jnp
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box as box
from afsat import prepare_problem
from utils import AFSATConfig
from samplers import sample_assignments
import solvers.optimisers as O

path = sys.argv[1]
batches = [int(b) for b in sys.argv[2].split(",")]
cfg = AFSATConfig(); cfg.runtime_common.n_devices = 1
prepared, prefixes, _, _ = prepare_problem(path, cfg, prefix_file="")
objs, meta = prepared.objectives, prepared.xor_rref_meta
weights = tuple(jnp.ones((o.clauses.lits.shape[0],)) for o in objs)
ev, vf = O.build_eval_verify(objs, False)
ev, _ = O.drop_projected_xor_evaluators(objs, ev)
se, sv = O.seq_eval_verify(ev, vf, xor_rref_meta=meta)
vg = jax.value_and_grad(se, has_aux=True)
import os
STEP = float(os.environ.get("STEP", "0"))
pgd = ProjectedGradient(fun=se, projection=box, has_aux=True, tol=0.0, maxiter=100, acceleration=True, stepsize=STEP)
ITERS = 100

def run_pgd(x, fv):
    st = pgd.init_state(x, fixed_vars=fv, weights=weights, hyperparams_proj=(-1, 1))
    def body(c, _):
        x, st = c
        x, st = pgd.update(x, st, fixed_vars=fv, weights=weights, hyperparams_proj=(-1, 1))
        return (x, st), None
    (x, st), _ = jax.lax.scan(body, (x, st), None, length=ITERS)
    return x

f_a = jax.jit(jax.vmap(lambda x, fv: vg(x, fv, weights)))
f_b = jax.jit(jax.vmap(run_pgd))

def timeit(f, *args, reps):
    jax.block_until_ready(f(*args))
    # Hold the GPU busy first: an idle GPU (e.g. just after compiling) sits in a low memory
    # P-state, and timing it there gave ~2.2x spikes at random batch sizes on a laptop part.
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        jax.block_until_ready(f(*args))
    t = time.perf_counter()
    for _ in range(reps):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / reps

print(f"{'B':>5} {'temp MB':>8} | {'(a) us/pt/eval':>15} {'pts*eval/s':>11} | {'(b) us/pt/iter':>15} {'pts*iter/s':>11}", flush=True)
for B in batches:
    x0, fixed = sample_assignments(jax.random.PRNGKey(B), B, prepared.n_var, "bias", jnp.asarray(prefixes))
    mem = f_a.lower(x0, fixed).compile().memory_analysis()
    temp = (mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes) / 1e6 if mem else float("nan")
    ta = timeit(f_a, x0, fixed, reps=max(3, 200 // B))
    tb = timeit(f_b, x0, fixed, reps=max(2, 16 // B)) if B <= int(sys.argv[3]) else float("nan")
    print(f"{B:5d} {temp:8.1f} | {ta / B * 1e6:15.1f} {B / ta:11.0f} | {tb / B / ITERS * 1e6:15.1f} {B * ITERS / tb:11.0f}", flush=True)
