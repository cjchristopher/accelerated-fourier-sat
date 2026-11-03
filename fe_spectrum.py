# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""Evaluations of a symmetric constraint's Walsh-Fourier expansion.

Selected with the environment variable AFSAT_FE_EVAL (default "nodes"):
  esp    ESP generating polynomial prod_i (w^j + a_i) at the roots of unity, weighted by the Fourier
         coefficients: the original construction. Loses accuracy beyond ~50 literals near vertices.
  spec   count-distribution generating polynomial prod_i (u_j + v_j a_i), u_j = (1+w^j)/2,
         v_j = (1-w^j)/2, weighted by the constraint's spectrum. Same polynomial (FourierSAT Thm 6);
         every factor has modulus <= 1, so it is stable at any length.
  nodes  spec rewritten as the ESP polynomial at the Cayley nodes -i cot(pi j/(n+1)), rescaled by
         v_j^n: the same operation count as esp. Its products reach sin(pi/(n+1))^-n, representable
         up to n = 176 in double precision (35 in single); an objective with longer clauses falls
         back to spec automatically.
Every mode evaluates only frequencies 0..(n+1)//2, using the exact conjugate symmetry of the
constants: the dropped terms are complex conjugates of kept ones (claude_scratch/stable_fe/
half_spectrum.pdf). Suffix "_full" evaluates all n+1 frequencies instead; "_half" is accepted as
the (default) folded form. "esp_full" runs the original kernel unchanged.

OR and NAE objectives skip the transform in every mode except "esp_full": their expansions
collapse to products (closed_form_eval), exactly and stably at any length. AFSAT_FE_CLOSED=0
turns that off. XOR objectives are always a product (optimisers.evaluate_xor).
Correctness check: scripts/check_fe_spectrum.py.

The constants reuse AFSAT's own roots of unity (exact rational angles, exact conjugate symmetry); the
spectrum row is summed with indices reduced mod n+1 and divided once. Derivation and measurements:
claude_scratch/stable_fe/stable_fe.pdf.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from math import comb, pi, sin

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from boolean_whf import ClauseProcessor, ClauseSignature, Objective, clause_type_ids

logger = logging.getLogger(__name__)

FE_EVAL_BASES = ("esp", "spec", "nodes")
FE_EVAL_DEFAULT = "nodes"
CLOSED_FORM_TYPES = ("cnf", "nae")
_TYPE_NAMES = {v: k for k, v in clause_type_ids.items()}


def node_max_len() -> int:
    """Longest clause whose node-form products, up to sin(pi/(n+1))^-n, stay finite in the active precision."""
    return 176 if jax.config.jax_enable_x64 else 35


def fe_eval_mode() -> tuple[str, bool]:
    """(base construction, folded) from AFSAT_FE_EVAL. ("esp", False) is the original kernel."""
    raw = os.environ.get("AFSAT_FE_EVAL", FE_EVAL_DEFAULT)
    base, _, suffix = raw.partition("_")
    if base not in FE_EVAL_BASES or suffix not in ("", "half", "full"):
        raise ValueError(f"AFSAT_FE_EVAL={raw!r}: expected one of {FE_EVAL_BASES}, optionally with '_full'")
    return base, suffix != "full"


def closed_forms_enabled() -> bool:
    return os.environ.get("AFSAT_FE_CLOSED", "1").strip().lower() not in ("0", "false", "no", "off")


def closed_form_eval(obj: Objective) -> Callable[[Array], Array] | None:
    """Per-clause FE values (N,) for an all-OR or all-NAE objective, without the transform; else None.

    With p_i = (1 + a_i)/2 the probability literal i is False (a_i = +1 is False):
      OR   f(k) = c = 2^(1-n) for every k >= 1, and sum_k e_k(a) = prod_i (1 + a_i), so
           FE = f(0) - c + c prod_i (1 + a_i) = 2 prod_i p_i - 1          (UNSAT iff all False)
      NAE  FE = 2 (prod_i p_i + prod_i (1 - p_i)) - 1                     (UNSAT iff all equal)
    Every factor lies in [0, 1], so neither loses accuracy with length. Padding slots are masked to 1.
    """
    types = {_TYPE_NAMES[int(t)] for t in np.asarray(obj.clauses.types).reshape(-1)}
    if len(types) != 1 or (kind := types.pop()) not in CLOSED_FORM_TYPES:
        return None
    mask = np.asarray(obj.clauses.mask)
    where = None if mask.all() else jnp.asarray(mask)
    all_false = lambda a: jnp.prod(0.5 * (1.0 + a), axis=-1, where=where)
    logger.info(f"FE evaluation: closed form for {obj.clauses.lits.shape[0]} {kind} clauses")
    if kind == "cnf":
        return lambda a: 2.0 * all_false(a) - 1.0
    return lambda a: 2.0 * (all_false(a) + all_false(-a)) - 1.0


@lru_cache(maxsize=None)
def spectrum(sig: ClauseSignature) -> tuple[int, ...]:
    """FourierSAT Def. 1: sp_c = constraint value (-1 SAT, +1 UNSAT) when c literals are True (-1).

    Checked once per signature against AFSAT's closed-form Fourier coefficients through the exact
    (integer) Walsh transform restricted to symmetric functions, so a convention mismatch fails loudly.
    """
    n, k = sig.len, sig.card
    c = np.arange(n + 1)
    sat = {
        "cnf": c >= 1, "card": c >= k, "ek": c == k, "eo": c == 1, "amo": c <= 1,
        "nae": (c > 0) & (c < n), "xor": c % 2 == 1,
    }[sig.type]
    sp = np.where(sat, -1, 1)
    exact = [
        Fraction(sum(int(sp[i]) * sum((-1) ** j * comb(d, j) * comb(n - d, i - j) for j in range(min(d, i) + 1))
                     for i in range(n + 1)), 2**n)
        for d in range(n + 1)
    ]
    afsat = np.asarray(ClauseProcessor()._wf_coeffs(sig), dtype=float)
    scale = max(abs(float(e)) for e in exact) or 1.0
    err = max(abs(float(e) - a) for e, a in zip(exact, afsat)) / scale
    if err > 1e-9:
        raise ValueError(f"Spectrum of {sig} disagrees with its Fourier coefficients (relative error {err:.1e})")
    return tuple(int(s) for s in sp)


def _half_weights(m: int, width: int) -> np.ndarray:
    """Weights folding frequency m-j onto j (conjugate pairs), for a clause with m = n+1 frequencies."""
    w = np.zeros(width)
    w[0] = 1.0
    top = (m - 1) // 2
    w[1 : top + 1] = 2.0
    if m % 2 == 0:
        w[m // 2] = 1.0  # w^j = -1: real, unpaired
    return w


def _row_constants(sig: ClauseSignature, unit: np.ndarray, idft: np.ndarray, base: str, width: int):
    """(factor constants, weight row), zero-padded to `width` frequencies, for one clause signature.
    `unit` is AFSAT's root-of-unity row for this signature (length n+1), `idft` its coefficient row."""
    n = sig.len
    m = n + 1
    if base == "esp":
        return (unit,), idft[:m]
    sp = np.asarray(spectrum(sig), dtype=float)
    idx = np.outer(np.arange(m), np.arange(m)) % m
    kappa = (np.conj(unit[idx]) @ sp) / m  # sum with reduced indices, divide once
    if base == "spec":
        return ((1 + unit) / 2, (1 - unit) / 2), kappa
    # nodes: v_j^n = i^n sin(pi j/m)^n e^{-i pi j n/m}; phase reduced exactly before the float conversion
    j = np.arange(1, m)
    cot_neg = np.concatenate([[0.0], [-1.0 / np.tan(pi * float(Fraction(int(t), m))) for t in j]])
    phase = [float((Fraction(n, 2) - Fraction(int(t) * n, m)) % 2) for t in j]
    vpow = np.array([sin(pi * float(Fraction(int(t), m))) ** n for t in j]) * np.exp(1j * pi * np.array(phase))
    return (cot_neg,), np.concatenate([[kappa[0]], kappa[1:] * vpow])


def make_clause_eval(obj: Objective, base: str, half: bool) -> Callable[[Array], Array]:
    """Per-clause FE values (N,) from signed literal values (N, K), for the selected construction."""
    if closed_forms_enabled() and (closed := closed_form_eval(obj)) is not None:
        return closed
    dft = np.asarray(obj.ffts.dft)[..., 0]  # (R, M): R = 1 (one signature) or N (heterogeneous)
    idft = np.asarray(obj.ffts.idft)
    rows, width = dft.shape
    K = width - 1
    types = np.asarray(obj.clauses.types).reshape(-1)
    cards = np.asarray(obj.clauses.cards).reshape(-1)
    mask = np.asarray(obj.clauses.mask)
    lens = mask.sum(axis=-1) if mask.shape[-1] == K else np.full(rows, K)
    out_width = width // 2 + 1 if half else width
    if base == "nodes" and K > node_max_len():
        logger.info(f"FE evaluation: clauses of length {K} exceed the node form's range ({node_max_len()}); "
                    "using the spectrum form for this objective")
        base = "spec"

    cache: dict[ClauseSignature, tuple] = {}
    n_consts = {"esp": 1, "spec": 2, "nodes": 1}[base]
    consts = [np.zeros((rows, out_width), dtype=complex) for _ in range(n_consts)]
    weight = np.zeros((rows, out_width), dtype=complex)
    for r in range(rows):
        sig = ClauseSignature(_TYPE_NAMES[int(types[r])], int(lens[r]), int(cards[r]))
        if sig not in cache:
            m = sig.len + 1
            c, w = _row_constants(sig, dft[r, :m], idft[r, :m], base, width)
            if half:
                w = w * _half_weights(m, m)
            cache[sig] = (c, w)
        c, w = cache[sig]
        keep = min(len(w), out_width)
        for dst, src in zip(consts, c):
            dst[r, :keep] = src[:keep]
        weight[r, :keep] = w[:keep]

    fmask = obj.forward_mask
    fmask = True if bool(jnp.all(fmask)) else jnp.asarray(fmask)[:, :out_width, :]
    weight = jnp.asarray(weight)
    logger.info(f"FE evaluation '{base}{'_half' if half else ''}': {len(cache)} signature(s), "
                f"{out_width} of {width} frequencies")

    if base == "esp":
        W = jnp.asarray(consts[0])[:, :, None]
        factors = lambda a: W + a[:, None, :]
    elif base == "spec":
        U, V = (jnp.asarray(c)[:, :, None] for c in consts)
        factors = lambda a: U + V * a[:, None, :]
    else:
        Z = jnp.asarray(consts[0].real)[:, :, None]

        def factors(a: Array) -> Array:
            shape = (a.shape[0], out_width, a.shape[1])
            return jax.lax.complex(jnp.broadcast_to(a[:, None, :], shape), jnp.broadcast_to(Z, shape).astype(a.dtype))

    def clause_eval(assignment: Array) -> Array:
        prods = jnp.prod(factors(assignment), axis=-1, where=fmask)
        if base == "nodes":
            prods = prods.at[:, 0].set(1.0)  # j = 0 is the point at infinity: Q(1) = 1
        return jnp.sum(jnp.real(weight * prods), axis=-1)

    return clause_eval
