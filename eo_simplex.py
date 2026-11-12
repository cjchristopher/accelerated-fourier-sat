# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""Exactly-one (EO) groups as simplices (opt-in: --eo_simplex).

One pairwise-disjoint family of EO groups is chosen from the parsed objectives. Each group of k Boolean variables
becomes a probability vector p on the k-simplex, read by the evaluator as x_v = 1 - 2 p_v (x = -1 is True):
- the chosen groups' own EO clauses leave the objective (the simplex enforces them);
- every other clause is evaluated as before. That is exact for a clause reading each chosen group at most once,
  because groups round independently. Groups that a non-OR clause reads twice are excluded from the family
  ("orthogonal" groups only); plain OR clauses that read a group twice are evaluated exactly by `or_term` instead
  of the evaluator;
- the descent is projected gradient with a mixed projection: each group onto its simplex (sort-and-shift), every
  other variable clipped to [-1, 1];
- points are decoded by argmax per group and sign elsewhere, then verified as usual.

Pinned variables (prefixes, units) are read from `fixed_vars` and the point itself, so each point may carry its own
prefix: a member pinned False keeps p = 0, and a group with a member pinned True is set to that vertex.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from boolean_whf import Objective, clause_type_ids

logger = logging.getLogger(__name__)
EO, CNF = clause_type_ids["eo"], clause_type_ids["cnf"]


@dataclass(frozen=True)
class EOSimplex:
    """The chosen family and the functions the search uses. `project` and `decode` take one point (n,) and its
    `fixed_vars` ((n,) or (1,)); `or_term` takes a point and the weight tuple and returns the weighted FE sum of
    the OR clauses it handles (or is None). `eval_masks` (one per objective) is 0 for every clause the simplex or
    `or_term` handles, so the evaluator skips them."""

    n_groups: int
    covered: int
    n_or: int
    eval_masks: tuple[np.ndarray, ...]
    project: Callable[[Array, Array], Array]
    decode: Callable[[Array, Array], Array]
    or_term: Callable[[Array, tuple[Array, ...]], Array] | None


def _rows(objs: tuple[Objective, ...]) -> list[tuple[int, int, list[int], list[int], int]]:
    """(objective, row, variables, signs, type) for every clause."""
    rows = []
    for k, o in enumerate(objs):
        lits = np.asarray(o.clauses.lits)
        mask = np.broadcast_to(np.asarray(o.clauses.mask), lits.shape)
        sign = np.broadcast_to(np.asarray(o.clauses.sign), lits.shape)
        types = np.broadcast_to(np.asarray(o.clauses.types).reshape(-1), (lits.shape[0],))
        rows += [(k, r, lits[r][mask[r]].tolist(), sign[r][mask[r]].tolist(), int(types[r])) for r in range(lits.shape[0])]
    return rows


def _choose_family(rows, tries: int = 64) -> tuple[list[tuple[tuple[int, ...], int, int]], int]:
    """Most-constrained-first packings of orthogonal all-positive EO groups under several tie-breaks; the best has
    no non-OR clause reading a group twice, then the largest coverage. Returns (groups, covered variables)."""
    groups = [(tuple(vs), k, r) for k, r, vs, ss, t in rows if t == EO and all(s > 0 for s in ss)]
    if not groups:
        return [], 0
    clauses_of: dict[int, list[int]] = {}
    for n, (_, _, vs, _, t) in enumerate(rows):
        if t != CNF:
            for v in vs:
                clauses_of.setdefault(v, []).append(n)

    def max_overlap(i: int) -> int:
        g, k, r = groups[i]
        hits: dict[int, int] = {}
        for v in g:
            for n in clauses_of.get(v, ()):
                if rows[n][:2] != (k, r):
                    hits[n] = hits.get(n, 0) + 1
        return max(hits.values(), default=0)

    orthogonal = {i for i in range(len(groups)) if max_overlap(i) <= 1}
    by_var: dict[int, set[int]] = {}
    for i in orthogonal:
        for v in groups[i][0]:
            by_var.setdefault(v, set()).add(i)
    best = None
    for seed in range(tries):
        rnd, avail, used, chosen = random.Random(seed), set(orthogonal), set(), []
        while avail:
            live = {v: s & avail for v, s in by_var.items() if v not in used}
            live = {v: s for v, s in live.items() if s}
            if not live:
                break
            v = min(live, key=lambda u: (len(live[u]), rnd.random()))
            i = rnd.choice(sorted(live[v]))
            chosen.append(i)
            used.update(groups[i][0])
            avail -= {j for u in groups[i][0] for j in by_var[u]}
        gid = {v: n for n, i in enumerate(chosen) for v in groups[i][0]}
        own = {(groups[i][1], groups[i][2]) for i in chosen}
        dup = sum(1 for k, r, vs, _, t in rows if (k, r) not in own and t != CNF
                  and len([gid[v] for v in vs if v in gid]) != len({gid[v] for v in vs if v in gid}))
        score = (-dup, len(used))
        if best is None or score > best[0]:
            best = (score, chosen)
    assert best is not None
    (neg_dup, covered), chosen = best
    if neg_dup:  # orthogonal groups never give a non-OR double read; kept as a guard
        return [], 0
    return [groups[i] for i in chosen], covered


def _or_term(family, rows, own, offsets):
    """Exact categorical evaluation of the plain OR clauses that read a chosen group more than once.

    Literal (s, v) is True when s * x_v = -1, and p_v = (1 - x_v) / 2 = P(v True). Within one group (exactly one
    member True), its positive literals P and negative literals N are all False with probability 0 if |N| >= 2 or N
    meets P, p_b if N = {b}, and 1 - sum_{v in P} p_v if N is empty. Groups and free variables are independent, so
    P(unsat) is the product of these affine factors, and FE = 2 P(unsat) - 1. Returns the (objective, row) pairs
    handled and fn(x, weights) -> weighted FE sum, or None if there are none."""
    gid = {v: n for n, (g, _, _) in enumerate(family) for v in g}
    picked, factors = [], []
    for k, r, vs, ss, t in rows:
        if t != CNF or (k, r) in own:
            continue
        ids = [gid[v] for v in vs if v in gid]
        if len(ids) == len(set(ids)):
            continue
        picked.append((k, r))
        per_group: dict[int, tuple[list[int], list[int]]] = {}
        fs, dead = [], False
        for v, sg in zip(vs, ss):
            if v in gid:
                per_group.setdefault(gid[v], ([], []))[0 if sg > 0 else 1].append(v)
            else:
                fs.append((0.5, [(v, 0.5 * sg)]))  # (1 + s x) / 2
        for pos, neg in per_group.values():
            if len(neg) >= 2 or (neg and neg[0] in pos):
                dead = True  # always satisfied: FE = -1
                break
            if neg:
                fs.append((0.5, [(neg[0], -0.5)]))  # p_b = (1 - x_b) / 2
            else:
                fs.append((1.0 - 0.5 * len(pos), [(v, 0.5) for v in pos]))
        factors.append(None if dead else fs)
    if not picked:
        return set(), None
    flat = np.array([offsets[k] + r for k, r in picked])
    is_live = np.array([f is not None for f in factors])
    live = [f for f in factors if f is not None]
    w_live, w_dead = jnp.asarray(flat[is_live]), jnp.asarray(flat[~is_live])
    F = max((len(f) for f in live), default=1)
    T = max((len(t) for f in live for _, t in f), default=1)
    a = np.ones((len(live), F))
    var = np.zeros((len(live), F, T), np.int32)
    coef = np.zeros((len(live), F, T))
    for i, f in enumerate(live):
        for j, (a0, terms) in enumerate(f):
            a[i, j] = a0
            for t, (v, c) in enumerate(terms):
                var[i, j, t], coef[i, j, t] = v, c
    a_j, var_j, coef_j = jnp.asarray(a), jnp.asarray(var), jnp.asarray(coef)

    def fn(x: Array, weights: tuple[Array, ...]) -> Array:
        w = jnp.concatenate(weights)
        value = -jnp.sum(w[w_dead])
        if len(live):
            pu = jnp.prod(a_j + jnp.sum(coef_j * x[var_j], axis=-1), axis=-1)
            value = value + jnp.sum(w[w_live] * (2.0 * pu - 1.0))
        return value

    return set(picked), fn


def build_eo_simplex(objs: tuple[Objective, ...], n_var: int) -> EOSimplex | None:
    """Choose the family and build the projection, decoding, OR term and evaluator masks; None if there is no
    orthogonal all-positive EO group to use."""
    rows = _rows(objs)
    family, covered = _choose_family(rows)
    if not family:
        return None
    own = {(k, r) for _, k, r in family}
    offsets = np.cumsum([0] + [int(o.clauses.lits.shape[0]) for o in objs])
    picked, or_fn = _or_term(family, rows, own, offsets)
    masks = [np.ones(int(o.clauses.lits.shape[0])) for o in objs]
    for k, r in own | picked:
        masks[k][r] = 0.0

    K = max(len(g) for g, _, _ in family)
    idx_np = np.full((len(family), K), n_var, np.int32)  # padding is out of range: gathers fill, scatters drop it
    for n, (g, _, _) in enumerate(family):
        idx_np[n, : len(g)] = g
    idx, members = jnp.asarray(idx_np), jnp.asarray(idx_np < n_var)
    ks = jnp.arange(1, K + 1)

    def _groups(x: Array, fixed_vars: Array):
        fixed = jnp.broadcast_to(fixed_vars, x.shape)
        xi = jnp.take(x, idx, mode="fill", fill_value=1.0)
        fi = jnp.take(fixed, idx, mode="fill", fill_value=False) & members
        live = members & ~(fi & (xi > 0))  # members pinned False stay at p = 0
        pinned_true = jnp.any(fi & (xi < 0), axis=1)  # the group is that vertex
        pinned_val = jnp.where(fi & (xi < 0), xi, 1.0)
        return xi, live, pinned_true, pinned_val

    def project(x: Array, fixed_vars: Array) -> Array:
        # Groups are projected from the unclipped point (a gradient step can leave the box, and clipping first would
        # not be the Euclidean projection); only the variables outside groups are clipped.
        xi, live, pinned_true, pinned_val = _groups(x, fixed_vars)
        p = jnp.where(live, (1.0 - xi) / 2.0, -jnp.inf)
        s = -jnp.sort(-p, axis=1)  # descending, non-members last
        csum = jnp.cumsum(jnp.where(jnp.isfinite(s), s, 0.0), axis=1)
        rho = jnp.maximum(jnp.sum(jnp.isfinite(s) & (s - (csum - 1.0) / ks > 0), axis=1), 1)
        tau = (jnp.take_along_axis(csum, rho[:, None] - 1, axis=1)[:, 0] - 1.0) / rho
        on_simplex = jnp.where(live, 1.0 - 2.0 * jnp.maximum(p - tau[:, None], 0.0), xi)
        upd = jnp.where(pinned_true[:, None], pinned_val, on_simplex)
        return jnp.clip(x, -1.0, 1.0).at[idx].set(upd, mode="drop")

    def decode(x: Array, fixed_vars: Array) -> Array:
        xi, live, pinned_true, pinned_val = _groups(x, fixed_vars)
        win = jnp.argmax(jnp.where(live, (1.0 - xi) / 2.0, -jnp.inf), axis=1)
        vertex = jnp.where(live, jnp.where(jax.nn.one_hot(win, K, dtype=bool), -1.0, 1.0), xi)
        vertex = jnp.where(jnp.any(live, axis=1)[:, None], vertex, jnp.sign(jnp.where(xi == 0, 1.0, xi)))
        upd = jnp.where(pinned_true[:, None], pinned_val, vertex)
        return jnp.sign(jnp.where(x == 0, 1.0, x)).at[idx].set(upd, mode="drop")

    return EOSimplex(
        n_groups=len(family),
        covered=covered,
        n_or=len(picked),
        eval_masks=tuple(masks),
        project=project,
        decode=decode,
        or_term=or_fn,
    )
