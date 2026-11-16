# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""Streaming advice for the Dagster worker: descents pooled across batches and across prefixes.

The discrete layer sends prefixes far faster than one batch of descents completes, and successive prefixes overlap
heavily. Instead of answering each prefix with one batch, the worker keeps descending under the newest prefix and
pools every descent into an AdvicePool. When the prefix changes, the pool keeps the descents consistent with the new
prefix (they already hold its fixed values, so their violated-clause counts are unchanged under it) and drops the
rest. After each batch the advice is recomputed from the pool's elite and re-emitted, so it improves while the
prefix holds.

Conventions follow the solver: a point is a vector in [-1, 1]^n in dense variable order, -1 meaning TRUE, and a
prefix vector holds -1 / +1 for fixed variables and 0 for free ones.
"""
from __future__ import annotations

import os

import numpy as np


class LineReader:
    """Non-blocking line reader for a file descriptor (the bridge's stdin). select() on sys.stdin is unreliable
    once Python has buffered input, so this reads the raw descriptor and keeps its own buffer."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b""
        self.eof = False
        os.set_blocking(fd, False)

    def read_lines(self, block: bool) -> list[str]:
        """All complete lines available now; with block, wait until at least one arrives (or EOF)."""
        while True:
            if block:
                os.set_blocking(self.fd, True)
            try:
                chunk = os.read(self.fd, 1 << 20)
            except BlockingIOError:
                chunk = None
            finally:
                if block:
                    os.set_blocking(self.fd, False)
            if chunk == b"":
                self.eof = True
            elif chunk:
                self.buf += chunk
            *lines, self.buf = self.buf.split(b"\n")
            if lines or self.eof or not block:
                return [l.decode() for l in lines]


class AdvicePool:
    """The best descents seen under prefixes consistent with the current one, and advice computed from them.

    capacity bounds the descents kept (by violated-clause count). The elite used for advice is the best
    elite_frac of the consistent descents seen, at least 2 and at most capacity; descents that fell out of the pool
    are counted in that total, scaled by the fraction retained when the prefix changes."""

    def __init__(self, n_var: int, capacity: int = 1024, elite_frac: float = 0.5):
        self.capacity, self.elite_frac = capacity, elite_frac
        self.signs = np.zeros((0, n_var), dtype=np.int8)
        self.margins = np.zeros((0, n_var), dtype=np.float16)
        self.unsat = np.zeros(0, dtype=np.int64)
        self.seen = 0.0
        self.prefix = None

    def __len__(self) -> int:
        return int(self.unsat.size)

    def set_prefix(self, prefix: np.ndarray) -> int:
        """Keep the descents that agree with every fixed value of prefix; returns how many were dropped."""
        prefix = np.asarray(prefix)
        self.prefix = prefix
        if not len(self):
            return 0
        fixed = np.flatnonzero(prefix)
        keep = np.all(self.signs[:, fixed] == prefix[fixed].astype(np.int8), axis=1) if fixed.size else \
            np.ones(len(self), dtype=bool)
        dropped = int(len(self) - keep.sum())
        if dropped:
            self.seen *= keep.sum() / len(self)
            self.signs, self.margins, self.unsat = self.signs[keep], self.margins[keep], self.unsat[keep]
        return dropped

    def add(self, points: np.ndarray, unsat: np.ndarray) -> None:
        """Pool a batch of descended points (run under the current prefix) with their violated-clause counts."""
        self.seen += len(unsat)
        signs = np.where(points < 0, -1, 1).astype(np.int8)
        signs = np.concatenate([self.signs, signs])
        margins = np.concatenate([self.margins, np.abs(points).astype(np.float16)])
        allu = np.concatenate([self.unsat, np.asarray(unsat, dtype=np.int64)])
        order = np.argsort(allu, kind="stable")[: self.capacity]
        self.signs, self.margins, self.unsat = signs[order], margins[order], allu[order]

    def elite_size(self) -> int:
        return int(min(len(self), max(2, np.ceil(self.elite_frac * self.seen))))

    def advice(self, k: int, min_agree: float = 0.0, also_fixed: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Phases and the order of the first k free variables: agreement among the elite, then its mean |x|.
        The phase is the elite's majority sign (the best descent's sign breaks a tie). Variables with agreement
        below min_agree are not advised (measured on SHA-1 cubes, agreement below about 0.8 is right only by
        chance), so the list can be shorter than k. also_fixed (a prefix vector, such as the discrete solver's
        newest trail) keeps the variables it assigns out of the k slots without changing what the descents are
        conditioned on: the discrete solver has already set them."""
        m = self.elite_size()
        s, mg = self.signs[:m], self.margins[:m].astype(np.float32)
        mean = s.mean(axis=0)
        agree = np.abs(mean)
        phase = np.where(mean != 0, np.sign(mean), s[0]).astype(np.int8)
        free = np.flatnonzero(self.prefix == 0) if self.prefix is not None else np.arange(s.shape[1])
        if also_fixed is not None:
            free = free[np.asarray(also_fixed)[free] == 0]
        if min_agree > 0:
            free = free[agree[free] >= min_agree]
        order = free[np.lexsort((-mg.mean(axis=0)[free], -agree[free]))][:k]
        stats = {"pool": len(self), "seen": int(self.seen), "elite": m, "best": int(self.unsat[0]),
                 "unanimous": int(np.count_nonzero(agree[free] == 1.0)),
                 "agree_at_k": float(agree[order].mean()) if order.size else 0.0}
        return phase, order, stats
