# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging

# import operator as ops
import os
import pickle
from fractions import Fraction
from functools import lru_cache

# from itertools import accumulate
from math import comb
from typing import NamedTuple, overload, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# TODO: When we enforce python 3.12, swap to `type X = Y` globally.
# type Clause = list[int]
# type Clauses = list[Clause]
Clause: TypeAlias = list[int]
Clauses: TypeAlias = list[Clause]

class ClauseSignature(NamedTuple):
    type: str
    len: int
    card: int


class FFT(NamedTuple):
    dft: Array | NDArray
    idft: Array | NDArray


class ClauseArrays(NamedTuple):
    lits: Array
    sign: Array
    mask: Array
    types: Array
    cards: Array


class Objective(NamedTuple):
    clauses: ClauseArrays
    ffts: FFT
    forward_mask: Array


clause_type_ids: dict[str, int] = {"card": 0, "xor": 1, "eo": 2, "nae": 3, "cnf": 4, "amo": 5, "ek": 6}


class AFSAT_DFTCache:
    """Unified caching system for AFSAT."""

    # TODO: Implement smart retrieval from disk of precomputed FFT matrices for (c_type, n, k) tuples.
    # Required for meaningful deployment for fast preprocessing.

    def __init__(self, cache_file: str = "") -> None:
        self.cache_file = cache_file
        self.memory_cache: dict[ClauseSignature, FFT] = {}
        self.modified = False

        if cache_file and os.path.exists(cache_file):
            self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.cache_file, "rb") as f:
                self.memory_cache = pickle.load(f)
        except (pickle.PickleError, EOFError):
            print(f"Warning: Could not load cache from {self.cache_file}, starting with empty cache")
            self.memory_cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.cache_file:
            return

        # Only save if there have been modifications
        if not self.modified:
            return

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.memory_cache, f)
            self.modified = False
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to save cache to {self.cache_file}: {e}")

    def __getitem__(self, key: ClauseSignature) -> FFT:
        """Get a value from the cache using bracket notation."""
        value = self.memory_cache.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found in cache")
        return value

    def __setitem__(self, key: ClauseSignature, value: FFT) -> None:
        """Set a value in the cache using bracket notation."""
        self.memory_cache[key] = value
        self.modified = True

    def __contains__(self, key: tuple) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache


class ClauseProcessor:
    def __init__(self, n_devices: int = 1, disk_cache: AFSAT_DFTCache | None = None) -> None:
        self.n_devices: int = n_devices
        self.live_cache: dict = {}
        self.disk_cache: AFSAT_DFTCache | None = None

        if disk_cache and isinstance(disk_cache, AFSAT_DFTCache):
            self.disk_cache = disk_cache

    def _wf_coeffs(self, sig: ClauseSignature) -> NDArray:
        """Calculate Walsh-Fourier coefficients for a given clause type.
        Returns the c_len+1 coefficients as per:
        [Constant, Degree 1 ESP, Degree 2 ESP, ..., Degree C_LEN ESP]
        """

        @lru_cache(maxsize=None)
        def _binomial_row(a: int, a_lim: int) -> list[int]:
            r"""Compute binomial coefficients $\binom{a}{0}, \binom{a}{1}$,...,$\binom{a}{A_{lim}}$ using recurrence relation for safe integer arithmetic."""
            S = [1] * (a_lim + 1)
            Si = 1
            symmetry = a // 2
            for i in range(1, a_lim + 1):
                if i <= symmetry:
                    Si = Si * (a - (i - 1)) // i
                    S[i] = Si
                else:
                    S[i] = S[a - i]
            return S

        def __int_noisepoly_coeffs(n: int, A: int, B: int) -> list[int]:
            r"""Compute coefficients of the polynomial $(1+\rho)^A(1-\rho)^B$

            This is used for constraint polynomials in noise analysis where:
            - $(1+\rho)$ corresponds to roots at $\rho = -1$
            - $(1-\rho)$ corresponds to roots at $\rho = 1$

            Note numpy.poly.poly.polyfromroots uses convolutions and changes to float to account for polynomial
            construction from generic roots. For even moderate $n$, this can cause numerical errors.
            Here our polynomial is monic, and only has roots in {-1, 1}, and so we can use the binomial theorem
            and symmetry exploitation to be much more accurate, and in some cases, faster.

            Args:
                n: Length of constraint
                A: Exponent on $(1+\rho)$
                B: Exponent on $(1-\rho)$

            Returns:
                List of n integer coefficients from lowest to highest degree
            """
            midpoint = (n - 1) // 2
            A_max = min(A, midpoint)
            B_max = min(B, midpoint)
            row_A = _binomial_row(A, min(A, A_max))
            row_B = _binomial_row(B, min(B, B_max))

            coeffs = [0] * n

            # $\textstyle (1+\rho)^A = \sum_{i=0}^{n}\binom{A}{i}\rho^i \quad\quad (1-\rho)^B = \sum_{j=0}^{n} \binom{B}{j}(-1)^{B-j}\rho^j$
            for t in range(midpoint + 1):
                i_lo = max(0, t - B)
                i_hi = min(t, A)
                for i in range(i_lo, i_hi + 1):
                    j = t - i
                    # $(-1)^j$ term from $(1-\rho)^B$ expansion
                    term = row_A[i] * row_B[j]
                    if j % 2 == 1:
                        term = -term
                    coeffs[t] += term

            # Mirror coefficients using palindromic symmetry
            sign = (-1) ** (B % 2)
            for t in range(midpoint + 1):
                mirror_idx = (n - 1) - t
                if mirror_idx != t:
                    coeffs[mirror_idx] = sign * coeffs[t]

            # Sanity check: coefficients should be palindromic up to sign
            assert all(abs(c1) == abs(c2) for c1, c2 in zip(coeffs, coeffs[::-1]))
            return coeffs

        def _coeff_denoms(n: int) -> list[int]:
            r"""The denominator for each $|S|$ has a $\binom{n-1}{|S|-1}$ term. Since each subsequent term in the series can be expressed
            as the product of the last term and the current term: $S_{i+1} = S_i\cdot(n-i)/(i+1)$, we save time computing the
            m = (n-1)th row of Pascal's triangle (where the top is row 0) progressively (vectorised) vs calling comb()
            The triangle is left-right symmetric, so m//2+1 is enough. Stay with native int as long as possible.
            """
            m = n - 1
            half = m // 2
            pascal_half = _binomial_row(m, half)
            if m % 2 == 0:
                full_row = pascal_half + pascal_half[-2::-1]
            else:
                full_row = pascal_half + pascal_half[::-1]
            return [c * (1 << m) for c in full_row]

        def __cnf(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for CNF (OR) clauses.
            $$\widehat{\texttt{OR}}(S) =
                \begin{dcases}
                    \frac{1}{2^{n-1}} - 1 & \abs{S} = 0 \\
                    \frac{1}{2^{n-1}} & \abs{S} \neq 0
                \end{dcases}$$
            """
            d = np.full(n + 1, 1 / (1 << (n - 1)))
            d[0] -= 1
            return d

        def __eo(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for exactly-one (EO) clauses.
            $$\widehat{\texttt{EO}}(S) =
                \begin{cases}
                    1-\frac{n}{2^{n-1}} & \abs{S} = 0 \\
                    \frac{2\abs{S}-n}{2^{n-1}} & \abs{S} \neq 0
                \end{cases}$$
            """
            d = np.arange(-n, n + 1, 2, dtype=float) / (1 << (n - 1))  # EO(S) = (2|S|-n)/2^(n-1)
            d[0] = 1 - n / (1 << (n - 1))
            return d

        def __ek(n: int, k: int) -> NDArray:
            r"""Walsh-Fourier coefficients for exactly-k ($E_k$) clauses.
            $$\begin{align*}
                \widehat{{\texttt{E}_{k}}}(S) &=
                    \begin{dcases}
                        1-\frac{\binom{n}{k}}{2^{n-1}} & \abs{S} = 0 \\
                        \frac{g_{\texttt{EK}}(\rho)_{[\rho^{|S|-1}]}}
                             {\binom{n-1}{|S|-1}2^{n-1}} & \abs{S} \neq 0
                    \end{dcases}\\
                g_{\texttt{EK}}(\rho) &= \frac{((2k-n) + n\rho)}{k}\binom{n-1}{k-1}(1+\rho)^{n-k-1}(1-p)^{k-1}\\
                &=\frac{1}{n}\binom{n}{k}((2k-n) + n\rho)(1+\rho)^{n-k-1}(1-p)^{k-1}
            \end{align*}$$
            No additional simplication here, beyond the special cases of $k=1$ or $k=n$, which are $\texttt{EO}$ and all True,
            respectively. For the former use the dedicated routine, for the latter it should be caught prior to this.
            For more info see the documentation for __card()
            """
            const = comb(n - 1, k - 1)
            g_coeffs_part = __int_noisepoly_coeffs(n - 1, n - k - 1, k - 1)
            g_coeffs_part = [const * coeff for coeff in g_coeffs_part]

            # multiply by $\frac{(2k - n) + n\rho}{k}$, so base coefficient is $\frac{2k-n}{k}$, and for a power above 0, the extra from $\frac{n}{k}$
            # Integer division is safe here as $g(\rho)$ is a sum of IVPs, and so is an IVP, so //k is safe.
            g_coeffs_p_base = [(2 * k - n) * g for g in g_coeffs_part] + [0]
            g_coeffs_p_extra = [0] + [n * g for g in g_coeffs_part]
            g_coeffs = [(base + extra) // k for base, extra in zip(g_coeffs_p_base, g_coeffs_p_extra)]

            coeffs = [numer / denom for numer, denom in zip(g_coeffs, _coeff_denoms(n))]
            d = np.array([1 - (comb(n, k) / (1 << (n - 1)))] + coeffs)
            return d

        def __nae(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for not-all-equal (NAE) clauses.
            For all even $|S|$, the coefficient at that size is $(1/2)^{n-2}$ due to cancellation of all even terms in $g(\rho)$,
            and exactly $\binom{n-1}{|S|-1}$ at all odd terms. This cancels the denominator in the general equation,
            leaving a factor of 2. This happens to essentially align with the $|S|=0$ case, but subtract 1. Hence:
            $$\widehat{\texttt{NAE}}(S) =
                \begin{dcases}
                    \frac{1}{2 ^ {n - 2}}-1 & \abs{S} = 0 \\
                    \frac{1}{2 ^ {n - 2}} & \abs{S}\ \ \text{even}\\
                    0 & \abs{S}\ \ \text{odd}
                \end{dcases}$$
            """
            d = np.zeros(n + 1, dtype=float)
            d[::2] = 1 / (1 << (n - 2))
            d[0] -= 1
            return d

        def __xor(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for XOR clauses.
            All zero expect for the full product of all literals (The nth ESP in n variables).
            $$\widehat{\texttt{XOR}}(S) =
                \begin{dcases}
                    0 & \abs{S} \neq [n] \\
                    \frac{2^{n-1}}{\binom{n-1}{n-1}2^{n-1}}=1 & \abs{S} = [n]\\
                \end{dcases}$$
            """
            d = np.zeros(n + 1, dtype=float)
            d[-1] = 1
            return d

        def __amo(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for at-most-one (AMO) clauses.
            $$\widehat{\texttt{AMO}}(S) =
            \begin{dcases}
                1-\frac{n+1}{2^{n-1}} & \abs{S} = 0 \\
                \frac{2\abs{S}-n-1}{2^{n-1}} & \abs{S} \neq 0
            \end{dcases}$$
            """

            d = np.arange(-n - 1, n, 2, dtype=float) / (1 << (n - 1))
            d[0] = 1 - (n + 1) / (1 << (n - 1))
            return d

        def __card(n: int, k: int) -> NDArray:
            r"""Walsh-Fourier coefficients for cardinality clauses.
            $$\begin{align*}
                \widehat{{\texttt{CARD}_{\geq k}}}(S) &=
                \begin{dcases}
                    1-\frac{\sum_{i=k}^{n}\binom{n}{i}}{2^{n-1}} & \abs{S} = 0 \\
                    \frac{g_{\texttt{CARD}}(\rho)_{[\rho^{|S|-1}]}}{\binom{n-1}{|S|-1}2^{n-1}} & \abs{S} \neq 0
                \end{dcases}\\
                g_{\texttt{CARD}}(\rho) &= \binom{n-1}{k-1}(1+\rho)^{n-k}(1-\rho)^{k-1}
            \end{align*}$$
            This does not simplify to as clean a closed form, so we require some manual computation.
            The dominating step for small $n$ would be np.polyfromroots, at $O(n\cdot log^{2}(n))$ for a clause of $n$ variables.
            However, to maintain numerical stability for larger $n$, we use a bespoke implementation in $O(n^2)$ for all $n$.
            """

            r"""
            N.B. $\texttt{CARD}_{\geq k} \equiv -\texttt{CARD}_{<k} \equiv \texttt{CARD}_{\leq k-1}$ (normalised in loading when part of Accelerated Fourier SAT)

            N.B. These should be detected and handled long before this function is called, but for posterity:
            Special Cases:
            $k=n$:
                e.g. clause $c = \texttt{CARD}_{\geq n}(x_1, x_2,\ldots, x_n)$ is just $c = x_1 \land x_2 \land \ldots \land x_n$.
                Then $g_{\texttt{CARD}} = (1-\rho)^{n-1}$. The coefficients are just the  $(n-1)$th row of Pascal's triangle, 
                with every second term negated, or... $(-1)^{|S|-1}\binom{n-1}{|S|-1}$, cancelling out.
                This means we get a simplified form:
                $$\widehat{{\texttt{CARD}_{\geq n}}}(S) =
                \begin{dcases}
                    1-\frac{1}{2^{n-1}} & \abs{S} = 0 \\
                    \frac{(-1)^{|S|-1}}{2^{n-1}} & \abs{S} \neq 0
                \end{dcases}$$
                Having said this... there is more merit to force reincoding this as n separate unit literals
                Notably those individual terms will be trivially minimisable. This form is distinctly not.
                Also we need to still compute the full transformation in $n$ variables if we use this form.

            $k=0$:
                The clause is trivial and should be dropped instead of evaluated as well - e.g
                clause $c = \texttt{CARD}_{\geq 0}(x_1, x_2,\ldots)$ is just $FE_c = -1$ (always SAT).

            Preserved code:
                $n = k \Rightarrow$ d = np.full(n + 1, (1 /(1 << (n - 1))), dtype=float); d[::2] *= -1; d[0] += 1
                $k=0\Rightarrow$ d = np.zeros(n + 1, dtype=float); d[0] = -1
            """
            # Transform formulated for $\geq k$ constraints, so negate $<k$ constraints and then correct.
            flip_neg_k = int(k / (abs(k)))
            k = abs(k)

            # Calculate $g(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1}$, and reformulate for zeroes: $(1-\rho) = -(\rho-1)$
            # Therefore $g(\rho) = (\rho+1)^{n-k}(\rho-1)^{k-1}(-1)^{k-1}$, and then multiply by combinatoric term $\binom{n-1}{k-1}$
            # We do this ourselves from the $\pm1$ roots for numeric stability.
            g_coeffs = __int_noisepoly_coeffs(n, n - k, k - 1)
            const = flip_neg_k * comb(n - 1, k - 1)
            g_coeffs = [const * coeff for coeff in g_coeffs]
            coeff_denoms = _coeff_denoms(n)

            # Compute the coefficients and check symmetry once more up to signs.
            coeffs = [numer / denom for numer, denom in zip(g_coeffs, coeff_denoms)]
            assert all([abs(c1) == abs(c2) for c1, c2 in zip(coeffs, coeffs[::-1])])

            const_term = (
                1 - (((1 << n) - sum(_binomial_row(n, k - 1))) / (1 << (n - 1)))
                if k <= n // 2
                else 1 - (sum(_binomial_row(n, n - k)) / (1 << (n - 1)))
            )
            d = np.array([flip_neg_k * const_term] + coeffs)
            # d = np.array([flip_neg_k * (2 * binom.cdf(k - 1, n, 0.5) - 1)] + coeffs)
            return d

        def __spectrum(sp: list[int]) -> NDArray:
            r"""Computes the Walsh-Fourier coefficients for an arbitrary symmetric spectrum.
            WARNING: This will be slow for complicated spectra, but if you have the spectra for a complicated symmetric
            contstraint then it is almost certainly likely to be worth the compute trade-off to use this exact form.

            $$\begin{align*}
                n &= \abs{sp} \\
                \widehat{{f_{sp}}}(S) &=
                \begin{dcases}
                    \frac{1}{2^n}\cdot\sum_{i=0}^{n}\binom{n}{i}sp_i & \abs{S} = 0 \\
                    \frac{g(\rho)_{[\rho^{|S|-1}]}}{\binom{n-1}{|S|-1}2^{n-1}} & \abs{S} \neq 0
                \end{dcases}\\
                g(\rho) &= \sum_{i=0}^{n-1}d_i\binom{n-1}{i}(1+\rho)^{n-1-i}(1-\rho)^{i}\\
                d_i &= \frac{1}{2}(sp_i - sp_{i+1})\\
            \end{align*}$$
            """
            n = len(sp) - 1
            d = [(sp[i] - sp[i + 1]) // 2 for i in range(n)]
            polys = [
                [comb(n - 1, i) * d[i] * j for j in __int_noisepoly_coeffs(n, n - i - 1, i)] for i in range(n) if d[i]
            ]
            g = [sum(i) for i in zip(*polys)]
            const = sum([comb(n, i) * sp[i] for i in range(10)]) / (2**n)
            f_exp = np.array([const] + [g[i] / (comb(n - 1, i) * (2 ** (n - 1))) for i in range(len(g))])
            return f_exp

        def __card_range(n: int, k_l: int, k_u: int) -> NDArray:
            r"""Construct the WF coefficients for an inclusive range constraint.
            Easiest way is to just get the relevant card encodings and add them.
            """

            ## "Faster" version with algebra exploitation?? $g(\rho) = H * Q$ , $d = k_U - k_L$
            # --- $H(\rho) = (1+\rho)^{n-1-k_U}(1-\rho)^{k_L}$ ---
            # --- $Q(\rho) = \binom{n-1}{k_L}(1+\rho)^d - \binom{n-1}{k_U}(1-\rho)^d$ ---
            d = k_u - k_l
            H = __int_noisepoly_coeffs(n, n - 1 - k_u, k_l)
            row = _binomial_row(d, d)
            Q_l, Q_u = comb(n - 1, k_l), comb(n - 1, k_u)
            Q = [row[j] * (Q_l - ((-1) ** j) * Q_u) for j in range(d + 1)]
            g_coeffs = [0] * n
            for j, qj in enumerate(Q):
                if qj == 0:
                    continue
                for i in range(n - j):
                    g_coeffs[i + j] += qj * H[i]

            # ## "Slower"?? version just computing the two full cardinalities and combine
            g_coeffs = [
                L - U  # -U since <k would normally add a -1 multiplier
                for (L, U) in zip(
                    [comb(n - 1, k_l - 1) * coeff for coeff in __int_noisepoly_coeffs(n, n - k_l, k_l - 1)],
                    [comb(n - 1, k_u - 1) * coeff for coeff in __int_noisepoly_coeffs(n, n - k_u, k_u - 1)],
                )
            ]

            coeff_denoms = _coeff_denoms(n)
            coeffs = [numer / denom for numer, denom in zip(g_coeffs, coeff_denoms)]
            const_term = 1 - sum(_binomial_row(n, k_u)[k_l : k_u + 1]) / (1 << (n - 1))
            # const_term = (2 * binom.cdf(k_l - 1, n, 0.5) - 1) + -1 * (2 * binom.cdf(k_u - 1, n, 0.5) - 1) + 1
            d = np.array([const_term] + coeffs)
            return d

        match sig.type:
            case "card":
                return __card(sig.len, sig.card)
            case "ek":
                return __ek(sig.len, sig.card)
            case "cnf":
                return __cnf(sig.len)
            case "eo":
                return __eo(sig.len)
            case "nae":
                return __nae(sig.len)
            case "xor":
                return __xor(sig.len)
            case "amo":
                return __amo(sig.len)
            case _:
                raise ValueError(f"Unknown clause type: {sig.type}")

    @overload
    def _pad(self, arrays_list: list[NDArray], pad_val: int | None = 0) -> tuple[NDArray, NDArray]: ...

    @overload
    def _pad(self, arrays_list: list[list[Clauses]], pad_val: int | None = 0) -> tuple[NDArray, NDArray]: ...

    @overload
    def _pad(self, arrays_list: list[list[int]], pad_val: int | None = 0) -> tuple[NDArray, NDArray]: ...

    def _pad(self, arrays_list, pad_val: int | None = 0) -> tuple[NDArray, NDArray]:
        arrays: list[NDArray] = [np.array(arr) if isinstance(arr, (list, int)) else arr for arr in arrays_list]
        rows = len(arrays)
        if rows > 1:
            # TODO: Think about how to handle heterogeneous clause sets when objective sharding is required?
            # N.B. If we ever need to shard objectives then the below will be required for hetero batches.
            # For homogenous batches (1 signature), we would need to adjust the shard spec
            # This would indicate the mesh and shard spec should be the first thing we do and should
            # get passed through to this stage.
            # Or perhaps we just call _pad when we are doing the batch size detection. If we determine
            # that objective sharding is relevant, then we can pad then? Notably, we already
            # use n_devices here, so sequencing is something to think about.
            # This will affect types and cards as well. Those leaves will need replication for homogenous
            # objectives, instead of sharding. A fully heterogeneous batch will need sharding at all leaves.

            # extra_rows = -rows % self.n_devices
            extra_rows = 0
        else:
            extra_rows = 0
        dtype = np.result_type(*arrays, pad_val)

        if arrays[0].ndim <= 1:
            max_length = max(len(arr) for arr in arrays)
            padded_shape = (rows + extra_rows, max_length)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros_like(padded, dtype=bool)

            for i, arr in enumerate(arrays):
                row_len = len(arr)
                padded[i, :row_len] = arr.ravel()
                mask[i, :row_len] = True

        else:
            try:
                assert np.all([arr.ndim == arrays[0].ndim for arr in arrays])
            except AssertionError:
                raise Exception("All sub arrays must have same number of dimensions")
            max_dims = np.max([arr.shape for arr in arrays], axis=0)
            padded_shape = (rows + extra_rows, *max_dims)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros(padded_shape, dtype=bool)

            for i, arr in enumerate(arrays):
                sl = tuple(slice(0, s) for s in arr.shape)
                padded[i][sl] = arr
                mask[i][sl] = True

        if np.all(mask):
            mask = np.array([True], dtype=bool)
        return padded, mask

    def __zero_if_close(self, arr: NDArray | Array, eps_tol_factor=100) -> NDArray | Array:
        tol = eps_tol_factor * np.finfo(arr.dtype).eps
        result = arr.copy()
        result.real[np.abs(result.real) < tol] = 0.0
        result.imag[np.abs(result.imag) < tol] = 0.0
        return result

    def _fft(self, signatures: list[ClauseSignature]) -> tuple[FFT, Array]:
        dfts = []
        idfts = []

        for sig in signatures:
            if sig in self.live_cache:
                # fastest
                clause_fft = self.live_cache[sig]
            elif self.disk_cache and sig in self.disk_cache:
                # check if computed in disk cache
                clause_fft = self.disk_cache[sig]
                self.live_cache[sig] = clause_fft
            else:
                # Compute FFT pair using closed form FWH Coefficients.
                # Longer clauses need more precision, so we are forced to calculate the DFT manually.
                coeffs = self._wf_coeffs(sig)[::-1]
                scale = sig.len + 1
                halfway = scale // 2 + 1
                angles = [-2 * np.pi * Fraction(k, scale) for k in range(1, halfway)]
                unities = [np.cos(angle) + 1j * np.sin(angle) for angle in angles]

                dft = np.ones(scale, dtype=complex)
                dft[1:halfway] = self.__zero_if_close(np.array(unities))
                dft[halfway:] = np.conj(dft[1 : halfway - (sig.len) % 2][::-1])
                dft = dft.reshape(-1, 1)

                idft_powers = np.array([[(i * j) % scale for i in range(scale)] for j in range(scale)], dtype=np.int32)
                idft: NDArray = np.conjugate(dft[idft_powers].squeeze())
                idft = (coeffs @ idft) / scale
                clause_fft = FFT(dft, idft)
                # update caches
                self.live_cache[sig] = clause_fft
                if self.disk_cache:
                    self.disk_cache[sig] = clause_fft

            dfts.append(clause_fft.dft)
            idfts.append(clause_fft.idft)

        dfts, dfts_mask = self._pad(dfts)
        idfts, _ = self._pad(idfts)
        return FFT(dft=jnp.array(dfts), idft=jnp.array(idfts)), jnp.array(dfts_mask)

    def _to_jax_array(self, signatures: list[ClauseSignature], clauses: Clauses) -> ClauseArrays:
        lits, mask = self._pad(clauses)

        if np.all(lits > 0) or np.all(lits < 0):
            sign = np.sign(lits.flatten()[0])
        else:
            sign = np.where(mask, np.sign(lits), 0)
        sign = jnp.array(sign, dtype=jnp.int8)

        lits = np.where(mask, np.abs(lits) - 1, lits)  # adjust for 0 indexing
        lits = jnp.array(lits, dtype=jnp.int32)
        mask = jnp.atleast_2d(jnp.array(mask))

        types = jnp.array([clause_type_ids[sig.type] for sig in signatures], dtype=jnp.int32)
        cards = jnp.array([sig.card for sig in signatures], dtype=jnp.int32)

        return ClauseArrays(lits, sign, mask, types, cards)

    def process(self, signatures: list[ClauseSignature], npclauses: Clauses) -> Objective:
        if len(signatures) > 1:
            try:
                assert len(signatures) == len(npclauses)
            except AssertionError:
                raise Exception("Heterogenous clause group must have equal number of clauses and signatures")

        clauses: ClauseArrays = self._to_jax_array(signatures, npclauses)

        ffts, dft_mask = self._fft(signatures)

        if jnp.all(dft_mask) and jnp.all(clauses.mask):
            forward_mask = jnp.array([True], dtype=bool)
        else:
            print("Heterogeneous masks", dft_mask.shape, clauses.mask.shape)
            forward_mask = dft_mask & clauses.mask[:, None, :]

        logger.info(f"Processed objective has {clauses.lits.shape[0]} clauses with signature(s): {signatures}")
        return Objective(clauses=clauses, ffts=ffts, forward_mask=forward_mask)
