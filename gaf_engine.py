"""
gaf_engine.py --- matrix-free computation engine for the Grassmann Algebraic Framework
=====================================================================================

Self-contained (NumPy only). Intended as the shared numerical layer that
evaluation scripts, trainers' analysis tools and notebooks import.

    import gaf_engine as gafe

Operator (post-audit specification, diagonal KEPT)
--------------------------------------------------
    S(a,b) = (a(x)b + b(x)a)/2            symmetric part
    G(a,b) = (a(x)b - b(x)a)/2            Grassmann (antisymmetric) part
    M(a,b; lam) = S(a,b) + lam * G(a,b),  lam = 2*kappa - 1
               = kappa * a(x)b + (1-kappa) * b(x)a      (convex form)

    M_ii = a_i b_i   (nilpotency constrains G only; G_ii = 0)
    M(a,b; kappa) = M(b,a; 1-kappa)                       (reversal identity)
    lam = sigma * gamma: gamma = symmetric magnitude from the pair,
                         sigma = one bit of construction sign, supplied externally

Design
------
Nothing here builds a d x d matrix unless you ask for it (`to_dense`, or the
dense reference functions `S`, `G`, `M`). Every composed object is a
`Composition`: a weighted sum of pair terms

    sum_t  s_t * S(p_t, q_t) + g_t * G(p_t, q_t)

stored as the k distinct vectors involved plus index/coefficient arrays.
Inner products, norms, Frobenius similarity, quadratic forms, mat-vec
products and the top eigenpair of the S-part are computed from Gram
matrices of those vectors, i.e. O(k^2 d + T^2) instead of O(d^2).
M(a,b;lam) is the term s = 1, g = lam; a self-composition w(x)w is (w, w, s=1).

Contents
--------
1. Parameter conversions: lam_from_kappa, kappa_from_lam, lam_from_sigma_gamma
2. Dense reference: outer, sym, skew, S, G, M, M_legacy, lift, frobenius_inner
3. Composition (matrix-free algebra) and compose / pair_set
4. Closed forms for a single pair: pair_norms, order_similarity
5. Magnitude estimators: gamma_bigram (with H0 floor), bigram_counts,
   gamma_from_counts, canonical_order, gamma_cos2 (paper6 legacy kappa)
6. Paraphrasability: paraphrasability, ceiling, paraphrase_norm, phrasehood,
   best_lexical_paraphrase, rank_paraphrases
7. Seven-way trigram factorization (A-G), vectorized: subset_sims_batch,
   trigram_patterns_batch, trigram_patterns
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence, Union

import numpy as np

__version__ = "1.0.0"
__all__ = [
    # parameters
    "lam_from_kappa", "kappa_from_lam", "lam_from_sigma_gamma",
    # dense reference
    "outer", "sym", "skew", "S", "G", "M", "M_with_lambda", "M_legacy", "lift",
    "frobenius_inner", "frobenius_sim",
    # matrix-free
    "Composition", "pair_set", "random_pairs", "compose", "compose_simpler",
    "compose_words",
    # closed forms
    "pair_norms", "order_similarity",
    # magnitude estimators
    "gamma_bigram", "gamma_bigram_by_freq", "bigram_counts",
    "gamma_from_counts", "canonical_order", "gamma_cos2",
    # paraphrasability
    "paraphrasability", "ceiling", "paraphrase_norm", "phrasehood",
    "best_lexical_paraphrase", "rank_paraphrases",
    # trigram factorization
    "TRIGRAM_PAIRS", "TRIGRAM_PATTERNS", "PATTERN_LABELS", "INFORMATIVE",
    "subset_sims_batch", "trigram_patterns_batch", "trigram_patterns",
]

EPS = 1e-12
Number = Union[float, int]
LamSpec = Union[Number, Sequence[float], np.ndarray, Callable[[int, int], float]]


# =============================================================================
# 1. Parameter conversions
# =============================================================================

def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def lam_from_kappa(kappa):
    """lam = 2*kappa - 1 (scalar or array)."""
    return _out(2.0 * np.asarray(kappa, dtype=float) - 1.0)


def kappa_from_lam(lam):
    """kappa = (lam + 1) / 2 (scalar or array)."""
    return _out((np.asarray(lam, dtype=float) + 1.0) / 2.0)


def lam_from_sigma_gamma(sigma, gamma):
    """lam = sigma * gamma. sigma in {+1, -1} is the construction sign
    (supplied externally); gamma >= 0 is the symmetric pair magnitude."""
    return _out(np.asarray(sigma, dtype=float) * np.asarray(gamma, dtype=float))


def _resolve_lam(lam=None, kappa=None):
    if lam is not None and kappa is not None:
        raise ValueError("give lam or kappa, not both")
    if kappa is not None:
        return lam_from_kappa(kappa)
    return 0.0 if lam is None else lam


# =============================================================================
# 2. Dense reference implementation (for verification and small d)
# =============================================================================

def outer(a, b):
    return np.outer(np.asarray(a, float), np.asarray(b, float))


def sym(X):
    X = np.asarray(X, float)
    return 0.5 * (X + X.T)


def skew(X):
    X = np.asarray(X, float)
    return 0.5 * (X - X.T)


def S(a, b):
    """Symmetric part of a(x)b."""
    return sym(outer(a, b))


def G(a, b):
    """Grassmann (antisymmetric) part of a(x)b. G_ii = 0 identically."""
    return skew(outer(a, b))


def M_with_lambda(a, b, lam=None, *, kappa=None, diagonal="keep"):
    """Dense M(a,b;lam) = S + lam*G.

    diagonal="keep" (default) is the corrected operator (M_ii = a_i b_i).
    diagonal="zero" reproduces the pre-audit numbers (zeroed diagonal).
    """
    lam = float(_resolve_lam(lam, kappa))
    X = S(a, b) + lam * G(a, b)
    if diagonal == "zero":
        X = X.copy()
        np.fill_diagonal(X, 0.0)
    elif diagonal != "keep":
        raise ValueError("diagonal must be 'keep' or 'zero'")
    return X


def M_legacy(a, b, lam=None, *, kappa=None):
    """Pre-audit operator with zeroed diagonal (regression only)."""
    return M_with_lambda(a, b, lam, kappa=kappa, diagonal="zero")


def lift(w):
    """Canonical injection R^n -> R^{n x n}: iota(w) = w(x)w = M(w,w;lam) for every lam."""
    return outer(w, w)


def frobenius_inner(X, Y):
    return float(np.sum(np.asarray(X, float) * np.asarray(Y, float)))


def frobenius_sim(X, Y):
    """Frobenius cosine. Accepts dense arrays or Compositions (mixed is densified)."""
    if isinstance(X, Composition) and isinstance(Y, Composition):
        return X.cos(Y)
    if isinstance(X, Composition):
        X = X.to_dense()
    if isinstance(Y, Composition):
        Y = Y.to_dense()
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    if nx < EPS or ny < EPS:
        return 0.0
    return frobenius_inner(X, Y) / (nx * ny)


# =============================================================================
# 3. Matrix-free compositions
# =============================================================================

class Composition:
    """Weighted sum of pair terms  sum_t s_t*S(p_t,q_t) + g_t*G(p_t,q_t).

    V : (k, d) distinct vectors; I, J : (T,) indices into V (p_t = V[I_t], q_t = V[J_t]);
    s, g : (T,) coefficients. M(a,b;lam) with weight w is (s=w, g=w*lam).

    All algebra is exact; no d x d matrix is formed.
    """

    __slots__ = ("V", "I", "J", "s", "g", "_gram")

    def __init__(self, V, I, J, s, g):
        V = np.atleast_2d(np.asarray(V, dtype=float))
        I = np.asarray(I, dtype=np.intp).ravel()
        J = np.asarray(J, dtype=np.intp).ravel()
        s = np.asarray(s, dtype=float).ravel()
        g = np.asarray(g, dtype=float).ravel()
        if not (len(I) == len(J) == len(s) == len(g)):
            raise ValueError("I, J, s, g must have equal length")
        if len(I) and (I.max() >= len(V) or J.max() >= len(V) or I.min() < 0 or J.min() < 0):
            raise IndexError("term index out of range")
        self.V, self.I, self.J, self.s, self.g = V, I, J, s, g
        self._gram = None

    # ---- constructors -------------------------------------------------------
    @classmethod
    def pair(cls, a, b, lam=None, *, kappa=None, weight=1.0):
        """M(a,b;lam) as a Composition."""
        lam = float(_resolve_lam(lam, kappa))
        return cls(np.vstack([a, b]), [0], [1], [weight], [weight * lam])

    @classmethod
    def lift(cls, w, weight=1.0):
        """w(x)w (self-composition; lam-independent)."""
        return cls(np.atleast_2d(np.asarray(w, float)), [0], [0], [weight], [0.0])

    @classmethod
    def from_vectors(cls, vectors, pairs="exhaustive", lam: LamSpec = 0.0, *, kappa=None):
        """Sum of M(v_i, v_j; lam_ij) over a pair set (ordered i <= j by position)."""
        V = np.atleast_2d(np.asarray(vectors, dtype=float))
        P = pair_set(len(V), pairs) if isinstance(pairs, str) else [tuple(p) for p in pairs]
        if kappa is not None:
            if callable(kappa):
                lam = (lambda i, j, _k=kappa: 2.0 * _k(i, j) - 1.0)
            else:
                lam = lam_from_kappa(kappa)
        lam_arr = _lam_for_pairs(lam, P)
        if not P:
            return cls(V, [], [], [], [])
        I, J = zip(*P)
        return cls(V, I, J, np.ones(len(P)), lam_arr)

    # ---- shape --------------------------------------------------------------
    @property
    def k(self):
        return self.V.shape[0]

    @property
    def d(self):
        return self.V.shape[1]

    @property
    def T(self):
        return len(self.I)

    def __repr__(self):
        return f"Composition(k={self.k}, d={self.d}, terms={self.T})"

    # ---- linear structure ---------------------------------------------------
    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        if not isinstance(other, Composition):
            return NotImplemented
        if other.d != self.d:
            raise ValueError("dimension mismatch")
        off = self.k
        return Composition(np.vstack([self.V, other.V]),
                           np.concatenate([self.I, other.I + off]),
                           np.concatenate([self.J, other.J + off]),
                           np.concatenate([self.s, other.s]),
                           np.concatenate([self.g, other.g]))

    __radd__ = __add__

    def __mul__(self, c):
        c = float(c)
        return Composition(self.V, self.I, self.J, self.s * c, self.g * c)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def S_part(self):
        return Composition(self.V, self.I, self.J, self.s, np.zeros_like(self.g))

    def G_part(self):
        return Composition(self.V, self.I, self.J, np.zeros_like(self.s), self.g)

    # ---- Frobenius geometry -------------------------------------------------
    def gram(self):
        if self._gram is None:
            self._gram = self.V @ self.V.T
        return self._gram

    def _kernels(self, other):
        X = self.gram() if other is self else self.V @ other.V.T
        PR = X[np.ix_(self.I, other.I)]
        QS = X[np.ix_(self.J, other.J)]
        PS = X[np.ix_(self.I, other.J)]
        QR = X[np.ix_(self.J, other.I)]
        a, b = PR * QS, PS * QR
        return 0.5 * (a + b), 0.5 * (a - b)   # <S,S>, <G,G> per term pair

    def inner(self, other, part="M"):
        """<self, other>_F; part in {'M','S','G'} (S and G are Frobenius-orthogonal)."""
        if self.T == 0 or other.T == 0:
            return 0.0
        KS, KG = self._kernels(other)
        vs = self.s @ KS @ other.s
        vg = self.g @ KG @ other.g
        return float({"M": vs + vg, "S": vs, "G": vg}[part])

    def norm(self, part="M"):
        return math.sqrt(max(self.inner(self, part), 0.0))

    def cos(self, other, part="M"):
        n1, n2 = self.norm(part), other.norm(part)
        if n1 < EPS or n2 < EPS:
            return 0.0
        return self.inner(other, part) / (n1 * n2)

    # ---- action on vectors --------------------------------------------------
    def quad(self, x):
        """x^T M x (G-free). x: (d,) -> float, or (N, d) -> (N,)."""
        X = np.atleast_2d(np.asarray(x, float))
        P = X @ self.V.T
        val = (P[:, self.I] * P[:, self.J]) @ self.s
        return float(val[0]) if np.ndim(x) == 1 else val

    def matvec(self, x):
        """M x, using M_t = (s+g)/2 p q^T + (s-g)/2 q p^T."""
        x = np.asarray(x, float)
        Vx = self.V @ x
        cp = 0.5 * (self.s + self.g) * Vx[self.J]
        cq = 0.5 * (self.s - self.g) * Vx[self.I]
        return cp @ self.V[self.I] + cq @ self.V[self.J]

    def to_dense(self, diagonal="keep"):
        Vi, Vj = self.V[self.I], self.V[self.J]
        A = (Vi * (0.5 * (self.s + self.g))[:, None]).T @ Vj
        B = (Vj * (0.5 * (self.s - self.g))[:, None]).T @ Vi
        X = A + B
        if diagonal == "zero":
            np.fill_diagonal(X, 0.0)
        elif diagonal != "keep":
            raise ValueError("diagonal must be 'keep' or 'zero'")
        return X

    # ---- spectrum of the symmetric part -------------------------------------
    def s_coefficients(self):
        """C (k x k) with S_part = V^T C V."""
        C = np.zeros((self.k, self.k))
        np.add.at(C, (self.I, self.J), 0.5 * self.s)
        np.add.at(C, (self.J, self.I), 0.5 * self.s)
        return C

    def top_eigen(self, tol=1e-10):
        """Largest eigenvalue of S_part and a unit eigenvector (in span(V)).

        Equivalent to argmax_w (w^T M w)/||w||^2 restricted to span(V).
        """
        Gm = self.gram()
        ev, Q = np.linalg.eigh(Gm)
        keep = ev > tol * max(ev.max(initial=0.0), EPS)
        if not keep.any():
            return 0.0, np.zeros(self.d)
        R = Q[:, keep] * np.sqrt(ev[keep])
        C = self.s_coefficients()
        mu, U = np.linalg.eigh(R.T @ C @ R)
        i = int(np.argmax(mu))
        w = self.V.T @ (C @ R @ U[:, i])
        n = np.linalg.norm(w)
        return float(mu[i]), (w / n if n > EPS else np.zeros(self.d))

    def rank(self, tol=1e-10):
        ev = np.linalg.eigvalsh(self.gram())
        return int((ev > tol * max(ev.max(initial=0.0), EPS)).sum())


def _lam_for_pairs(lam: LamSpec, pairs):
    T = len(pairs)
    if callable(lam):
        return np.array([float(lam(i, j)) for i, j in pairs], dtype=float)
    arr = np.asarray(lam, dtype=float)
    if arr.ndim == 0:
        return np.full(T, float(arr))
    if arr.shape != (T,):
        raise ValueError(f"lam array must have length {T} (one per pair)")
    return arr


def pair_set(n: int, kind: str = "exhaustive"):
    """Pair sets of the lattice: 'diagonal' (DAOP), 'adjacent', 'exhaustive' (i<j),
    'nonadjacent' (i<j, j-i>1), 'window:K' (0 < j-i <= K), or 'ngram:N' --- every
    pair inside a sliding N-gram window, i.e. gap <= N-1 (== window:(N-1)):
    'ngram:1' is unigrams (no pairs), 'ngram:2' == 'adjacent', and as N grows
    toward the sentence length it approaches 'exhaustive'."""
    if kind == "diagonal":
        return [(i, i) for i in range(n)]
    if kind == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "exhaustive":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if kind == "nonadjacent":
        return [(i, j) for i in range(n) for j in range(i + 2, n)]
    if kind.startswith("window:") or kind.startswith("ngram:"):
        val = int(kind.split(":", 1)[1])
        K = val if kind.startswith("window:") else max(0, val - 1)
        return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + K + 1))]
    raise ValueError(f"unknown pair set: {kind!r}")


def random_pairs(n: int, k=None, *, seed=None, rng=None):
    """k random distinct unordered pairs (i < j), the count-matched NULL control.

    By default k = n - 1, the cardinality of 'adjacent', so a random pair set has
    exactly as many terms as the adjacent one but with positions chosen at random
    from the same space as 'exhaustive' (all i < j). Comparing 'adjacent' against
    many draws of random_pairs is a permutation test on the pair set: it isolates
    whether the *structure* of adjacency matters beyond the mere *number* of pairs.

    Pairs are returned left-to-right (i < j), matching the sigma orientation of
    'adjacent' and 'exhaustive'. For n <= 2 there is only one pair, so the control
    coincides with 'adjacent'; the contrast grows with sentence length.

    Reproducible: pass seed (new default_rng) or rng (an existing Generator, so a
    caller can advance one generator across a whole dataset pass).
    """
    if n < 2:
        return []
    allp = [(i, j) for i in range(n) for j in range(i + 1, n)]   # C(n,2), == exhaustive
    if k is None:
        k = n - 1                                                # match 'adjacent'
    k = min(int(k), len(allp))
    r = rng if rng is not None else np.random.default_rng(seed)
    idx = r.choice(len(allp), size=k, replace=False)
    return [allp[i] for i in sorted(int(x) for x in idx)]


def compose_simpler(vectors, pairs="exhaustive", lam: LamSpec = 0.0, *, kappa=None):
    """Compose a sequence of vectors over a pair set. Returns a Composition.

    lam: scalar, per-pair array, or callable (i, j) -> lam_ij  (e.g. sigma*gamma).
    """
    return Composition.from_vectors(vectors, pairs, lam, kappa=kappa)


def compose_words(words: Sequence[str], vectors: Mapping, pairs="exhaustive",
                  lam: Union[LamSpec, Callable[[str, str], float]] = 0.0,
                  oov="skip"):
    """Look words up in any mapping (dict, gensim KeyedVectors, ...) and compose.

    A callable lam is applied to the WORD STRINGS of each pair, (u, v) -> lam_uv,
    e.g. lam=lambda u, v: canonical_order(cnt, u, v) * gamma_from_counts(cnt, u, v).
    Out-of-vocabulary words are skipped (oov="skip") or raise KeyError (oov="raise");
    pair positions refer to the kept words.
    Returns (Composition or None, kept_words).
    """
    if oov not in ("skip", "raise"):
        raise ValueError("oov must be 'skip' or 'raise'")
    kept = []
    for w in words:
        if w in vectors:
            kept.append(w)
        elif oov == "raise":
            raise KeyError(w)
    if not kept:
        return None, kept
    V = np.vstack([np.asarray(vectors[w], float) for w in kept])
    lam_spec = (lambda i, j, _f=lam: _f(kept[i], kept[j])) if callable(lam) else lam
    return compose_simpler(V, pairs, lam_spec), kept


# =============================================================================
# 4. Closed forms for a single pair
# =============================================================================

def pair_norms(a, b):
    """(||S(a,b)||_F^2, ||G(a,b)||_F^2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab2 = (a @ a) * (b @ b)
    d2 = float(a @ b) ** 2
    return 0.5 * (ab2 + d2), 0.5 * (ab2 - d2)


def order_similarity(a, b, lam):
    """cos_F(M(a,b;lam), M(b,a;lam)) = (||S||^2 - lam^2||G||^2)/(||S||^2 + lam^2||G||^2).
    Even in lam: registers THAT order differs, not WHICH order."""
    s2, g2 = pair_norms(a, b)
    lam = np.asarray(lam, float)
    den = s2 + lam ** 2 * g2
    return _out(np.where(den > EPS, (s2 - lam ** 2 * g2) / np.maximum(den, EPS), 0.0))


# =============================================================================
# 5. Magnitude estimators (gamma) and corpus helpers
# =============================================================================

def gamma_bigram_by_freq(n_ab, n_ba, floor=True):
    """gamma(a,b) from directed adjacent counts.

    floor=True : max(0, |N_ab - N_ba| - sqrt(2N/pi)) / N,  N = N_ab + N_ba
                 (removes E|difference| under the binomial null, so ranking
                  by gamma is not a ranking by rarity)
    floor=False: |N_ab - N_ba| / N
    Pairs never seen (N = 0) get gamma = 0.
    """
    n_ab = np.asarray(n_ab, float)
    n_ba = np.asarray(n_ba, float)
    N = n_ab + n_ba
    diff = np.abs(n_ab - n_ba)
    num = np.maximum(0.0, diff - np.sqrt(2.0 * N / math.pi)) if floor else diff
    return _out(np.where(N > 0, num / np.maximum(N, 1.0), 0.0))


def bigram_counts(tokens: Iterable[str]) -> Counter:
    """Directed adjacent co-occurrence counts N[(a,b)]."""
    toks = list(tokens)
    return Counter(zip(toks, toks[1:]))


def gamma_from_counts(counts: Mapping, a: str, b: str, floor=True) -> float:
    return gamma_bigram_by_freq(counts.get((a, b), 0), counts.get((b, a), 0), floor=floor)


def canonical_order(counts: Mapping, a: str, b: str) -> int:
    """+1 if 'a b' is the more frequent order, -1 if 'b a', 0 if tied/unseen.

    A corpus proxy for sigma in the canonical-vs-marked ORDER case only.
    Voice (active/passive) sign cannot be read off counts and must be annotated.
    """
    return int(np.sign(counts.get((a, b), 0) - counts.get((b, a), 0)))


def gamma_cos2(a, b):
    """cos^2(a,b): the embedding-derived symmetric magnitude used as intrinsic
    kappa in the paper6 era (kappa = (1 + cos^2)/2, i.e. lam = cos^2 with sigma=+1).
    Kept for reproducing those runs; by the impossibility theorem it cannot
    supply the sign."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float((a @ b / (na * nb)) ** 2)


# =============================================================================
# 6. Paraphrasability  (w ~ P)
# =============================================================================
#
# Para(w,P) = <w(x)w, M_P>_F / (||w(x)w|| ||M_P||) = w^T S_P w / (||w||^2 ||M_P||)
#
# The numerator is S-only (G is annihilated by the symmetric probe); the
# denominator is NOT: ||M_P||^2 = ||S_P||^2 + lam^2 ||G_P||^2 for uniform lam.
# Hence Para with normalize="M" decreases with |lam|. Quantities that are
# lam-invariant: Para with normalize="S", and the ratio Para / ceiling for
# either normalization (paraphrase_norm).

def _probe_norm(comp: Composition, normalize: str):
    if normalize not in ("M", "S"):
        raise ValueError("normalize must be 'M' or 'S'")
    return comp.norm(normalize)


def paraphrasability(w, comp: Composition, normalize="M") -> float:
    w = np.asarray(w, float)
    nw2 = float(w @ w)
    n = _probe_norm(comp, normalize)
    if nw2 < EPS or n < EPS:
        return 0.0
    return comp.quad(w) / (nw2 * n)


def ceiling(comp: Composition, normalize="M") -> float:
    """max_w Para(w, P): the rank-one (single-word) ceiling. Equals
    lambda_max(S_P)/||M_P|| (or /||S_P||); for one pair it is
    (||a|| ||b|| + a.b) / (2 ||M||), attained at the normalized bisector."""
    n = _probe_norm(comp, normalize)
    if n < EPS:
        return 0.0
    mu, _ = comp.top_eigen()
    if comp.d > comp.rank():          # a null direction exists => max >= 0
        mu = max(mu, 0.0)
    return mu / n


def paraphrase_norm(w, comp: Composition) -> float:
    """Para / ceiling in [.., 1]; lam-invariant and normalization-invariant."""
    c = ceiling(comp, "S")
    return paraphrasability(w, comp, "S") / c if c > EPS else 0.0


def phrasehood(comp: Composition, normalize="M") -> float:
    """1 - ceiling: how much of P no single word can carry."""
    return 1.0 - ceiling(comp, normalize)


def best_lexical_paraphrase(comp: Composition):
    """Unit vector maximizing Para (top eigenvector of S_P). Use rank_paraphrases
    to find the nearest vocabulary items."""
    return comp.top_eigen()[1]


def rank_paraphrases(comp: Composition, E, topk=10, *, normalize="M",
                     relative=True, exclude=None):
    """Score every row of an embedding matrix E (N, d) as a single-word paraphrase.

    Returns (indices, scores) of the top-k. relative=True divides by the ceiling
    (Para_norm). exclude: iterable of row indices to skip (e.g. the phrase's own words).
    """
    E = np.asarray(E, float)
    n = _probe_norm(comp, normalize)
    q = comp.quad(E)
    nr2 = np.einsum("nd,nd->n", E, E)
    scores = np.where(nr2 > EPS, q / (np.maximum(nr2, EPS) * max(n, EPS)), -np.inf)
    if relative:
        c = ceiling(comp, normalize)
        if c > EPS:
            scores = scores / c
    if exclude is not None:
        scores[np.asarray(list(exclude), dtype=np.intp)] = -np.inf
    topk = min(topk, len(scores))
    idx = np.argpartition(-scores, topk - 1)[:topk]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# =============================================================================
# 7. Seven-way trigram factorization (vectorized)
# =============================================================================
#
# For trigram (a,b,c) with trained embedding t, each pattern P is a nonempty
# subset of {M(a,b), M(b,c), M(a,c)} and is scored by
#     sim_P = <t(x)t, sum_P M>_F / (||t||^2 ||sum_P M||_F).
# A and B (single adjacent pair) are the trivially incomplete cases;
# C-G are the structurally informative ones.

TRIGRAM_PAIRS = ((0, 1), (1, 2), (0, 2))          # ab, bc, ac
TRIGRAM_PATTERNS = {                               # masks over (ab, bc, ac)
    "A": (1, 0, 0), "B": (0, 1, 0), "C": (0, 0, 1),
    "D": (1, 1, 0), "E": (0, 1, 1), "F": (1, 0, 1),
    "G": (1, 1, 1),
}
PATTERN_LABELS = {
    "A": "M(a,b)", "B": "M(b,c)", "C": "M(a,c)",
    "D": "M(a,b)+M(b,c)", "E": "M(b,c)+M(a,c)", "F": "M(a,b)+M(a,c)",
    "G": "M(a,b)+M(b,c)+M(a,c)",
}
INFORMATIVE = frozenset("CDEFG")


def subset_sims_batch(probe, W, pairs, masks, lam=0.0, chunk=65536):
    """Generic batched scorer: sim_F(probe(x)probe, sum_t mask_t * M(W[i_t], W[j_t]; lam_t)).

    probe : (N, d)       one probe vector per item (lifted to probe(x)probe)
    W     : (N, k, d)    k word vectors per item
    pairs : T pairs (i, j) into the k vectors
    masks : (P, T)       0/1 (or real weights) selecting terms per pattern
    lam   : scalar, (T,), or (N, T)
    returns (N, P) similarities
    """
    probe = np.atleast_2d(np.asarray(probe, float))
    W = np.asarray(W, float)
    if W.ndim == 2:
        W = W[None]
    masks = np.atleast_2d(np.asarray(masks, float))
    I = np.array([p[0] for p in pairs], dtype=np.intp)
    J = np.array([p[1] for p in pairs], dtype=np.intp)
    N, T = len(probe), len(pairs)
    lam = np.asarray(lam, float)
    if lam.ndim == 1 and lam.shape[0] != T:
        raise ValueError("1-D lam must have one entry per pair; pass (N, T) for per-item lam")
    L = np.broadcast_to(lam, (N, T))
    out = np.empty((N, masks.shape[0]))
    for lo in range(0, N, chunk):
        hi = min(N, lo + chunk)
        Wc, pc, Lc = W[lo:hi], probe[lo:hi], L[lo:hi]
        Gm = np.einsum("nkd,nld->nkl", Wc, Wc)
        tp = np.einsum("nd,nkd->nk", pc, Wc)
        num = (tp[:, I] * tp[:, J]) @ masks.T
        PR = Gm[:, I[:, None], I[None, :]]
        QS = Gm[:, J[:, None], J[None, :]]
        PS = Gm[:, I[:, None], J[None, :]]
        QR = Gm[:, J[:, None], I[None, :]]
        a, b = PR * QS, PS * QR
        K = 0.5 * (a + b) + (Lc[:, :, None] * Lc[:, None, :]) * 0.5 * (a - b)
        den2 = np.einsum("pt,ntu,pu->np", masks, K, masks)
        den = np.einsum("nd,nd->n", pc, pc)[:, None] * np.sqrt(np.clip(den2, 0.0, None))
        out[lo:hi] = np.where(den > EPS, num / np.maximum(den, EPS), 0.0)
    return out


def trigram_patterns_batch(t, a, b, c, lam_ab=0.0, lam_bc=0.0, lam_ac=0.0):
    """Seven-way scores for N trigrams. t, a, b, c: (N, d). lam_*: scalar or (N,).

    Returns (sims (N, 7), labels list 'ABCDEFG').
    """
    t, a, b, c = (np.atleast_2d(np.asarray(x, float)) for x in (t, a, b, c))
    N = len(t)
    W = np.stack([a, b, c], axis=1)
    L = np.stack([np.broadcast_to(np.asarray(x, float), (N,))
                  for x in (lam_ab, lam_bc, lam_ac)], axis=1)
    labels = list(TRIGRAM_PATTERNS)
    masks = np.array([TRIGRAM_PATTERNS[p] for p in labels], float)
    return subset_sims_batch(t, W, TRIGRAM_PAIRS, masks, L), labels


def trigram_patterns(t, a, b, c, lam_ab=0.0, lam_bc=0.0, lam_ac=0.0):
    """Single trigram: dict with per-pattern sims, best, margin, informative."""
    sims, labels = trigram_patterns_batch(t, a, b, c, lam_ab, lam_bc, lam_ac)
    row = sims[0]
    order = np.argsort(-row)
    best = labels[order[0]]
    return {
        "sims": dict(zip(labels, map(float, row))),
        "best": best,
        "margin": float(row[order[0]] - row[order[1]]),
        "informative": best in INFORMATIVE,
    }


# =============================================================================
# 8. Backward-compatible aliases for gaf_core users (non-clashing names only)
# =============================================================================
#
# Let code written against gaf_core's names run under `import gaf_engine` too.
# Only names whose contract does NOT clash with the engine are aliased here;
# M, compose and gamma_bigram keep their engine signatures (the gaf_core
# variants, with the legacy contract, live in the gaf_core.py facade).

from typing import Literal as _Literal

GammaMode = _Literal["sin2", "abs_sin", "const"]
Diagonal = _Literal["keep", "zero"]


def gamma(a, b, mode="sin2", const=1.0):
    """Geometric order-sensitivity magnitude (gaf_core semantics):
    sin^2(theta) = 1 - gamma_cos2.  mode in {'sin2','abs_sin','const'}.
    For the embedding-intrinsic cos^2 estimator use gamma_cos2."""
    if mode == "const":
        return float(const)
    s2 = 1.0 - gamma_cos2(a, b)
    if mode == "sin2":
        return float(s2)
    if mode == "abs_sin":
        return float(math.sqrt(max(s2, 0.0)))
    raise ValueError(f"unknown gamma mode: {mode!r}")


def lam_from_gamma(gamma_val, sigma=+1):
    """lam = sigma * gamma, with gaf_core argument order (gamma first)."""
    if sigma not in (-1, +1):
        raise ValueError("sigma must be -1 or +1")
    return lam_from_sigma_gamma(sigma, gamma_val)


__all__ += ["gamma", "lam_from_gamma", "GammaMode", "Diagonal"]


# =============================================================================
# 9. High-level operators under the canonical names (the former gaf_core API)
# =============================================================================
#
# M, compose and gamma_bigram carry the high-level gaf_core semantics:
# gamma is derived from the pair (geometry or corpus counts) rather than passed
# in. The explicit, matrix-free primitives are M_with_lambda, compose_simpler
# and gamma_bigram_by_freq. gaf_core.py re-exports this module verbatim, so
# `import gaf_core` and `import gaf_engine` expose identical names and behaviour.

def M(a, b, sigma=+1, gamma_mode="sin2", gamma_const=1.0, diagonal="keep"):
    """High-level operator: M = S + sigma*gamma(a,b)*G, gamma from pair geometry.

        M(a, b) = S(a, b) + sigma * gamma(a, b) * G(a, b)

    For an explicit lambda, use M_with_lambda(a, b, lam).
    """
    g = gamma(a, b, mode=gamma_mode, const=gamma_const)
    return M_with_lambda(a, b, lam_from_gamma(g, sigma), diagonal=diagonal)


def compose(vectors, pairs, sigmas=None, *, gamma_mode="sin2",
            gamma_const=1.0, diagonal="keep"):
    """High-level composition over an explicit pair set P, returned as a DENSE
    matrix with gamma derived per pair (gaf_core semantics). The choice of P
    *is* the parse; sigmas gives sigma per pair (default +1).

    For the fast matrix-free object with an explicit per-pair lambda, use
    compose_simpler(vectors, pairs, lam=...), which returns a Composition.
    """
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    P = [tuple(p) for p in pairs]
    if sigmas is None:
        sigmas = [+1] * len(P)
    if len(sigmas) != len(P):
        raise ValueError("sigmas must align with pairs")
    if not P:
        return np.zeros((V.shape[1], V.shape[1]), dtype=float)
    lam_arr = np.array(
        [lam_from_gamma(gamma(V[i], V[j], gamma_mode, gamma_const), s)
         for (i, j), s in zip(P, sigmas)], dtype=float)
    return compose_simpler(V, P, lam=lam_arr).to_dense(diagonal=diagonal)


def gamma_bigram(a, b, N, min_count=5, correct=True):
    """High-level corpus gamma: word strings + count dict + min_count gate.

        gamma(a,b) = max(0, |N_ab - N_ba| - sqrt(2N/pi)) / N,  gated by min_count.

    For raw counts, use gamma_bigram_by_freq(n_ab, n_ba, floor=...).
    """
    n_ab = N.get((a, b), 0)
    n_ba = N.get((b, a), 0)
    if n_ab + n_ba < min_count:
        return 0.0
    return gamma_bigram_by_freq(n_ab, n_ba, floor=correct)


__all__ += ["M_with_lambda", "compose_simpler", "gamma_bigram_by_freq"]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 50
    a, b, c, t = rng.standard_normal((4, d))
    print(f"gaf_engine {__version__}")
    m = Composition.pair(a, b, lam=0.6)
    print("  ||M||_F matrix-free vs dense :",
          round(m.norm(), 10), round(np.linalg.norm(M_with_lambda(a, b, 0.6)), 10))
    print("  order similarity (lam=0.6)   :", round(order_similarity(a, b, 0.6), 6))
    P = compose_simpler([a, b, c], "exhaustive", lam=0.5)
    print("  ceiling / phrasehood         :", round(ceiling(P), 4), round(phrasehood(P), 4))
    r = trigram_patterns(t, a, b, c, 0.3, 0.3, 0.3)
    print("  trigram best pattern         :", r["best"], "margin", round(r["margin"], 4))
    print("  gamma_bigram_by_freq(30, 10) :", round(gamma_bigram_by_freq(30, 10), 4))
