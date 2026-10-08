"""
Invariant tests for gaf_engine.py.

    python test_gaf_engine.py      # plain runner, no pytest needed
    pytest test_gaf_engine.py      # also works

Each test checks either (i) matrix-free == dense reference, or (ii) a
proposition of the formal specification / a closed form from the working notes.
"""

import math
import numpy as np
import gaf_engine as gafe

RNG = np.random.default_rng(20261003)
D = 24
TOL = 1e-9


def vecs(n, d=D):
    return RNG.standard_normal((n, d))


def close(x, y, tol=TOL):
    return np.allclose(x, y, atol=tol, rtol=tol)


# ---------------------------------------------------------------- operator ---

def test_S_symmetric_G_antisymmetric():
    a, b = vecs(2)
    assert close(gafe.S(a, b), gafe.S(a, b).T)
    assert close(gafe.G(a, b), -gafe.G(a, b).T)


def test_G_diagonal_zero_M_diagonal_kept():
    a, b = vecs(2)
    assert close(np.diag(gafe.G(a, b)), 0.0)
    for lam in (-1, -0.3, 0, 0.7, 1):
        assert close(np.diag(gafe.M_with_lambda(a, b, lam)), a * b)


def test_S_orthogonal_G():
    a, b = vecs(2)
    assert abs(gafe.frobenius_inner(gafe.S(a, b), gafe.G(a, b))) < TOL


def test_convex_form():
    a, b = vecs(2)
    for kap in (0.0, 0.2, 0.5, 0.9, 1.0):
        lhs = gafe.M_with_lambda(a, b, kappa=kap)
        rhs = kap * np.outer(a, b) + (1 - kap) * np.outer(b, a)
        assert close(lhs, rhs)


def test_reversal_identity():
    a, b = vecs(2)
    for kap in (0.1, 0.35, 0.8):
        assert close(gafe.M_with_lambda(a, b, kappa=kap), gafe.M_with_lambda(b, a, kappa=1 - kap))


def test_self_composition_is_lift():
    (w,) = vecs(1)
    for lam in (-1, 0, 0.4, 1):
        assert close(gafe.M_with_lambda(w, w, lam), gafe.lift(w))


def test_legacy_zero_diagonal():
    a, b = vecs(2)
    X = gafe.M_legacy(a, b, 0.5)
    assert close(np.diag(X), 0.0)
    Y = gafe.M_with_lambda(a, b, 0.5).copy()
    np.fill_diagonal(Y, 0.0)
    assert close(X, Y)


def test_kappa_lam_roundtrip():
    k = np.linspace(0, 1, 11)
    assert close(gafe.kappa_from_lam(gafe.lam_from_kappa(k)), k)
    assert gafe.lam_from_sigma_gamma(-1, 0.4) == -0.4


# ------------------------------------------------- matrix-free vs dense ---

def _random_comp(n=4, kind="exhaustive"):
    V = vecs(n)
    lam = RNG.uniform(-1, 1, len(gafe.pair_set(n, kind)))
    return gafe.compose_simpler(V, kind, lam), V, lam


def test_to_dense_matches_reference():
    c, V, lam = _random_comp()
    ref = sum(gafe.M_with_lambda(V[i], V[j], l) for (i, j), l in zip(gafe.pair_set(4, "exhaustive"), lam))
    assert close(c.to_dense(), ref)


def test_inner_norm_cos_match_dense():
    c1, _, _ = _random_comp(4)
    c2, _, _ = _random_comp(5, "adjacent")
    D1, D2 = c1.to_dense(), c2.to_dense()
    assert close(c1.inner(c2), gafe.frobenius_inner(D1, D2))
    assert close(c1.norm(), np.linalg.norm(D1))
    assert close(c1.cos(c2), gafe.frobenius_sim(D1, D2))
    assert close(gafe.frobenius_sim(c1, c2), c1.cos(c2))


def test_S_G_split():
    c, _, _ = _random_comp()
    Dn = c.to_dense()
    assert close(c.norm("S") ** 2, np.linalg.norm(gafe.sym(Dn)) ** 2)
    assert close(c.norm("G") ** 2, np.linalg.norm(gafe.skew(Dn)) ** 2)
    assert close(c.norm() ** 2, c.norm("S") ** 2 + c.norm("G") ** 2)


def test_frobenius_decomposition_uniform_lam():
    # <M_A, M_B> = <S_A, S_B> + lam^2 <G_A, G_B> for a shared uniform lam
    lam = 0.6
    A = gafe.compose_simpler(vecs(3), "exhaustive", lam)
    B = gafe.compose_simpler(vecs(4), "adjacent", lam)
    SA, SB = A.S_part(), B.S_part()
    GA = gafe.compose_simpler(A.V, "exhaustive", 1.0).G_part()
    GB = gafe.compose_simpler(B.V, "adjacent", 1.0).G_part()
    assert close(A.inner(B), SA.inner(SB) + lam ** 2 * GA.inner(GB))


def test_addition_scaling():
    c1, _, _ = _random_comp()
    c2, _, _ = _random_comp()
    assert close((2.0 * c1 - c2).to_dense(), 2 * c1.to_dense() - c2.to_dense())
    assert close(sum([c1, c2]).to_dense(), c1.to_dense() + c2.to_dense())


def test_quad_is_G_free_and_matches_dense():
    c, _, _ = _random_comp()
    x = vecs(1)[0]
    assert close(c.quad(x), x @ c.to_dense() @ x)
    assert close(c.quad(x), c.S_part().quad(x))
    X = vecs(5)
    assert close(c.quad(X), np.einsum("nd,de,ne->n", X, c.to_dense(), X))


def test_matvec_matches_dense():
    c, _, _ = _random_comp()
    x = vecs(1)[0]
    assert close(c.matvec(x), c.to_dense() @ x)


def test_pair_sets():
    assert gafe.pair_set(4, "adjacent") == [(0, 1), (1, 2), (2, 3)]
    assert len(gafe.pair_set(5, "exhaustive")) == 10
    assert gafe.pair_set(4, "nonadjacent") == [(0, 2), (0, 3), (1, 3)]
    assert gafe.pair_set(3, "diagonal") == [(0, 0), (1, 1), (2, 2)]
    assert gafe.pair_set(4, "window:2") == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    # ngram:N == window:(N-1): n=2 is adjacent, n=1 is unigrams (no pairs),
    # and a wide enough window coincides with exhaustive.
    assert gafe.pair_set(5, "ngram:2") == gafe.pair_set(5, "adjacent")
    assert gafe.pair_set(5, "ngram:3") == gafe.pair_set(5, "window:2")
    assert gafe.pair_set(4, "ngram:1") == []
    assert gafe.pair_set(4, "ngram:4") == gafe.pair_set(4, "exhaustive")


def test_compose_words_callable_on_strings():
    vocab = {w: v for w, v in zip("abcd", vecs(4))}
    cnt = gafe.bigram_counts("a b a b a b c d".split())
    lam = lambda u, v: gafe.canonical_order(cnt, u, v) * gafe.gamma_from_counts(cnt, u, v, floor=False)
    comp, kept = gafe.compose_words(["a", "b", "zzz", "c"], vocab, "adjacent", lam)
    assert kept == ["a", "b", "c"]
    ref = gafe.M_with_lambda(vocab["a"], vocab["b"], lam("a", "b")) + gafe.M_with_lambda(vocab["b"], vocab["c"], lam("b", "c"))
    assert close(comp.to_dense(), ref)


# ------------------------------------------------------- closed forms ---

def test_pair_norms_and_order_similarity():
    a, b = vecs(2)
    s2, g2 = gafe.pair_norms(a, b)
    assert close(s2, np.linalg.norm(gafe.S(a, b)) ** 2)
    assert close(g2, np.linalg.norm(gafe.G(a, b)) ** 2)
    for lam in (-1, -0.5, 0, 0.5, 1):
        ref = gafe.frobenius_sim(gafe.M_with_lambda(a, b, lam), gafe.M_with_lambda(b, a, lam))
        assert close(gafe.order_similarity(a, b, lam), ref)
    assert close(gafe.order_similarity(a, b, 0.0), 1.0)          # averaging is order-blind
    assert close(gafe.order_similarity(a, b, 0.5), gafe.order_similarity(a, b, -0.5))  # even in lam


def test_headedness_equidistance():
    # cos(M(u,v), u(x)u) == cos(M(u,v), v(x)v) for every u, v, lam  (headedness E1)
    u, v = vecs(2)
    for lam in (-1, -0.2, 0.0, 0.7, 1):
        m = gafe.Composition.pair(u, v, lam)
        assert close(m.cos(gafe.Composition.lift(u)), m.cos(gafe.Composition.lift(v)))


# ------------------------------------------------------------- gamma ---

def test_gamma_bigram():
    assert gafe.gamma_bigram_by_freq(0, 0) == 0.0
    assert close(gafe.gamma_bigram_by_freq(30, 10, floor=False), 0.5)
    N = 40
    assert close(gafe.gamma_bigram_by_freq(30, 10), max(0, 20 - math.sqrt(2 * N / math.pi)) / N)
    assert gafe.gamma_bigram_by_freq(3, 2) == 0.0          # |diff| below null floor
    assert close(gafe.gamma_bigram_by_freq(10, 30), gafe.gamma_bigram_by_freq(30, 10))   # symmetric
    arr = gafe.gamma_bigram_by_freq(np.array([30, 0]), np.array([10, 0]))
    assert arr.shape == (2,)


def test_gamma_cos2_matches_paper6_kappa():
    a, b = vecs(2)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert close(gafe.kappa_from_lam(gafe.gamma_cos2(a, b)), (1 + cos ** 2) / 2)


# -------------------------------------------------- paraphrasability ---

def _dense_ceiling(comp, normalize="M"):
    Sd = gafe.sym(comp.to_dense())
    mu = max(np.linalg.eigvalsh(Sd).max(), 0.0)
    return mu / comp.norm(normalize)


def test_ceiling_matches_dense_eigen():
    for kind in ("adjacent", "exhaustive"):
        c, _, _ = _random_comp(4, kind)
        assert close(gafe.ceiling(c), _dense_ceiling(c))
        assert close(gafe.ceiling(c, "S"), _dense_ceiling(c, "S"))


def test_single_pair_ceiling_closed_form_and_bisector():
    a, b = vecs(2)
    for lam in (0.0, 0.6):
        m = gafe.Composition.pair(a, b, lam)
        expect = (np.linalg.norm(a) * np.linalg.norm(b) + a @ b) / (2 * m.norm())
        assert close(gafe.ceiling(m), expect)
    w = gafe.best_lexical_paraphrase(gafe.Composition.pair(a, b, 0.3))
    bis = a / np.linalg.norm(a) + b / np.linalg.norm(b)
    bis /= np.linalg.norm(bis)
    assert close(abs(w @ bis), 1.0)


def test_ceiling_is_attained_and_bounds_para():
    c, _, _ = _random_comp()
    w = gafe.best_lexical_paraphrase(c)
    assert close(gafe.paraphrasability(w, c), gafe.ceiling(c))
    for x in vecs(50):
        assert gafe.paraphrasability(x, c) <= gafe.ceiling(c) + 1e-12


def test_para_lam_dependence():
    # Numerator is S-only, but ||M_P|| grows with |lam|: Para(normalize="M") is NOT
    # lam-invariant. Para(normalize="S") and Para/ceiling ARE.
    V = vecs(3)
    w = vecs(1)[0]
    P0 = gafe.compose_simpler(V, "exhaustive", 0.0)
    P1 = gafe.compose_simpler(V, "exhaustive", 0.9)
    assert not close(gafe.paraphrasability(w, P0, "M"), gafe.paraphrasability(w, P1, "M"))
    assert close(gafe.paraphrasability(w, P0, "S"), gafe.paraphrasability(w, P1, "S"))
    assert close(gafe.paraphrase_norm(w, P0), gafe.paraphrase_norm(w, P1))
    r0 = gafe.paraphrasability(w, P0, "M") / gafe.ceiling(P0, "M")
    r1 = gafe.paraphrasability(w, P1, "M") / gafe.ceiling(P1, "M")
    assert close(r0, r1)


def test_rank_paraphrases():
    c, _, _ = _random_comp()
    E = vecs(200)
    E[17] = 3.0 * gafe.best_lexical_paraphrase(c)
    idx, sc = gafe.rank_paraphrases(c, E, topk=5)
    assert idx[0] == 17 and close(sc[0], 1.0)
    assert np.all(np.diff(sc) <= 1e-12)
    idx2, _ = gafe.rank_paraphrases(c, E, topk=5, exclude=[17])
    assert 17 not in idx2


# ------------------------------------------------ trigram factorization ---

def _scalar_pattern(t, a, b, c, lams, mask):
    terms = [(a, b, lams[0]), (b, c, lams[1]), (a, c, lams[2])]
    comp = sum(gafe.Composition.pair(x, y, l) for (x, y, l), m in zip(terms, mask) if m)
    return comp.cos(gafe.Composition.lift(t))


def test_trigram_batch_matches_composition_path():
    N = 7
    T, A, B, C = (vecs(N) for _ in range(4))
    lab = RNG.uniform(-1, 1, N)
    lbc = RNG.uniform(-1, 1, N)
    lac = RNG.uniform(-1, 1, N)
    sims, labels = gafe.trigram_patterns_batch(T, A, B, C, lab, lbc, lac)
    assert labels == list("ABCDEFG") and sims.shape == (N, 7)
    for n in range(N):
        for p, lbl in enumerate(labels):
            ref = _scalar_pattern(T[n], A[n], B[n], C[n], (lab[n], lbc[n], lac[n]),
                                  gafe.TRIGRAM_PATTERNS[lbl])
            assert close(sims[n, p], ref)


def test_trigram_single_and_informative_flag():
    t, a, b, c = vecs(4)
    r = gafe.trigram_patterns(t, a, b, c, 0.2, 0.2, 0.2)
    assert set(r["sims"]) == set("ABCDEFG")
    assert r["margin"] >= 0
    assert r["informative"] == (r["best"] in "CDEFG")


def test_trigram_chunking_invariant():
    N = 50
    T, A, B, C = (vecs(N) for _ in range(4))
    W = np.stack([A, B, C], 1)
    masks = np.array(list(gafe.TRIGRAM_PATTERNS.values()), float)
    full = gafe.subset_sims_batch(T, W, gafe.TRIGRAM_PAIRS, masks, 0.4)
    chunked = gafe.subset_sims_batch(T, W, gafe.TRIGRAM_PAIRS, masks, 0.4, chunk=7)
    assert close(full, chunked)


# -------------------------------------------- canonical (legacy) names ---

def test_canonical_names_are_gaf_core_semantics():
    a, b = vecs(2)
    g = gafe.gamma(a, b)
    assert close(gafe.M(a, b, sigma=+1), gafe.M_with_lambda(a, b, g))
    assert close(gafe.M(a, b, sigma=-1), gafe.M_with_lambda(a, b, -g))
    assert close(np.diag(gafe.M(a, b)), a * b)              # diagonal kept
    V = vecs(4)
    P = gafe.pair_set(4, "adjacent")
    dense = gafe.compose(V, P)                              # dense, not Composition
    man = sum(gafe.M(V[i], V[j]) for i, j in P)
    assert isinstance(dense, np.ndarray) and close(dense, man)
    N = {("x", "y"): 30, ("y", "x"): 10}
    assert close(gafe.gamma_bigram("x", "y", N), gafe.gamma_bigram_by_freq(30, 10))
    assert gafe.gamma_bigram("x", "y", N, min_count=100) == 0.0


def test_random_pairs():
    # count matches 'adjacent' by default; pairs are valid i<j within exhaustive
    for n in (2, 3, 5, 8):
        P = gafe.random_pairs(n, seed=0)
        assert len(P) == max(0, n - 1)
        exh = set(gafe.pair_set(n, "exhaustive"))
        assert all(i < j and (i, j) in exh for i, j in P)
        assert len(set(P)) == len(P)                       # distinct
    # reproducible by seed; variable across seeds for n >= 4
    assert gafe.random_pairs(8, seed=1) == gafe.random_pairs(8, seed=1)
    draws = {tuple(gafe.random_pairs(8, seed=s)) for s in range(20)}
    assert len(draws) > 1
    # explicit k and the n<=1 edge
    assert len(gafe.random_pairs(10, k=4, seed=0)) == 4
    assert gafe.random_pairs(1) == []
    # a shared rng advances across calls (dataset-wide control)
    r = np.random.default_rng(7)
    a1 = gafe.random_pairs(6, rng=r); a2 = gafe.random_pairs(6, rng=r)
    assert isinstance(a1, list) and isinstance(a2, list)


# ----------------------------------------------------------------- runner ---

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
