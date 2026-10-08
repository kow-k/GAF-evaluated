#!/usr/bin/env python3
"""
gaf_eval_order_discrimination.py --- does GAF composition detect word order,
and does its sensitivity track meaning, where averaging is blind by construction?

A graded-similarity corpus (STS, SICK) can't show this: order only changes meaning
when the two sentences share their words (same bag, different arrangement), and
that is exactly where a similarity score saturates. So we test order directly, on
minimal pairs that ARE reorderings of each other, with a known label:

    expect "different"  meaning-changing reordering   -> method SHOULD give LOW sim
                        (subject/object swap, scramble)
    expect "same"       meaning-preserving reordering  -> method SHOULD give HIGH sim
                        (coordination order: "salt and pepper" / "pepper and salt")

Averaging maps either reordering to an IDENTICAL mean vector, so sim == 1.0 for
both: it cannot detect order at all, and cannot separate the two classes
(AUC = 0.5). The questions for GAF are two, in increasing difficulty:

    1. ORDER DETECTION   does a reordering move the representation at all?
                         (mean 1 - sim on reordering pairs; averaging ~ 0)
    2. MEANING TRACKING  is the movement LARGER for meaning-changing reorderings
                         than for meaning-preserving ones?  (AUC of 1 - sim,
                         different vs same; 0.5 = chance)

(1) is nearly free for GAF (any swap moves the Grassmann part). (2) is the hard
one and is where a pair-specific magnitude sigma*gamma(A,B) would have to earn its
keep over a uniform lambda.

    python gaf_eval_order_discrimination.py --pairs pairs_en.json -e cc.en.vec.gz
    python gaf_eval_order_discrimination.py -e cc.en.vec.gz          # built-in demo

pairs JSON (your pairs_en.json layout is accepted):
    {"categories": {"order":   {"expect": "different", "pairs": [["a b c","c b a"], ...]},
                    "coord":   {"expect": "same",      "pairs": [["x and y","y and x"]]}}}

Depends on numpy + gaf_engine (scipy optional).

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import gaf_engine as ge

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)

# Built-in demo so the script runs with no stimulus file. Replace with --pairs.
DEMO = {
    "categories": {
        "svo_swap": {"expect": "different", "pairs": [
            ["the dog chased the cat", "the cat chased the dog"],
            ["the man hit the ball", "the ball hit the man"],
            ["john loves mary", "mary loves john"],
            ["the wolf ate the sheep", "the sheep ate the wolf"],
            ["the teacher praised the student", "the student praised the teacher"],
            ["the cat killed the rat", "the rat killed the cat"],
        ]},
        "coordination": {"expect": "same", "pairs": [
            ["salt and pepper", "pepper and salt"],
            ["black and white", "white and black"],
            ["bread and butter", "butter and bread"],
            ["cats and dogs", "dogs and cats"],
            ["knife and fork", "fork and knife"],
            ["men and women", "women and men"],
        ]},
    }
}


# ---------------------------------------------------------------------------
# pair sets + composition (built here; engine used only for the algebra)
# ---------------------------------------------------------------------------

def ps_adjacent(n):
    return [(i, i + 1) for i in range(n - 1)]


def ps_exhaustive(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def ps_window(n, m):
    K = max(0, m - 1)
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + K + 1))]


def _opener(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") \
        if str(path).endswith(".gz") else open(path, encoding="utf-8", errors="ignore")


def load_vectors(path, vocab):
    vocab = set(vocab)
    vecs, dim = {}, None
    with _opener(path) as f:
        first = f.readline().split()
        if not (len(first) == 2 and first[0].isdigit()):
            w = first[0]
            if w in vocab:
                vecs[w] = np.asarray(first[1:], dtype=float); dim = len(first) - 1
        for line in f:
            p = line.rstrip("\n").split(" ")
            if len(p) < 3:
                continue
            w = p[0]
            if w in vocab:
                try:
                    v = np.asarray(p[1:], dtype=float)
                except ValueError:
                    continue
                if dim is None:
                    dim = len(v)
                if len(v) == dim:
                    vecs[w] = v
    if not vecs:
        raise SystemExit(f"no overlap between stimulus vocabulary and {path}")
    return vecs, dim


def make_lam(words, V, mode, const, ctx):
    """Per-pair lambda builder. geom: sin^2(theta) (embedding geometry); const:
    fixed; bigram: gamma(a,b) from corpus directional bigram counts (word level);
    gamma-class: gamma(A,B) from the class-pair table (A,B = embedding-derived
    classes of the two words), with optional word-level backoff when a corpus/
    counts are also supplied and the word pair is attested >= min_count. sigma=+1."""
    if mode == "const":
        return float(const)
    if mode == "geom":
        return lambda i, j: ge.lam_from_gamma(ge.gamma(V[i], V[j], "sin2"), +1)
    counts = ctx.get("counts") or {}
    mc = ctx.get("min_count", 5)
    funcw = ctx.get("funcw") or set()
    if mode == "bigram":
        def f(i, j):
            a, b = words[i], words[j]
            if a in funcw or b in funcw:
                return 0.0                      # function word -> order-free
            nab, nba = counts.get((a, b), 0), counts.get((b, a), 0)
            if nab + nba < mc:
                return 0.0
            return ge.gamma_bigram_by_freq(nab, nba)
        return f
    if mode == "gamma-class":
        w2c, G = ctx["w2c"], ctx["classgamma"]
        backoff = bool(ctx.get("counts"))
        def f(i, j):
            a, b = words[i], words[j]
            if a in funcw or b in funcw:
                return 0.0                      # function word -> order-free
            if backoff:
                nab, nba = counts.get((a, b), 0), counts.get((b, a), 0)
                if nab + nba >= mc:
                    return ge.gamma_bigram_by_freq(nab, nba)   # lexical, when attested
            ca, cb = w2c.get(a), w2c.get(b)
            if ca is None or cb is None:
                return 0.0
            return float(G.get(frozenset((ca, cb)), 0.0))      # class-level fallback
        return f
    raise ValueError(f"unknown lam mode: {mode!r}")


def sent_comp(words, V, pairs, mode, const, ctx):
    if len(pairs) == 0:
        return None
    return ge.compose_simpler(V, pairs, lam=make_lam(words, V, mode, const, ctx))


def load_seeds(path):
    raw = json.load(open(path, encoding="utf-8"))
    return {c: [w.lower() for w in ws] for c, ws in raw.items()
            if not c.startswith("_") and isinstance(ws, list)}


def class_centroids(seeds, vecs):
    C = {}
    for cls, ws in seeds.items():
        vs = [vecs[w] for w in ws if w in vecs]
        if vs:
            C[cls] = np.mean(vs, axis=0)
    return C


def assign_class(w, vecs, centroids):
    if w not in vecs:
        return None
    v = vecs[w]; nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None
    best, bs = None, -2.0
    for cls, c in centroids.items():
        nc = np.linalg.norm(c)
        if nc < 1e-12:
            continue
        s = float(v @ c / (nv * nc))
        if s > bs:
            bs, best = s, cls
    return best


def load_class_gamma(path):
    """Read the class-pair gamma table -> {frozenset({A,B}): gamma}. Uses the
    unsigned 'gamma' column (magnitude of order sensitivity); class_pair is
    'A|B'. PRON kept as its own class, since the table distinguishes it."""
    G = {}
    with _opener(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            a, b = row["class_pair"].split("|")
            G[frozenset((a.strip(), b.strip()))] = float(row["gamma"])
    return G


def load_counts(corpus, counts_file, vocab):
    """Directional adjacent-bigram counts N[(a,b)] restricted to the stimulus
    vocab. From a corpus (text, one line per sentence) or a precomputed file
    (JSON {"a b": n}, or TSV 'a<TAB>b<TAB>n')."""
    from collections import Counter as _C
    N = _C()
    if counts_file:
        if counts_file.endswith(".json"):
            raw = json.load(open(counts_file, encoding="utf-8"))
            for k, v in raw.items():
                parts = k.split()
                if len(parts) == 2:
                    N[(parts[0].lower(), parts[1].lower())] += int(v)
        else:
            with _opener(counts_file) as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 3:
                        N[(p[0].lower(), p[1].lower())] += int(float(p[2]))
    elif corpus:
        V = set(vocab)
        with _opener(corpus) as f:
            for line in f:
                t = [w.lower() for w in _WORD.findall(line)]
                for a, b in zip(t, t[1:]):
                    if a in V and b in V:
                        N[(a, b)] += 1
    return dict(N)


# ---------------------------------------------------------------------------
# stimulus loading (accepts your pairs_en.json layout or the demo)
# ---------------------------------------------------------------------------

def _pair_strings(item):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    if isinstance(item, dict):
        for a, b in (("a", "b"), ("s1", "s2"), ("fwd", "rev"), ("src", "tgt")):
            if a in item and b in item:
                x, y = item[a], item[b]
                return (" ".join(x) if isinstance(x, list) else str(x),
                        " ".join(y) if isinstance(y, list) else str(y))
    raise ValueError(f"unrecognised pair item: {item!r}")


_SAME = {"same", "similar", "equivalent", "paraphrase", "equal", "preserved"}
_DIFF = {"different", "diff", "changed", "contrast", "distinct"}


def _norm_expect(e):
    e = str(e).lower().strip()
    if e in _SAME:
        return "same"
    if e in _DIFF:
        return "different"
    return None


def load_pairs(path):
    """Return [(s1, s2, expect, category)]. expect normalised to {'same','different'};
    'similar' (your voice category) maps to 'same'."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", data)
    out = []
    for cat, blk in cats.items():
        if cat.startswith("_") or not isinstance(blk, dict) or "pairs" not in blk:
            continue
        expect = _norm_expect(blk.get("expect", "different"))
        if expect is None:
            continue
        for item in blk.get("pairs", []):
            exp = expect
            if isinstance(item, dict) and "expect" in item:
                exp = _norm_expect(item["expect"]) or expect
            try:
                s1, s2 = _pair_strings(item)
            except ValueError:
                continue
            out.append((s1, s2, exp, cat))
    if not out:
        raise SystemExit("no labelled pairs found")
    return out


def tok(s):
    return [t.lower() for t in _WORD.findall(s or "")]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def auc(scores, labels):
    """AUC with 1 = positive (expect 'different'); score high = more 'different'."""
    pos = [s for s, l in zip(scores, labels) if l == 1 and np.isfinite(s)]
    neg = [s for s, l in zip(scores, labels) if l == 0 and np.isfinite(s)]
    if not pos or not neg:
        return float("nan")
    c = 0.0
    for p in pos:
        for n in neg:
            c += 1.0 if p > n else 0.5 if p == n else 0.0
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--embeddings", required=True)
    ap.add_argument("--pairs", default=None, help="stimulus JSON (default: built-in demo)")
    ap.add_argument("--ngrams", default="", help="extra n-gram window conditions, e.g. 3,4")
    ap.add_argument("--lam-mode", choices=["geom", "const", "bigram", "gamma-class"],
                    default="geom",
                    help="geom: sin^2(theta) from embeddings; const: fixed lambda; "
                         "bigram: word-level gamma(a,b) from corpus bigram counts "
                         "(needs --corpus/--counts); gamma-class: class-level "
                         "gamma(A,B) from --class-gamma + --seeds, with optional "
                         "word-level backoff when --corpus/--counts is also given")
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--corpus", default=None,
                    help="text corpus (one sentence per line) to count directional "
                         "adjacent bigrams over the stimulus vocabulary")
    ap.add_argument("--counts", default=None,
                    help="precomputed counts (JSON {'a b': n} or TSV a<TAB>b<TAB>n)")
    ap.add_argument("--bigram-min-count", type=int, default=5,
                    help="min a+b adjacency total to trust a word-level gamma "
                         "(bigram mode, and the backoff threshold in gamma-class)")
    ap.add_argument("--function-words", default="",
                    help="comma list of words treated as order-free (gamma=0 on any "
                         "pair that includes one), e.g. 'and,or,but' for conjunctions "
                         "that have no seed class")
    ap.add_argument("--seeds", default=None,
                    help="gamma-class mode: seed words per class (JSON), for "
                         "nearest-centroid class assignment")
    ap.add_argument("--class-gamma", default=None,
                    help="gamma-class mode: class-pair gamma table (TSV with columns "
                         "class_pair 'A|B' and gamma)")
    ap.add_argument("--reorder-only", action="store_true",
                    help="keep only exact reorderings (identical token multiset); "
                         "drops word-adding transforms (voice, negation) so the AUC "
                         "is a clean order test. Match length/swap type across the "
                         "'same' and 'different' sets for AUC to mean 'tracks meaning'.")
    ap.add_argument("-o", "--output", default="order_discrimination_results.json")
    args = ap.parse_args()

    if args.pairs:
        pairs = load_pairs(args.pairs)
        src = args.pairs
    else:
        pairs = []
        for cat, blk in DEMO["categories"].items():
            for it in blk["pairs"]:
                pairs.append((it[0], it[1], blk["expect"], cat))
        src = "built-in demo"
        print("no --pairs given; using the built-in demo stimulus set", file=sys.stderr)

    ngrams = [int(x) for x in args.ngrams.split(",") if x.strip()] if args.ngrams else []

    # tokenise, gather vocab
    toks = [(tok(s1), tok(s2), exp, cat) for s1, s2, exp, cat in pairs]
    vocab = set()
    for ta, tb, _, _ in toks:
        vocab.update(ta); vocab.update(tb)
    stim_vocab = set(vocab)

    seeds = None
    if args.lam_mode == "gamma-class":
        if not (args.seeds and args.class_gamma):
            raise SystemExit("--lam-mode gamma-class needs --seeds and --class-gamma")
        seeds = load_seeds(args.seeds)
        for ws in seeds.values():
            vocab.update(ws)                          # seed words need embeddings too

    vecs, dim = load_vectors(args.embeddings, vocab)
    print(f"{len(pairs)} pairs from {src}; {len(stim_vocab & set(vecs))}/"
          f"{len(stim_vocab)} stimulus vocab in embeddings (dim {dim})", file=sys.stderr)

    methods = ["averaging", "adjacent"] + [f"ng{m}" for m in ngrams] + ["exhaustive"]

    # ---- build the lambda context for the chosen mode ----
    ctx = {"min_count": args.bigram_min_count,
           "funcw": {w.strip().lower()
                     for w in args.function_words.split(",") if w.strip()}}
    if ctx["funcw"]:
        print(f"function words (order-free, gamma=0): {sorted(ctx['funcw'])}",
              file=sys.stderr)
    if args.corpus or args.counts:
        ctx["counts"] = load_counts(args.corpus, args.counts, vocab)
        att = sum(1 for (a, b) in ctx["counts"]
                  if ctx["counts"].get((a, b), 0) + ctx["counts"].get((b, a), 0)
                  >= args.bigram_min_count)
        print(f"bigram counts: {len(ctx['counts'])} directed types "
              f"({att} ordered pairs >= min-count {args.bigram_min_count})",
              file=sys.stderr)
    if args.lam_mode == "bigram" and "counts" not in ctx:
        raise SystemExit("--lam-mode bigram needs --corpus or --counts")
    if args.lam_mode == "gamma-class":
        cents = class_centroids(seeds, vecs)
        w2c = {}
        for w in stim_vocab:
            c = assign_class(w, vecs, cents)
            if c is not None:
                w2c[w] = c
        ctx["w2c"] = w2c
        ctx["classgamma"] = load_class_gamma(args.class_gamma)
        by_cls = defaultdict(list)
        for w in sorted(stim_vocab):
            by_cls[w2c.get(w, "?")].append(w)
        print("class assignments (stimulus vocab, nearest seed centroid):",
              file=sys.stderr)
        for c in sorted(by_cls):
            print(f"   {c:<6} {' '.join(by_cls[c])}", file=sys.stderr)
        print(f"class-gamma: {len(ctx['classgamma'])} class-pairs; "
              f"word-level backoff = {'ON' if 'counts' in ctx else 'OFF'}",
              file=sys.stderr)

    def kept_of(tokens):
        ws = [w for w in tokens if w in vecs]
        return ws, (np.asarray([vecs[w] for w in ws]) if ws else np.zeros((0, dim)))

    def method_sim(method, wa, Va, wb, Vb):
        if method == "averaging":
            if len(Va) == 0 or len(Vb) == 0:
                return None
            ma, mb = Va.mean(0), Vb.mean(0)
            da, db = np.linalg.norm(ma), np.linalg.norm(mb)
            if da < 1e-12 or db < 1e-12:
                return None
            return float(ma @ mb / (da * db))
        if method == "adjacent":
            pa, pb = ps_adjacent(len(Va)), ps_adjacent(len(Vb))
        elif method == "exhaustive":
            pa, pb = ps_exhaustive(len(Va)), ps_exhaustive(len(Vb))
        else:                                        # ngN
            m = int(method[2:])
            pa, pb = ps_window(len(Va), m), ps_window(len(Vb), m)
        cA = sent_comp(wa, Va, pa, args.lam_mode, args.lam_const, ctx)
        cB = sent_comp(wb, Vb, pb, args.lam_mode, args.lam_const, ctx)
        if cA is None or cB is None:
            return None
        return cA.cos(cB)

    # per-pair similarity for every method, plus bookkeeping
    rows = []          # (expect, cat, is_reorder, {method: sim})
    for ta, tb, exp, cat in toks:
        wa, Va = kept_of(ta)
        wb, Vb = kept_of(tb)
        if len(Va) < 2 or len(Vb) < 2:
            continue
        is_reorder = Counter(ta) == Counter(tb)      # identical multiset of tokens
        sims = {m: method_sim(m, wa, Va, wb, Vb) for m in methods}
        rows.append((exp, cat, is_reorder, sims))

    n_reorder = sum(1 for r in rows if r[2])
    print(f"kept {len(rows)} pairs ({n_reorder} are exact reorderings; the rest "
          f"differ in tokens too)", file=sys.stderr)
    if args.reorder_only:
        rows = [r for r in rows if r[2]]
        print(f"--reorder-only: keeping {len(rows)} exact-reordering pairs",
              file=sys.stderr)

    # ---- aggregate ----
    cats = sorted({r[1] for r in rows})
    report = {"source": src, "embeddings": args.embeddings, "lam_mode": args.lam_mode,
              "n_pairs": len(rows), "n_reorder": n_reorder, "methods": methods,
              "by_category": {}, "summary": {}}

    def mean(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    # per-category mean similarity per method
    for cat in cats:
        exp = next(r[0] for r in rows if r[1] == cat)
        report["by_category"][cat] = {"expect": exp,
                                      "n": sum(1 for r in rows if r[1] == cat)}
        for m in methods:
            report["by_category"][cat][m] = mean([r[3][m] for r in rows if r[1] == cat])

    # summary per method: order detection + meaning tracking (AUC)
    diff_mask = [1 if r[0] == "different" else 0 for r in rows]
    reorder_rows = [r for r in rows if r[2]]
    for m in methods:
        same_sim = mean([r[3][m] for r in rows if r[0] == "same"])
        diff_sim = mean([r[3][m] for r in rows if r[0] == "different"])
        # order detection: how far below 1.0 reordering pairs land
        det = mean([(1.0 - r[3][m]) for r in reorder_rows if r[3][m] is not None])
        a = auc([(1.0 - r[3][m]) if r[3][m] is not None else float("nan") for r in rows],
                diff_mask)
        report["summary"][m] = {
            "sim_same": same_sim, "sim_different": diff_sim,
            "gap_same_minus_diff": (same_sim - diff_sim)
            if not (math.isnan(same_sim) or math.isnan(diff_sim)) else float("nan"),
            "order_detection_1_minus_sim_on_reorderings": det,
            "auc_different_vs_same": a,
        }

    # ---- print ----
    print("\n" + "=" * 76)
    print("Order discrimination: GAF vs averaging on reordering minimal pairs")
    print(f"  stimuli={Path(src).name if args.pairs else src}  "
          f"embeddings={Path(args.embeddings).name}  lambda={args.lam_mode}")
    print("=" * 76)
    print("\nMean similarity by category (lower = more 'different'):")
    hdr = f"{'category':<16}{'expect':>10}{'n':>4}  " + "".join(f"{m[:9]:>11}" for m in methods)
    print(hdr); print("-" * len(hdr))
    for cat in cats:
        b = report["by_category"][cat]
        line = f"{cat:<16}{b['expect']:>10}{b['n']:>4}  "
        line += "".join((f"{b[m]:>11.3f}" if not math.isnan(b[m]) else f"{'--':>11}")
                        for m in methods)
        print(line)

    print("\nSummary (order detection = mean 1-sim on exact reorderings; "
          "AUC separates\n'different' from 'same', 0.5 = blind):")
    hdr2 = (f"{'method':<12}{'sim_same':>10}{'sim_diff':>10}{'gap':>8}"
            f"{'detect':>9}{'AUC':>7}")
    print(hdr2); print("-" * len(hdr2))
    for m in methods:
        s = report["summary"][m]
        print(f"{m:<12}{s['sim_same']:>10.3f}{s['sim_different']:>10.3f}"
              f"{s['gap_same_minus_diff']:>8.3f}"
              f"{s['order_detection_1_minus_sim_on_reorderings']:>9.3f}"
              f"{s['auc_different_vs_same']:>7.3f}")
    print("\nReading: averaging should show detect~0 and AUC~0.5 (order-blind). GAF\n"
          "shows detect>0 (it registers order); AUC>0.5 means its sensitivity also\n"
          "tracks meaning (bigger movement for meaning-changing reorderings).")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
