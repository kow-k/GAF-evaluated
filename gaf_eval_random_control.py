#!/usr/bin/env python3
"""
gaf_eval_random_control.py --- the count-matched random pair set as the true
null control for GAF composition.

Adjacent (n-1 pairs) and Exhaustive (C(n,2) pairs) differ in BOTH structure and
cardinality, so "Adjacent beats Exhaustive" (or the reverse) conflates the two.
The honest control for Adjacent is a pair set with the SAME number of pairs but
positions chosen at random:

    adjacent    (i, i+1)                 n-1 pairs, structured
    random      n-1 pairs ~ U(all i<j)   n-1 pairs, unstructured   <-- control
    exhaustive  all i<j                  C(n,2) pairs

Because a random pair set is one draw from many, we draw R of them and treat the
collection as a permutation null: if Adjacent's correlation with human judgments
lands above (say) the 95th percentile of the random draws, adjacency structure
carries signal beyond pair count. If Adjacent sits inside the random cloud, the
earlier Adjacent-vs-Exhaustive contrast was only ever measuring |P|.

Metric: Frobenius cosine between the two composed sentence matrices (the metric
that is sensitive to the Grassmann part), correlated with the gold score
(Spearman + Pearson). Reported overall, by content-word Jaccard layer, and by
sentence-length bin --- the Adjacent-vs-random gap is 0 for n=2 and grows with n,
so a single pooled number dilutes it.

    python gaf_eval_random_control.py -c stsb.tsv -e cc.en.300.vec.gz
    python gaf_eval_random_control.py -c SICK.txt -e glove.6B.200d.txt --sick -R 100
    python gaf_eval_random_control.py -c data.csv --col-a s1 --col-b s2 --col-score y \
        --spacy --with-dep            # adds a dependency-arc condition

Depends only on numpy, scipy, gaf_engine; spaCy optional (--spacy / --with-dep).

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import gaf_engine as ge

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:                                   # pragma: no cover
    spearmanr = pearsonr = None

# Pair sets are built here, not imported, so the tester runs against any recent
# gaf_engine (it needs only the stable algebra: compose_simpler, gamma, etc.).

def ps_adjacent(n):
    return [(i, i + 1) for i in range(n - 1)]


def ps_exhaustive(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def ps_window(n, m):
    """All pairs inside a sliding m-gram window: gap <= m-1 (m=2 is adjacent)."""
    K = max(0, m - 1)
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + K + 1))]


def ps_windowed(n, size, overlap):
    """Union of within-window pairs for windows of `size` words tiled at stride
    (size - overlap); consecutive windows share `overlap` words at their edges.

        size=2, overlap=1 -> adjacent pairs
        size=3, overlap=2 -> dense trigram window (== ps_window(n,3), gap<=2)
        size=3, overlap=1 -> trigram tiling with fewer pairs (stride 2)
        overlap=0          -> non-overlapping windows (a partition)

    overlap must be in [0, size-1].
    """
    if size < 2 or n < 2:
        return []
    if not (0 <= overlap < size):
        raise ValueError(f"overlap must be in [0, size-1]; got size={size}, overlap={overlap}")
    stride = size - overlap
    pairs, start = set(), 0
    while start < n - 1:
        end = min(start + size, n)
        for i in range(start, end):
            for j in range(i + 1, end):
                pairs.add((i, j))
        if end >= n:
            break
        start += stride
    return sorted(pairs)


def add_strand(pairs, n, mode="self", branch="right"):
    """Escape hatch for tokens a pair set leaves uncovered (orphans). Three modes:
      'self'     -> add a self-pair (i,i) = w(x)w (S-only lift, G(w,w)=0, no order
                    claim) for every uncovered token, so it still contributes content;
      'drop'     -> leave orphans out (the pair set may be empty);
      'neighbor' -> attach each orphan to its adjacent token in the branching
                    direction: right-branching (English) pairs i with i+1 (leftmost
                    terminal of the phrase on its right), left-branching (Japanese)
                    pairs i with i-1 (rightmost terminal of the phrase on its left),
                    falling back to the other side at a sentence edge and to a self-
                    pair only for a singleton. This is the head-blind, linguistically
                    motivated replacement for self-pairing; at a phrase seam it
                    reduces to directional adjacency.
    Adjacent/exhaustive/n-gram cover every token already, so this affects only
    constituency, disjoint windows and random. Pairs are oriented left-to-right."""
    if mode == "drop":
        return list(pairs)
    cov = set()
    for a, b in pairs:
        cov.add(a); cov.add(b)
    uncovered = [i for i in range(n) if i not in cov]
    if mode == "self":
        return list(pairs) + [(i, i) for i in uncovered]
    if mode == "neighbor":
        extra = []
        for i in uncovered:
            if branch == "left":
                j = i - 1 if i - 1 >= 0 else i + 1
            else:  # right (default)
                j = i + 1 if i + 1 < n else i - 1
            if j < 0 or j >= n or j == i:
                extra.append((i, i))                 # singleton fallback
            else:
                extra.append((i, j) if i < j else (j, i))
        return list(pairs) + extra
    return list(pairs)


def ps_random(n, rng, k=None):
    """k random distinct i<j pairs (default k = n-1, matching adjacent)."""
    if n < 2:
        return []
    allp = ps_exhaustive(n)
    k = (n - 1) if k is None else min(k, len(allp))
    idx = rng.choice(len(allp), size=min(k, len(allp)), replace=False)
    return [allp[i] for i in sorted(int(x) for x in idx)]


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)

STOP = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "am", "it", "its", "he", "she", "they", "we", "you", "i", "this",
    "that", "these", "those", "there", "here", "his", "her", "their", "our",
    "your", "my", "him", "them", "us", "me", "not", "no", "so", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could", "may", "might",
    "s", "t",
}


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------

def simple_tokens(text):
    return [t.lower() for t in _WORD.findall(text or "")]


def content_words(tokens):
    return {t for t in tokens if t not in STOP and len(t) > 2}


def jaccard(a, b):
    sa, sb = content_words(a), content_words(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


# ---------------------------------------------------------------------------
# data + embeddings
# ---------------------------------------------------------------------------

def _opener(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") \
        if str(path).endswith(".gz") else open(path, encoding="utf-8", errors="ignore")


def load_dataset(path, col_a=None, col_b=None, col_score=None, sick=False,
                 delim="auto", no_header=False):
    """Return a list of (sent_a, sent_b, gold_float). Format-aware:

    * delimiter: `delim` in {auto,tab,comma} (auto = tab if the first line has at
      least as many tabs as commas, else comma; .csv therefore loads as comma).
    * columns: `--col-a/-b/-score` each accept a HEADER NAME or a 0-based INTEGER
      index (negatives allowed, Python-style). Mixing is fine. With no `--col-*`,
      SICK / STS-B / GLUE headers are autodetected by name.
    * header: present by default (first row skipped). Pass `no_header=True` for a
      headerless file; then columns must be indices, and with none given the
      classic STS-benchmark layout (score@4, sentence1@5, sentence2@6) is assumed."""
    with _opener(path) as f:
        head = f.readline()
    if delim == "tab":
        d = "\t"
    elif delim == "comma":
        d = ","
    else:
        d = "\t" if head.count("\t") >= head.count(",") else ","
    fields0 = [h.strip() for h in head.rstrip("\n").split(d)]
    low = [h.lower() for h in fields0]

    def pick(cands):
        for c in cands:
            if c in low:
                return low.index(c)
        return None

    def as_idx(spec):
        """Resolve a --col-* spec: an int string -> index; else a header name."""
        if spec is None:
            return None
        s = str(spec).strip()
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return low.index(s.lower()) if s.lower() in low else None

    header = not no_header
    if col_a is not None and col_b is not None and col_score is not None:
        ia, ib, isc = as_idx(col_a), as_idx(col_b), as_idx(col_score)
        if None in (ia, ib, isc):
            raise SystemExit(f"could not resolve --col-* in header {fields0}\n"
                             f"  (got a-> {ia}, b-> {ib}, score-> {isc}); use a name "
                             f"present in the header or a 0-based integer index")
    elif no_header:                                  # headerless, no explicit cols
        ia, ib, isc = 5, 6, 4                         # standard STS-benchmark layout
    elif sick or "relatedness_score" in low or "sentence_a" in low:
        ia = pick(["sentence_a", "sentence_a_ja", "sentence1"])
        ib = pick(["sentence_b", "sentence_b_ja", "sentence2"])
        isc = pick(["relatedness_score", "relatedness_score_ja", "score"])
    else:                                            # STS-B / GLUE style
        ia = pick(["sentence1", "sentence_1", "s1", "sentencea"])
        ib = pick(["sentence2", "sentence_2", "s2", "sentenceb"])
        isc = pick(["score", "gold", "similarity", "label", "y"])
    if ia is None or ib is None or isc is None:
        raise SystemExit(f"could not find sentence/score columns in header: {fields0}\n"
                         f"  pass --col-a --col-b --col-score (name or 0-based index), "
                         f"and --no-header if the file has no header row")

    rows = []
    with _opener(path) as f:
        r = csv.reader(f, delimiter=d)
        if header:
            next(r, None)                            # skip header row
        for parts in r:
            if not parts or max(ia, ib, isc) >= len(parts) or min(ia, ib, isc) < -len(parts):
                continue
            try:
                y = float(parts[isc])
            except ValueError:
                continue
            rows.append((parts[ia], parts[ib], y))
    if not rows:
        raise SystemExit("no usable rows parsed (check --delim / --col-* / --no-header)")
    return rows


def _load_gensim(path, vocab):
    """Load the needed vocab from a gensim model / KeyedVectors (native .save() .bin,
    or a word2vec-format binary). Returns (vecs, dim). For a FastText model this uses
    the stored in-vocabulary word vectors (subword contributions already summed)."""
    try:
        from gensim.models import KeyedVectors, Word2Vec, FastText
    except ImportError:
        raise SystemExit("reading a gensim model needs gensim: `pip install gensim` "
                         "(or convert once with gensim_to_vec.py).")
    def wv(o):
        return getattr(o, "wv", o)       # model-or-KV -> KeyedVectors
    kv, errs = None, []
    for desc, fn in (
        ("KeyedVectors.load", lambda: wv(KeyedVectors.load(path, mmap="r"))),
        ("Word2Vec.load",     lambda: Word2Vec.load(path).wv),
        ("FastText.load",     lambda: FastText.load(path).wv),
        ("word2vec-bin",      lambda: KeyedVectors.load_word2vec_format(path, binary=True)),
    ):
        try:
            kv = fn(); break
        except Exception as e:
            errs.append(f"{desc}: {e}")
    if kv is None:
        raise SystemExit(f"could not load {path} via gensim:\n  " + "\n  ".join(errs))
    k2i = kv.key_to_index
    vecs = {w: np.asarray(kv[w], dtype=float) for w in vocab if w in k2i}
    return vecs, int(kv.vector_size)


def load_vectors(path, vocab):
    """Load only the needed vocab from a word-vector file. Text formats
    (.vec/.vec.gz/.txt, word2vec header optional) are read directly; gensim native
    saves (.bin/.model/.kv) are read via gensim (smaller on disk). A text file that
    yields no overlap is retried as a gensim file."""
    vocab = set(vocab)
    if path.lower().endswith((".bin", ".model", ".kv")):
        vecs, dim = _load_gensim(path, vocab)
        if not vecs:
            raise SystemExit(f"no overlap between dataset vocabulary and {path}")
        return vecs, dim
    vecs, dim = {}, None
    with _opener(path) as f:
        first = f.readline().split()
        if len(first) == 2 and first[0].isdigit():   # header line (word2vec/.vec)
            pass
        else:
            w = first[0]
            if w in vocab:
                vecs[w] = np.asarray(first[1:], dtype=float)
                dim = len(first) - 1
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
        try:                                           # maybe a gensim file, odd extension
            gv, gd = _load_gensim(path, vocab)
        except Exception:
            gv = {}
        if gv:
            return gv, gd
        raise SystemExit(f"no overlap between dataset vocabulary and {path}")
    return vecs, dim


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def lam_callable(V, mode, const):
    """Per-pair lambda for compose_simpler: lam(i,j) = sigma * gamma(V[i],V[j])."""
    if mode == "const":
        return float(const)                          # scalar: global lambda
    if mode == "geom":
        return lambda i, j: ge.lam_from_gamma(ge.gamma(V[i], V[j], "sin2"), +1)
    raise ValueError(f"unknown lam mode: {mode!r}")


def sent_matrix(V, pairs, lam_mode, lam_const):
    if not pairs:
        return None
    lam = lam_callable(V, lam_mode, lam_const)
    return ge.compose_simpler(V, pairs, lam=lam)


def sim(cA, cB):
    if cA is None or cB is None:
        return None
    return cA.cos(cB)


# ---------------------------------------------------------------------------
# dependency pairs (optional)
# ---------------------------------------------------------------------------

def dep_pairs_for(doc, keep_index):
    """Dependency-arc pairs (head, child) remapped to in-vocab token positions,
    oriented left-to-right. keep_index maps spaCy token i -> kept position."""
    P = []
    for tok in doc:
        h = tok.head.i
        if tok.i == h:
            continue
        a, b = keep_index.get(h), keep_index.get(tok.i)
        if a is None or b is None or a == b:
            continue
        P.append((a, b) if a < b else (b, a))
    return sorted(set(P))


def _spans_benepar(sent):
    """Constituent spans (doc token.i coords) from a benepar parse of one spaCy sent."""
    try:
        return [(c.start, c.end) for c in sent._.constituents]
    except Exception:
        return []


class StanzaConstSpans:
    """get_spans(sent) backed by stanza's constituency parser (pure PyTorch --- no
    transformers / tokenizers / Rust). spaCy's own tokens are fed to stanza
    PRETOKENIZED, so the tree's leaves align 1:1 with spaCy token positions; returned
    spans are in doc token.i coords, a drop-in for _spans_benepar.

    Costly bit is the neural parse. Two mitigations: (1) .prime(sents) parses every
    not-yet-seen sentence in ONE batched call instead of one call per sentence; (2) a
    disk cache keyed by the sentence's surface-token tuple (spans are
    position-independent, 0-based) so reruns --- different lam modes, R, bands --- never
    re-parse. Delete the cache file to force a fresh parse."""

    def __init__(self, lang="en", cache_path=None):
        self.lang = lang
        self.cache_path = cache_path
        self.cache = {}      # "w1\tw2\t..." -> [[lo,hi], ...]  (0-based, per sentence)
        self.dirty = False
        self._pipe = None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
            print(f"stanza const cache: {len(self.cache)} sentences from {cache_path}",
                  file=sys.stderr)

    def _ensure(self):
        if self._pipe is None:
            import stanza
            self._pipe = stanza.Pipeline(lang=self.lang,
                                         processors="tokenize,pos,constituency",
                                         tokenize_pretokenized=True, verbose=False)
        return self._pipe

    @staticmethod
    def _spans_of_tree(tree):
        def walk(node, leaf):
            if not node.children:                      # a word leaf
                i = leaf[0]; leaf[0] += 1
                return i, i + 1, []
            lo = hi = None; out = []
            for ch in node.children:
                cs, ce, sub = walk(ch, leaf)
                out.extend(sub)
                lo = cs if lo is None else min(lo, cs)
                hi = ce if hi is None else max(hi, ce)
            out.append((lo, hi))
            return lo, hi, out
        _, _, spans = walk(tree, [0])
        return spans

    @staticmethod
    def _key(words):
        return "\t".join(words)

    def prime(self, sents):
        """Parse all not-yet-cached distinct sentences in one batched stanza call."""
        todo, keys, seen = [], [], set()
        for s in sents:
            words = [t.text for t in s]
            if not words:
                continue
            k = self._key(words)
            if k in self.cache or k in seen:
                continue
            seen.add(k); keys.append(k); todo.append(words)
        if not todo:
            return
        print(f"stanza: parsing {len(todo)} new sentences...", file=sys.stderr)
        doc = self._ensure()(todo)
        for k, ss in zip(keys, doc.sentences):
            self.cache[k] = [[a, b] for (a, b) in self._spans_of_tree(ss.constituency)]
            self.dirty = True

    def __call__(self, sent):
        words = [t.text for t in sent]
        if not words:
            return []
        k = self._key(words)
        if k not in self.cache:
            self.prime([sent])
        off = sent.start
        return [(off + a, off + b) for (a, b) in self.cache[k]]

    def flush(self):
        if self.cache_path and self.dirty:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
            self.dirty = False
            print(f"stanza const cache written: {len(self.cache)} sentences -> "
                  f"{self.cache_path}", file=sys.stderr)


def const_pairs_for(doc, keep_index, max_span=None, get_spans=_spans_benepar):
    """Within-smallest-phrase pairs from a constituency parse, remapped to in-vocab
    token positions. Pair (i,j) iff their lowest common constituent is a *proper
    sub-phrase* (narrower than the whole sentence) --- i.e. they are grouped by some
    bracket below the clause root --- optionally with span <= max_span leaves.
    keep_index maps spaCy token i -> kept position; get_spans(sent) yields the
    constituent spans in doc token.i coords (benepar or stanza; see above).

    This captures within-NP (det/adj/noun) and within-VP (verb-object) grouping but,
    by constituency structure, not subject-predicate (whose lowest common node is the
    clause root) --- a deliberate contrast with dependency arcs."""
    pairs = set()
    kept = sorted(keep_index)
    for sent in doc.sents:
        spans = get_spans(sent)
        if not spans:
            continue
        root_w = sent.end - sent.start
        here = [i for i in kept if sent.start <= i < sent.end]
        for x in range(len(here)):
            for y in range(x + 1, len(here)):
                i, j = here[x], here[y]
                cont = [(e - s) for (s, e) in spans if s <= i < e and s <= j < e]
                if not cont:
                    continue
                w = min(cont)
                if w < root_w and (max_span is None or w <= max_span):
                    a, b = keep_index[i], keep_index[j]
                    if a != b:
                        pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# correlation over the dataset for one pair-set rule
# ---------------------------------------------------------------------------

def corr(xs, ys):
    if len(xs) < 3:
        return float("nan"), float("nan")
    if spearmanr is None:
        x, y = np.asarray(xs), np.asarray(ys)
        p = float(np.corrcoef(x, y)[0, 1])
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        s = float(np.corrcoef(rx, ry)[0, 1])
        return s, p
    return float(spearmanr(xs, ys).correlation), float(pearsonr(xs, ys)[0])


def jac_ranges(edges):
    """Half-open [lo, hi) ranges from sorted edges; the last range includes 1.0."""
    out = []
    for i in range(len(edges) - 1):
        out.append((edges[i], edges[i + 1], i == len(edges) - 2))
    return out


def jac_label(lo, hi, last):
    return f"[{lo:g},{hi:g}" + ("]" if last else ")")


def jac_range_of(j, ranges):
    for lo, hi, last in ranges:
        if (lo <= j <= hi) if last else (lo <= j < hi):
            return jac_label(lo, hi, last)
    return None


def corr_masked(sim, gold, mask):
    """Spearman/Pearson over finite entries where mask is True."""
    m = mask & np.isfinite(sim) & np.isfinite(gold)
    if m.sum() < 3:
        return float("nan"), float("nan")
    return corr(sim[m], gold[m])


# Okabe-Ito colorblind-safe palette, stable condition -> colour mapping.
_PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_layered(plot_dir, sims, gold, jac, nmax, jranges, jlabels, cond_order,
                 w, min_count, dataset):
    """One panel per Jaccard range: sliding-window Spearman rho vs EXACT length n,
    one line per condition. Random (the null) is drawn dashed/grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = cond_order + ["random"]
    style = {}
    for i, c in enumerate(conds):
        if c == "random":
            style[c] = dict(color="#999999", ls="--", marker=None, lw=1.6, zorder=1)
        else:
            style[c] = dict(color=_PALETTE[i % len(_PALETTE)],
                            marker=_MARKERS[i % len(_MARKERS)], ls="-",
                            lw=1.8, ms=4, zorder=3)

    ns_all = np.arange(int(nmax.min()), int(nmax.max()) + 1)

    def curve(mask, s):
        xs, ys = [], []
        for c in ns_all:
            m = mask & (np.abs(nmax - c) <= w) & np.isfinite(s) & np.isfinite(gold)
            if m.sum() >= min_count:
                rho = corr(s[m], gold[m])[0]
                if not math.isnan(rho):
                    xs.append(c); ys.append(rho)
        return xs, ys

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    npan = len(jlabels)
    ncol = 2 if npan > 1 else 1
    nrow = (npan + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.2 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for p, lab in enumerate(jlabels):
        ax = axes[p // ncol][p % ncol]
        rmask = np.array([jac_range_of(j, jranges) == lab for j in jac])
        for c in conds:
            xs, ys = curve(rmask, sims[c])
            if xs:
                ax.plot(xs, ys, label=c, **style[c])
        ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
        ax.set_title(f"Jaccard {lab}   (n={int(rmask.sum())})", fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)
        if p % ncol == 0:
            ax.set_ylabel("Spearman rho vs gold")
        if p // ncol == nrow - 1:
            ax.set_xlabel("exact sentence length n (= max token count)")
    for q in range(npan, nrow * ncol):
        axes[q // ncol][q % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"GAF: rho vs exact length, layered by Jaccard  ({dataset}, "
                 f"window +/-{w}, min {min_count})", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = Path(plot_dir) / "layered_rho_by_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--corpus", required=True, help="dataset (tsv/csv[.gz])")
    ap.add_argument("-e", "--embeddings", required=True, help=".vec/.vec.gz/.txt")
    ap.add_argument("--col-a", help="sentence-A column: header name or 0-based index")
    ap.add_argument("--col-b", help="sentence-B column: header name or 0-based index")
    ap.add_argument("--col-score", help="score column: header name or 0-based index")
    ap.add_argument("--delim", choices=["auto", "tab", "comma"], default="auto",
                    help="field delimiter (auto picks tab/comma from the first line)")
    ap.add_argument("--no-header", action="store_true",
                    help="file has no header row; --col-* must be indices (default "
                         "STS-benchmark layout score@4,s1@5,s2@6 if none given)")
    ap.add_argument("--sick", action="store_true", help="force SICK column names")
    ap.add_argument("--ngrams", default="3,4",
                    help="comma list of adjacent n-gram orders to add as conditions "
                         "(pairs with gap <= n-1); n=2 is the 'adjacent' column. "
                         "'' disables. The ladder n=2..L approaches exhaustive.")
    ap.add_argument("--windows", default="",
                    help="comma list of windowed conditions 'SIZE:OVERLAP' (overlap = "
                         "shared words between consecutive windows, 0..SIZE-1). Each "
                         "adds a column labelled wSoD. E.g. '2:1'=adjacent, "
                         "'3:2'=dense trigram, '3:1'=sparse trigram, '3:0'=disjoint.")
    ap.add_argument("--window-size", type=int, default=None,
                    help="single windowed condition: window size (use with "
                         "--overlap-degree); appended to --windows")
    ap.add_argument("--overlap-degree", type=int, default=None,
                    help="single windowed condition: shared words at the edges "
                         "(0..window-size-1)")
    ap.add_argument("-R", "--n-random", type=int, default=50,
                    help="random pair-set draws forming the permutation null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-mode", choices=["geom", "const"], default="geom",
                    help="geom: lambda=sin^2(theta) per pair; const: global lambda")
    ap.add_argument("--lam-const", type=float, default=1.0)
    ap.add_argument("--jac-edges", default="0,0.25,0.5,0.75,1.0",
                    help="content-word Jaccard layer edges -> half-open ranges "
                         "[0,.25) [.25,.5) [.5,.75) [.75,1.0]")
    ap.add_argument("--dump-pairs", default=None,
                    help="write a per-pair CSV (gold, jaccard, na, nb, and each "
                         "condition's similarity) for offline analysis")
    ap.add_argument("--dump-pairset", default=None,
                    help="diagnostic: comma-separated item indices (same idx as "
                         "--dump-pairs); print each method's actual word pairs on both "
                         "sentences of those rows, then continue.")
    ap.add_argument("--plot-dir", default=None,
                    help="write layered rho-vs-exact-n plots (one panel per Jaccard "
                         "range) to this directory")
    ap.add_argument("--len-window", type=int, default=1,
                    help="plot: sliding half-width over exact length n (rho at n uses "
                         "pairs with length in [n-w, n+w])")
    ap.add_argument("--len-min", type=int, default=25,
                    help="plot: minimum pairs in a length window to plot a rho point")
    ap.add_argument("--spacy", action="store_true", help="spaCy tokens/lemmas")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--no-lemma", action="store_true",
                    help="use surface word forms (t.text.lower()) instead of lemmas "
                         "for vocab lookup and Jaccard; parses are unaffected. Use "
                         "with embeddings trained on non-lemmatized text.")
    ap.add_argument("--with-dep", action="store_true",
                    help="add a dependency-arc condition (implies --spacy)")
    ap.add_argument("--with-const", action="store_true",
                    help="add a constituency within-phrase condition via benepar "
                         "(implies --spacy; needs `pip install benepar` + model)")
    ap.add_argument("--const-parser", choices=["benepar", "stanza"], default="benepar",
                    help="constituency backend: 'benepar' (needs transformers) or "
                         "'stanza' (pure PyTorch, no transformers/Rust). Same "
                         "within-smallest-phrase pairing either way.")
    ap.add_argument("--benepar-model", default="benepar_en3")
    ap.add_argument("--stanza-lang", default="en")
    ap.add_argument("--const-cache", default="stanza_const_cache.json",
                    help="stanza only: cache parsed spans here (keyed by surface "
                         "tokens) so reruns skip parsing. Delete it to re-parse; "
                         "pass '' to disable.")
    ap.add_argument("--const-max-span", type=int, default=None,
                    help="constituency: also require the lowest common phrase to span "
                         "<= this many leaves (default: only 'narrower than the clause')")
    ap.add_argument("--strand", choices=["self", "drop", "neighbor"], default="self",
                    help="tokens left uncovered by a pair set (orphans): 'self' adds a "
                         "self-pair w(x)w (lift; S-only, no order term) so the word still "
                         "counts; 'drop' omits it; 'neighbor' attaches it to its adjacent "
                         "token in the branching direction (--branch), the head-blind "
                         "phrase-structure repair that reduces to directional adjacency. "
                         "Only affects constituency / disjoint windows / random.")
    ap.add_argument("--branch", choices=["right", "left"], default="right",
                    help="branching direction for --strand neighbor: 'right' (English: "
                         "orphan pairs with the token on its right) or 'left' (Japanese: "
                         "orphan pairs with the token on its left).")
    ap.add_argument("--max-pairs", type=int, default=None, help="cap rows (debug)")
    ap.add_argument("-o", "--output", default="random_control_results.json")
    args = ap.parse_args()

    use_spacy = args.spacy or args.with_dep or args.with_const
    nlp = None
    if use_spacy:
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["ner"])
        if args.with_const and args.const_parser == "benepar":
            try:
                import benepar  # noqa: F401  (registers the spaCy component)
            except ImportError:
                raise SystemExit("--with-const needs benepar: pip install benepar, then "
                                 "python -c \"import benepar; benepar.download('benepar_en3')\"")
            if "benepar" not in nlp.pipe_names:
                nlp.add_pipe("benepar", config={"model": args.benepar_model})

    const_spans = _spans_benepar
    if args.with_const and args.const_parser == "stanza":
        try:
            import stanza  # noqa: F401  (fail early with a clear message)
        except ImportError:
            raise SystemExit("--const-parser stanza needs stanza: pip install stanza, then "
                             "python -c \"import stanza; stanza.download('en')\"")
        const_spans = StanzaConstSpans(args.stanza_lang,
                                       cache_path=(args.const_cache or None))

    rows = load_dataset(args.corpus, args.col_a, args.col_b, args.col_score, args.sick,
                        delim=args.delim, no_header=args.no_header)
    if args.max_pairs:
        rows = rows[:args.max_pairs]
    jedges = [float(x) for x in args.jac_edges.split(",")]
    jranges = jac_ranges(jedges)
    jlabels = [jac_label(lo, hi, last) for lo, hi, last in jranges]

    # tokenize everything once; keep spaCy docs if needed for dependency
    def toks(s):
        if nlp is not None:
            d = nlp(s)
            if args.no_lemma:
                return [t.text.lower() for t in d], d
            return [t.lemma_.lower() for t in d], d
        return simple_tokens(s), None

    prepared = []
    vocab = set()
    for sa, sb, y in rows:
        ta, da = toks(sa)
        tb, db = toks(sb)
        vocab.update(ta); vocab.update(tb)
        prepared.append((ta, da, tb, db, y))

    vecs, dim = load_vectors(args.embeddings, vocab)
    print(f"loaded {len(rows)} pairs; {len(vecs)}/{len(vocab)} vocab in embeddings "
          f"(dim {dim})", file=sys.stderr)

    def kept(tokens):
        V, idx, m = [], {}, 0
        for i, w in enumerate(tokens):
            if w in vecs:
                idx[i] = m
                V.append(vecs[w]); m += 1
        return (np.asarray(V) if V else np.zeros((0, dim))), idx

    # Constituency (stanza): parse every sentence once, batched, before the loop,
    # so the per-item const_pairs_for calls are pure cache lookups.
    if args.with_const and isinstance(const_spans, StanzaConstSpans):
        all_sents = []
        for _, da, _, db, _ in prepared:
            for d in (da, db):
                if d is not None:
                    all_sents.extend(d.sents)
        const_spans.prime(all_sents)
        const_spans.flush()

    # Build, per example, the kept vectors and the fixed pair sets; record exact
    # lengths (na, nb) and Jaccard.
    items = []
    dropped = 0
    for ta, da, tb, db, y in prepared:
        Va, ia = kept(ta)
        Vb, ib = kept(tb)
        na, nb = len(Va), len(Vb)
        if na < 2 or nb < 2:
            dropped += 1
            continue
        rec = {
            "Va": Va, "Vb": Vb, "na": na, "nb": nb, "y": y,
            "jac": jaccard(ta, tb),
            "adj": (ps_adjacent(na), ps_adjacent(nb)),
            "exh": (ps_exhaustive(na), ps_exhaustive(nb)),
            "wa": [ta[i] for i in sorted(ia)],    # kept tokens, in kept-position order
            "wb": [tb[i] for i in sorted(ib)],
        }
        if args.with_dep and da is not None and db is not None:
            rec["dep"] = (dep_pairs_for(da, ia), dep_pairs_for(db, ib))
        if args.with_const and da is not None and db is not None:
            rec["const"] = (const_pairs_for(da, ia, args.const_max_span, const_spans),
                            const_pairs_for(db, ib, args.const_max_span, const_spans))
        items.append(rec)
    N = len(items)
    print(f"kept {N} pairs ({dropped} dropped: <2 in-vocab tokens)", file=sys.stderr)

    ngrams = [int(x) for x in args.ngrams.split(",") if x.strip()] if args.ngrams else []

    # windowed conditions: list of (size, overlap)
    windows = []
    for spec in (s for s in args.windows.split(",") if s.strip()):
        W, D = (int(x) for x in spec.split(":"))
        windows.append((W, D))
    if args.window_size is not None or args.overlap_degree is not None:
        if args.window_size is None or args.overlap_degree is None:
            raise SystemExit("--window-size and --overlap-degree must be given together")
        windows.append((args.window_size, args.overlap_degree))
    for W, D in windows:
        if not (W >= 2 and 0 <= D < W):
            raise SystemExit(f"bad window spec {W}:{D} (need size>=2, 0<=overlap<size)")

    # ---- diagnostic: dump the actual word pairs each method forms ----
    if args.dump_pairset:
        want = {int(x) for x in args.dump_pairset.split(",") if x.strip() != ""}

        def _fmt(P, words):
            return "  ".join(f"{words[a]}–{words[b]}" for a, b in P) if P else "(none)"

        def _render(words, dep, const):
            n = len(words)
            rows_ = [("adjacent", ps_adjacent(n)), ("ngram:3", ps_window(n, 3)),
                     ("exhaustive", ps_exhaustive(n))]
            if dep is not None:
                rows_.append(("dependency", dep))
            if const is not None:
                rows_.append(("constituency", const))
            out = ["    tokens: " + " ".join(f"{k}:{w}" for k, w in enumerate(words))]
            for name, P in rows_:
                out.append(f"      {name:<12}: {_fmt(P, words)}")
            return "\n".join(out)

        for i, rec in enumerate(items):
            if i not in want:
                continue
            dep = rec.get("dep"); const = rec.get("const")
            print(f"\n=== item {i}  gold={rec['y']:.2f}  jaccard={rec['jac']:.3f} ===")
            print("  A:")
            print(_render(rec["wa"], dep[0] if dep else None, const[0] if const else None))
            print("  B:")
            print(_render(rec["wb"], dep[1] if dep else None, const[1] if const else None))
        print()

    # ---- per-pair design vectors (exact, no binning) ----
    gold = np.array([r["y"] for r in items], dtype=float)
    jac = np.array([r["jac"] for r in items], dtype=float)
    na_arr = np.array([r["na"] for r in items])
    nb_arr = np.array([r["nb"] for r in items])
    nmax = np.maximum(na_arr, nb_arr)          # sentence-pair length, exact

    def pair_rule(rec, kind, m=None):
        if kind == "adjacent":
            return rec["adj"]
        if kind == "exhaustive":
            return rec["exh"]
        if kind == "dependency":
            return rec.get("dep", ([], []))
        if kind == "constituency":
            return rec.get("const", ([], []))
        if kind == "ngram":
            return ps_window(rec["na"], m), ps_window(rec["nb"], m)
        if kind == "window":
            W, D = m
            return ps_windowed(rec["na"], W, D), ps_windowed(rec["nb"], W, D)
        raise ValueError(kind)

    def det_sims(kind, m=None):
        out = np.full(N, np.nan)
        for i, rec in enumerate(items):
            pa, pb = pair_rule(rec, kind, m)
            pa = add_strand(pa, rec["na"], args.strand, args.branch)
            pb = add_strand(pb, rec["nb"], args.strand, args.branch)
            s = sim(sent_matrix(rec["Va"], pa, args.lam_mode, args.lam_const),
                    sent_matrix(rec["Vb"], pb, args.lam_mode, args.lam_const))
            if s is not None:
                out[i] = s
        return out

    def avg_sims():
        out = np.full(N, np.nan)
        for i, rec in enumerate(items):
            ma, mb = rec["Va"].mean(0), rec["Vb"].mean(0)
            da, db = np.linalg.norm(ma), np.linalg.norm(mb)
            if da > 1e-12 and db > 1e-12:
                out[i] = float(ma @ mb / (da * db))
        return out

    # deterministic conditions -> per-pair similarity columns
    win_labels = [f"w{W}o{D}" for W, D in windows]
    cond_order = (["averaging", "adjacent"] + [f"ng{m}" for m in ngrams]
                  + win_labels + ["exhaustive"]
                  + (["dependency"] if args.with_dep else [])
                  + (["constituency"] if args.with_const else []))
    sims = {"averaging": avg_sims(), "adjacent": det_sims("adjacent"),
            "exhaustive": det_sims("exhaustive")}
    for m in ngrams:
        sims[f"ng{m}"] = det_sims("ngram", m)
    for (W, D), lab in zip(windows, win_labels):
        sims[lab] = det_sims("window", (W, D))
    if args.with_dep:
        sims["dependency"] = det_sims("dependency")
    if args.with_const:
        sims["constituency"] = det_sims("constituency")

    # random permutation null: R draws, per-pair sims kept for the per-pair mean
    rand_sims = np.full((args.n_random, N), np.nan)
    for rdx in range(args.n_random):
        rng = np.random.default_rng(args.seed + rdx)
        for i, rec in enumerate(items):
            pa = add_strand(ps_random(rec["na"], rng), rec["na"], args.strand, args.branch)
            pb = add_strand(ps_random(rec["nb"], rng), rec["nb"], args.strand, args.branch)
            s = sim(sent_matrix(rec["Va"], pa, args.lam_mode, args.lam_const),
                    sent_matrix(rec["Vb"], pb, args.lam_mode, args.lam_const))
            if s is not None:
                rand_sims[rdx, i] = s
    sims["random"] = np.nanmean(rand_sims, axis=0)

    # ---- strata: overall + the Jaccard ranges (exact n handled in the plots) ----
    masks = {"overall": np.ones(N, dtype=bool)}
    for lab in jlabels:
        masks[lab] = np.array([jac_range_of(j, jranges) == lab for j in jac])
    strata = ["overall"] + jlabels

    results = {c: {k: corr_masked(sims[c], gold, masks[k]) for k in strata}
               for c in cond_order}
    counts = {k: int(masks[k].sum()) for k in strata}

    # random null per stratum (per-draw correlations -> mean/percentiles, vs adjacent)
    rand_summary = {}
    for k in strata:
        mk = masks[k]
        sp = np.array([corr_masked(rand_sims[r], gold, mk)[0] for r in range(args.n_random)])
        sp = sp[np.isfinite(sp)]
        adj_sp = results["adjacent"][k][0]
        if sp.size and not math.isnan(adj_sp):
            p_perm = (np.sum(sp >= adj_sp) + 1) / (sp.size + 1)
            z = (adj_sp - sp.mean()) / sp.std() if sp.std() > 1e-12 else float("nan")
        else:
            p_perm, z = float("nan"), float("nan")
        rand_summary[k] = {
            "spearman_mean": float(sp.mean()) if sp.size else float("nan"),
            "spearman_std": float(sp.std()) if sp.size else float("nan"),
            "spearman_p05": float(np.percentile(sp, 5)) if sp.size else float("nan"),
            "spearman_p95": float(np.percentile(sp, 95)) if sp.size else float("nan"),
            "adjacent_vs_random_z": float(z),
            "adjacent_ge_random_p": float(p_perm),
        }

    # ---- report ----
    print("\n" + "=" * 78)
    print("GAF composition, layered by content-word Jaccard (exact n in the plots)")
    print(f"  dataset={Path(args.corpus).name}  embeddings={Path(args.embeddings).name}"
          f"  R={args.n_random}  lambda={args.lam_mode}")
    print("=" * 78)
    cols = cond_order
    header = f"{'stratum':<12}{'n':>6}  " + "".join(f"{c[:9]:>11}" for c in cols)
    header += f"{'rand':>9}{'rand95':>9}{'z':>7}{'p':>7}"
    print(header); print("-" * len(header))
    for k in strata:
        line = f"{k:<12}{counts[k]:>6}  "
        for c in cols:
            v = results[c][k][0]
            line += f"{v:>11.3f}" if not math.isnan(v) else f"{'--':>11}"
        rs = rand_summary[k]
        line += f"{rs['spearman_mean']:>9.3f}{rs['spearman_p95']:>9.3f}"
        line += f"{rs['adjacent_vs_random_z']:>7.1f}{rs['adjacent_ge_random_p']:>7.3f}"
        print(line)
    print("\n(Spearman rho vs gold; rand/rand95 = count-matched random null mean/p95;\n"
          " z>0 & p<0.05 => adjacent beats the random null in that Jaccard range.)")

    # ---- per-pair dump ----
    if args.dump_pairs:
        Path(args.dump_pairs).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_pairs, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gold", "jaccard", "na", "nb", "nmax"] + cond_order
                       + ["random"])
            for i in range(N):
                w.writerow([i, f"{gold[i]:.6g}", f"{jac[i]:.6g}",
                            na_arr[i], nb_arr[i], nmax[i]]
                           + [f"{sims[c][i]:.6g}" if np.isfinite(sims[c][i]) else ""
                              for c in cond_order]
                           + [f"{sims['random'][i]:.6g}"
                              if np.isfinite(sims['random'][i]) else ""])
        print(f"wrote {args.dump_pairs} ({N} rows)")

    # ---- layered rho-vs-exact-n plots (one panel per Jaccard range) ----
    if args.plot_dir:
        plot_layered(args.plot_dir, sims, gold, jac, nmax, jranges, jlabels,
                     cond_order, args.len_window, args.len_min,
                     Path(args.corpus).name)

    # ---- JSON ----
    payload = {
        "dataset": args.corpus, "embeddings": args.embeddings,
        "n_pairs": N, "dropped": dropped, "R": args.n_random,
        "lam_mode": args.lam_mode, "lam_const": args.lam_const,
        "jac_edges": jedges,
        "counts": counts,
        **{c: {k: results[c][k] for k in strata} for c in cond_order},
        "random_null": {k: rand_summary[k] for k in strata},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
