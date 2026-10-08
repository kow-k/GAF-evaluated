#!/usr/bin/env python3
"""Test the per-pair order-detection mechanism directly from an embedding.

For a 2-word swap a·b -> b·a under lambda=1, GAF detection has a closed form:
    detection = 1 - cos(S+G, S-G) = 2r/(1+r),   r = ||G||^2/||S||^2 = sin^2(theta)/(1+cos^2(theta))
where theta is the angle between the TWO swapped words (not the global anisotropy).
This script computes, per stimulus category in pairs_en.json, the mean angle between
adjacent content-word pairs and the implied detection, so FastText vs GloVe can be
compared on the geometry that actually drives order detection.

Usage:
    python measure_pair_angles.py cc.en.vec.gz pairs_en.json
    python measure_pair_angles.py glove.6B.300d.txt pairs_en.json
"""
import sys, gzip, json
import numpy as np

def opener(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if p.endswith(".gz") \
           else open(p, "r", encoding="utf-8", errors="ignore")

def load_vectors(path, vocab):
    vocab = set(vocab); vecs = {}; dim = None
    with opener(path) as f:
        first = f.readline().split()
        if not (len(first) == 2 and first[0].isdigit()):
            if first and first[0] in vocab:
                vecs[first[0]] = np.asarray(first[1:], dtype=float); dim = len(first)-1
        for line in f:
            p = line.rstrip("\n").split(" ")
            if len(p) < 3: continue
            if p[0] in vocab:
                try: v = np.asarray(p[1:], dtype=float)
                except ValueError: continue
                if dim is None: dim = len(v)
                if len(v) == dim: vecs[p[0]] = v
    return vecs

def r_of(cos):
    cos = max(-1.0, min(1.0, cos))
    s2 = 1 - cos*cos
    return s2 / (1 + cos*cos)          # ||G||^2/||S||^2

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    emb, pairs_path = sys.argv[1], sys.argv[2]
    data = json.load(open(pairs_path))
    cats = data.get("categories", {})
    # collect vocab
    vocab = set()
    for c in cats.values():
        for a, b in c["pairs"]:
            vocab.update(a.lower().split()); vocab.update(b.lower().split())
    V = load_vectors(emb, vocab)
    def unit(w):
        v = V.get(w)
        if v is None: return None
        n = np.linalg.norm(v)
        return v/n if n > 0 else None
    print(f"emb: {emb}   vocab found: {len(V)}/{len(vocab)}")
    print(f"{'category':<14}{'expect':<10}{'n_pairs':>8}{'<cos>':>8}{'<sin^2>':>9}{'<detect>':>10}")
    for name, c in cats.items():
        coss, dets = [], []
        for a, b in c["pairs"]:
            toks = a.lower().split()
            # adjacent content-word pairs of the original ordering
            for i in range(len(toks)-1):
                ua, ub = unit(toks[i]), unit(toks[i+1])
                if ua is None or ub is None: continue
                cos = float(np.dot(ua, ub)); coss.append(cos)
                r = r_of(cos); dets.append(2*r/(1+r))
        if coss:
            print(f"{name:<14}{c.get('expect',''):<10}{len(coss):>8}"
                  f"{np.mean(coss):>8.3f}{np.mean([1-x*x for x in coss]):>9.3f}{np.mean(dets):>10.3f}")

if __name__ == "__main__":
    main()
