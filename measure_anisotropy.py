#!/usr/bin/env python3
"""Measure embedding anisotropy and the mean geometric order-magnitude
gamma = sin^2(theta) that GAF's lam-mode=geom assigns to word pairs.

A more anisotropic space (high mean pairwise cosine) yields smaller sin^2 theta,
hence a smaller effective lambda, hence M ~ S (bag-like). Run on each embedding
file to see whether geom-lambda differs enough to explain the FastText/GloVe gap.

Usage:
    python measure_anisotropy.py glove.6B.300d.txt
    python measure_anisotropy.py cc.en.vec.gz
"""
import sys, gzip, random
import numpy as np

def opener(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if p.endswith(".gz") \
           else open(p, "r", encoding="utf-8", errors="ignore")

def load(path, max_words=50000):
    vecs = []
    with opener(path) as f:
        first = f.readline().split()
        # fastText .vec has a "N dim" header; GloVe .txt does not
        is_header = len(first) == 2 and all(t.isdigit() for t in first)
        if not is_header:
            parts = first
            try:
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
            except ValueError:
                pass
        for line in f:
            if len(vecs) >= max_words:
                break
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            try:
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
            except ValueError:
                continue
    V = np.vstack(vecs)
    return V

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    V = load(path)
    n, d = V.shape
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / np.clip(norms, 1e-9, None)          # unit vectors
    rng = np.random.default_rng(0)
    # mean embedding direction -> global anisotropy
    mean_dir = U.mean(axis=0)
    aniso_meanvec = float(np.linalg.norm(mean_dir))   # 0=isotropic, 1=all aligned
    # sample random pairs
    K = 200000
    i = rng.integers(0, n, K); j = rng.integers(0, n, K)
    ok = i != j
    cos = np.sum(U[i[ok]] * U[j[ok]], axis=1)
    cos = np.clip(cos, -1, 1)
    gamma = 1.0 - cos**2                          # sin^2 theta = geom lambda magnitude
    print(f"file: {path}")
    print(f"  vocab sampled: {n}, dim: {d}")
    print(f"  ||mean unit vector|| (anisotropy, 0=iso..1=aligned): {aniso_meanvec:.4f}")
    print(f"  mean pairwise cosine: {cos.mean():.4f}  (sd {cos.std():.4f})")
    print(f"  mean geom-lambda = <sin^2 theta>: {gamma.mean():.4f}  (sd {gamma.std():.4f})")
    print(f"    quartiles of geom-lambda: "
          f"{np.percentile(gamma,25):.3f} / {np.percentile(gamma,50):.3f} / {np.percentile(gamma,75):.3f}")

if __name__ == "__main__":
    main()
