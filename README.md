# GAF pair-set & word-order sensitivity

Code to reproduce the experiments in *Pair-set and word-order sensitivity in
Grassmann Algebraic Framework (GAF) composition* (Kuroda, with Claude).

GAF composes an ordered word pair as **M(a,b) = S(a,b) + λ·G(a,b)**, where
`S = (a⊗b + b⊗a)/2` is symmetric (binding) and `G = (a⊗b − b⊗a)/2` is the
antisymmetric Grassmann term (order), with `λ = σ·γ`. A sentence is composed over a
**pair set** `P` — *the choice of P is the parse* — and scored by the Frobenius cosine
of the two composed matrices against the gold similarity (Spearman ρ).

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
```

## Data (not redistributed)

Download these yourself and set the paths used in `reproduce/`:

| data | source |
|---|---|
| SICK (`SICK_test_annotated.txt`) | Marelli et al. 2014, SICK dataset |
| STS benchmark (`STS_test.csv`) | SemEval-2017 STS-B |
| FastText `cc.en.vec.gz` | fastText crawl vectors (cc.en.300) |
| GloVe `glove.6B.300d.txt` | GloVe 6B |
| wiki2017 triple (GloVe / w2v-SG / ft-SG) | trained on a 2017 Wikipedia dump (gensim `.bin`) |

Embeddings load from text (`.vec`/`.txt`, gz ok) **or** gensim native saves
(`.bin`/`.model`/`.kv`) directly; `gensim_to_vec.py` converts if you prefer text.

## Scripts

| file | role |
|---|---|
| `gaf_engine.py` | numerical core (S/G operator, pair sets, γ) |
| `gaf_core.py` | thin compatibility facade over `gaf_engine` |
| `gaf_eval_random_control.py` | pair-set evaluation (adjacent / n-gram / windows / dependency / constituency / exhaustive / count-matched random null), Jaccard-layered, `--dump-pairs`, `--dump-pairset` |
| `gaf_eval_order_discrimination.py` | reordering minimal-pair test: order detection + meaning-tracking AUC |
| `measure_anisotropy.py` | embedding anisotropy + mean geometric λ ⟨sin²θ⟩ |
| `measure_pair_angles.py` | per-stimulus content-word pair angles and closed-form detection |
| `gensim_to_vec.py` | gensim model/KeyedVectors → text `.vec` |
| `test_gaf_engine.py` | unit tests for the engine |
| `pairs_en.json`, `seeds_en.json`, `gamma_en.tsv` | controlled stimuli, word-class seeds, class-level γ(A,B) table |

Run the tests with `python -m pytest test_gaf_engine.py` (or `python test_gaf_engine.py`).

## Reproduce

`run_all.sh` wraps the commands below; edit the paths at the top of each.
It runs them in order; edit the paths at its top.

**Experiment 1 — pair sets & parses on SICK** (dependency vs constituency, =1.0 split,
`parse_crossover`):
```bash
python gaf_eval_random_control.py -c SICK_test_annotated.txt --sick -e cc.en.vec.gz \
    --with-dep --with-const --const-parser stanza --spacy -R 100 \
    --dump-pairs sick_parse.csv --plot-dir sick_parse --const-cache stanza_const_cache.json
```

**STS, both tokenizations** (the tokenization gate): same call on `STS_test.csv`
(drop `--sick`; add `--no-header`/`--col-*` if needed) with and without `--spacy`.

**Experiment 2 — order discrimination** (meaning-tracking AUC = 0.803):
```bash
python gaf_eval_order_discrimination.py -e cc.en.vec.gz --pairs pairs_en.json \
    --lam-mode gamma-class --seeds seeds_en.json --class-gamma gamma_en.tsv \
    --function-words and,or,but --reorder-only -o order_ft.json
```

**Experiment 3 — the bag-deficit law** (binding gain vs bag quality across embeddings):
run the Experiment-1 call per embedding (`-e cc.en.vec.gz`, `glove.6B.300d.txt`, and the
wiki2017 `.bin` triple), then compute `dependency − averaging` per embedding against each
embedding's `averaging` ρ. Geometry checks: `measure_anisotropy.py <emb>` and
`measure_pair_angles.py <emb> pairs_en.json`.

**Diagnostic — why a parse fails** (actual word pairs per method on chosen rows):
```bash
python gaf_eval_random_control.py -c SICK_test_annotated.txt --sick -e cc.en.vec.gz \
    --with-dep --with-const --const-parser stanza --spacy \
    --dump-pairset 2608,4387,2885,956 --const-cache stanza_const_cache.json
```

## Citation

<!-- fill in once the paper has a venue / arXiv id -->

## License

See `LICENSE`.
