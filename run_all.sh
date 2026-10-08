#!/usr/bin/env bash
# Reproduce the paper's experiments. Edit the paths below to your local data.
set -euo pipefail

# ---- paths (EDIT) ----
SICK=SICK_test_annotated.txt
STS=STS_test.csv
FT=cc.en.vec.gz                 # FastText crawl (weak-bag case)
GLOVE=glove.6B.300d.txt         # GloVe 6B
GLOVE_W17=models/glove-wiki2017-300/model.bin   # gensim .bin triple
W2V_W17=models/w2v-skg-wiki2017-300/model.bin
FT_W17=models/ft-skg-wiki2017-300/model.bin
CACHE=stanza_const_cache.json
R=100

echo "== unit tests =="
python -m pytest -q test_gaf_engine.py || python test_gaf_engine.py

echo "== Exp 1: SICK pair sets + parses (FastText) =="
python gaf_eval_random_control.py -c "$SICK" --sick -e "$FT" \
    --with-dep --with-const --const-parser stanza --spacy -R "$R" \
    --dump-pairs sick_parse.csv --plot-dir sick_parse --const-cache "$CACHE"

echo "== STS: tokenization gate (content-lemma, then whitespace) =="
python gaf_eval_random_control.py -c "$STS" -e "$FT" \
    --with-dep --with-const --const-parser stanza --spacy -R "$R" \
    --dump-pairs sts_parse.csv --const-cache "$CACHE"
python gaf_eval_random_control.py -c "$STS" -e "$FT" -R "$R" \
    --dump-pairs sts_white.csv            # whitespace tokens (no --spacy)

echo "== Exp 2: order discrimination (gamma-class) =="
python gaf_eval_order_discrimination.py -e "$FT" --pairs pairs_en.json \
    --lam-mode gamma-class --seeds seeds_en.json --class-gamma gamma_en.tsv \
    --function-words and,or,but --reorder-only -o order_ft.json

echo "== Exp 3: bag-deficit law — one SICK run per embedding =="
for E in "$FT" "$GLOVE" "$GLOVE_W17" "$W2V_W17" "$FT_W17"; do
  tag=$(basename "$E" | tr -c 'A-Za-z0-9' '_')
  python gaf_eval_random_control.py -c "$SICK" --sick -e "$E" \
      --with-dep --spacy -R "$R" --dump-pairs "sick_${tag}.csv" --const-cache "$CACHE"
done
# geometry checks
python measure_anisotropy.py "$FT";   python measure_anisotropy.py "$GLOVE"
python measure_pair_angles.py "$FT" pairs_en.json
python measure_pair_angles.py "$GLOVE" pairs_en.json

echo "== Diagnostic: pair-set dump on maximal-contrast rows =="
python gaf_eval_random_control.py -c "$SICK" --sick -e "$FT" \
    --with-dep --with-const --const-parser stanza --spacy \
    --dump-pairset 2608,4387,2885,956 --const-cache "$CACHE"

echo "done."
