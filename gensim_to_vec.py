#!/usr/bin/env python3
"""Export a gensim model/KeyedVectors (.bin / native .save) to a text .vec file
that gaf_eval_random_control.py can read.

Handles: KeyedVectors.save(), Word2Vec.save(), FastText.save(), and
word2vec-format binary. For FastText it exports the in-vocabulary word vectors
(word + subword contributions already summed in .wv), which is the representation
GAF should consume.

Usage:
    python gensim_to_vec.py model.bin out.vec
"""
import sys

def load_any(path):
    from gensim.models import KeyedVectors, Word2Vec, FastText
    errs = []
    def wv(obj):                       # normalise model-or-KV to KeyedVectors
        return getattr(obj, "wv", obj)
    # 1) native KeyedVectors.save() (may leniently return a full model)
    try:
        return wv(KeyedVectors.load(path, mmap="r"))
    except Exception as e:
        errs.append(f"KeyedVectors.load: {e}")
    # 2) full Word2Vec / FastText model .save()  -> take .wv
    for name, cls in (("Word2Vec", Word2Vec), ("FastText", FastText)):
        try:
            return cls.load(path).wv
        except Exception as e:
            errs.append(f"{name}.load: {e}")
    # 3) word2vec-format (binary then text)
    for b in (True, False):
        try:
            return KeyedVectors.load_word2vec_format(path, binary=b)
        except Exception as e:
            errs.append(f"load_word2vec_format(binary={b}): {e}")
    raise SystemExit("could not load " + path + "\n  " + "\n  ".join(errs))

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    kv = load_any(sys.argv[1])
    kv.save_word2vec_format(sys.argv[2], binary=False)
    print(f"wrote {sys.argv[2]}: {len(kv.index_to_key)} words, dim {kv.vector_size}")

if __name__ == "__main__":
    main()
