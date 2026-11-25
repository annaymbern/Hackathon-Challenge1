"""The AI part: find the products whose ingredients mean the same thing.

Counting shared words fails on exactly the case this dataset is full of. Half
our yogurts have their ingredients written in Italian, English, German or
Croatian rather than Spanish, and "leche entera, fermentos lacticos" and
"latte intero, fermenti lattici" share no word at all while describing the
same product. A method that compares words can only see that they are
different.

A sentence embedding model turns each ingredient list into a list of numbers
chosen so that texts meaning similar things land near each other, whatever
language they are written in. Then "most similar product" becomes a distance,
and the language stops mattering.

We use a pretrained multilingual model rather than training one from scratch:
with a few hundred products there is nothing like enough data to learn a
language from zero, and the course is explicit that using existing models is
fine. What is ours is everything around it, and, once finetune.py has run, a
version of the model adjusted on our own products.

Run it on its own to encode the products and see an example:
    python src/similarity.py
    python src/similarity.py --barcode 8410000000017
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'embeddings'
FINETUNED_DIR = PROJECT_ROOT / 'models' / 'finetuned'

# Where the finetuned model lives when it is not on this machine. It weighs
# about 450 MB, too much for the repository, and Colab deletes it when the
# session ends, so it is published on the Hugging Face Hub instead and
# downloaded on first use. Set YOGURT_MODEL_ID to override.
HUB_MODEL_ID = os.environ.get('YOGURT_MODEL_ID', '')

# Small, fast, and trained on fifty languages. The size matters here: it runs
# in a couple of minutes on a laptop with no GPU, which is what the four of us
# have.
DEFAULT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

_LOADED = {}


def _fingerprint(model_name, texts):
    """Identifies one set of vectors: these texts, through this model.

    Without it, changing the cleaning rules or adding products would silently
    leave the old vectors in place and every number after that would be wrong.
    """
    digest = hashlib.sha256(model_name.encode('utf-8'))
    for text in texts:
        digest.update(b'\x00')
        digest.update(text.encode('utf-8'))
    return digest.hexdigest()[:16]


def load_model(model_name=DEFAULT_MODEL):
    """The sentence-transformers model, downloaded once and kept in memory.

    Imported here rather than at the top of the file so that the baselines and
    the benchmark still run on a machine where the library is missing or the
    model has not been downloaded.
    """
    if model_name in _LOADED:
        return _LOADED[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'sentence-transformers is not installed. Run:\n'
            '    pip install -r requirements.txt') from exc
    _LOADED[model_name] = SentenceTransformer(model_name)
    return _LOADED[model_name]


def embed(texts, model_name=DEFAULT_MODEL, use_cache=True, batch_size=32):
    """One vector per product, unit length so a dot product is the cosine.

    Encoding a few hundred products takes about a minute on a laptop, so the
    result is cached on disk. The cache is keyed by the texts themselves, so
    it refreshes by itself when the data or the cleaning changes.
    """
    texts = list(texts)
    key = _fingerprint(model_name, texts)
    cache_path = CACHE_DIR / f'{Path(model_name).name}-{key}.npy'

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    model = load_model(model_name)
    vectors = model.encode(texts, batch_size=batch_size,
                           convert_to_numpy=True, show_progress_bar=True,
                           normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
    return vectors


def rank_with(vectors, query_index, k=3):
    """The k nearest products to the query, by cosine similarity.

    The vectors are unit length, so the whole comparison is one matrix
    multiplication. With a few hundred products that is instant, which is why
    there is no vector database here: it would be a dependency that buys us
    nothing at this size.
    """
    scores = vectors @ vectors[query_index]
    scores = np.asarray(scores, dtype=float).copy()
    scores[query_index] = -np.inf
    order = np.argsort(scores, kind='stable')[::-1][:k]
    return [int(i) for i in order]


def _ranker(model_name=DEFAULT_MODEL):
    """Wrap a model as the (products, query_index, k) function the benchmark
    calls. The vectors are computed on the first call and reused after."""
    cache = {}

    def rank(products, query_index, k=3):
        texts = tuple(products['text'])
        if texts not in cache:
            cache[texts] = embed(texts, model_name)
        return rank_with(cache[texts], query_index, k)

    return rank


def rank_embeddings(products, query_index, k=3):
    """The pretrained model, straight out of the box."""
    return _ranker(DEFAULT_MODEL)(products, query_index, k)


def finetuned_model_name():
    """Where to get our finetuned model, or None if there is nowhere to get it.

    A local copy wins over the Hub: if someone has just retrained it, that is
    the version they want to test, not the one published last week.
    """
    if FINETUNED_DIR.exists():
        return str(FINETUNED_DIR)
    return HUB_MODEL_ID or None


#: Registered so `python src/benchmark.py --methods all` picks them up.
#: The finetuned entry only appears when there is a finetuned model to use, so
#: the benchmark never claims to measure something that does not exist yet.
METHODS = {'embeddings': _ranker(DEFAULT_MODEL)}

_finetuned = finetuned_model_name()
if _finetuned:
    METHODS['embeddings_finetuned'] = _ranker(_finetuned)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Model name, or a path to a finetuned one.')
    parser.add_argument('--barcode', default=None,
                        help='Show the most similar products to this one.')
    parser.add_argument('--index', type=int, default=0,
                        help='Which product to use as the query, by row number.')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--compare', action='store_true',
                        help='Show the pretrained and the finetuned model side '
                             'by side, so you can read the actual suggestions '
                             'instead of trusting the score.')
    args = parser.parse_args()

    from evaluation import load_products

    products = load_products()

    if args.barcode:
        matches = products.index[products['barcode'].astype(str) == str(args.barcode)]
        if len(matches) == 0:
            raise SystemExit(f'No product with barcode {args.barcode}.')
        query_index = int(matches[0])
    else:
        query_index = args.index
    if not 0 <= query_index < len(products):
        raise SystemExit(f'--index must be between 0 and {len(products) - 1}.')

    query = products.iloc[query_index]
    print(f"Query: {query['product_name']}  ({query['brands']})")
    print(f"  types: {'|'.join(sorted(query['types'])) or 'unknown'}")
    print(f"  ingredients: {str(query['ingredients_text'])[:110]}\n")

    models = [('pretrained', args.model)]
    if args.compare:
        finetuned = finetuned_model_name()
        if not finetuned:
            raise SystemExit(f'No finetuned model at {FINETUNED_DIR} and no '
                             'YOGURT_MODEL_ID set. Run src/finetune.py first.')
        models.append(('finetuned', finetuned))

    for label, model_name in models:
        vectors = embed(products['text'], model_name, use_cache=not args.no_cache)
        print(f'--- {label} ---')
        for position, index in enumerate(rank_with(vectors, query_index, args.k), start=1):
            other = products.iloc[index]
            score = float(vectors[index] @ vectors[query_index])
            shared = '|'.join(sorted(other['types'])) or 'unknown'
            same = '' if not (other['types'] & query['types']) else '  <- same type'
            print(f"  {position}. {other['product_name']}  ({other['brands']})")
            print(f"     similarity {score:.3f}   types: {shared}{same}")
        print()


if __name__ == '__main__':
    main()
