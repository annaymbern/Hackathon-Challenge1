"""Adjust the embedding model on our own products.

The pretrained model knows what sentences mean in general. It does not know
what matters in a supermarket: that "Greek" and "0% fat" are the distinctions
a shopper cares about, while "pasteurised" and "may contain traces of nuts"
appear on nearly everything and separate nothing.

This teaches it that, using the one thing we have a lot of and did not have to
label ourselves: the subcategory tags. Two products tagged Greek should end up
near each other, a Greek one and a drinkable one should end up apart. The
training examples are triplets, (product, similar product, different product),
and the model is nudged until the similar one is closer than the different one.

    python src/finetune.py                 # train and save to models/finetuned
    python src/finetune.py --epochs 2
    python src/benchmark.py --methods all --split test

Two rules this script keeps, because without them the benchmark afterwards
means nothing:

  The test products never appear in training. The split is written to disk
  first and read back, so it is the same split every time and the same one the
  benchmark uses.

  The split is by product, not by pair. Splitting pairs would put a product in
  training and the same product in the test set, and the model would be
  measured on something it had already been shown.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import load_products

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = PROJECT_ROOT / 'data' / 'train_test_split.csv'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'finetuned'

BASE_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
TEST_FRACTION = 0.30
POSITIVES_PER_PRODUCT = 5
RANDOM_SEED = 20261123


def build_split(products, test_fraction=TEST_FRACTION, path=SPLIT_PATH, seed=RANDOM_SEED):
    """Decide once which products are for training and which for measuring.

    Only products with a subcategory tag can be test queries, since without one
    there is no way to judge the answer. Untagged products stay in the
    searchable catalogue either way: the demo can still return them, we just
    cannot score them.
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, dtype={'barcode': str})

    rng = np.random.default_rng(seed)
    tagged = np.array(products.index[products['types'].apply(bool)].tolist())
    rng.shuffle(tagged)
    cut = int(len(tagged) * test_fraction)
    test = set(tagged[:cut].tolist())

    rows = pd.DataFrame({
        'barcode': products['barcode'],
        'split': ['test' if i in test else 'train' for i in products.index],
        'has_type_tag': products['types'].apply(bool),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, encoding='utf-8-sig')
    return rows


def build_triplets(products, train_index, positives=POSITIVES_PER_PRODUCT, seed=RANDOM_SEED):
    """(anchor, similar, different) examples, all drawn from training products.

    Why triplets and not just pairs: the usual trick is to let everything else
    in the batch act as the "different" example, but with only five product
    types a random other product in the batch is often the same type as the
    anchor. The model would be told that two plain yogurts are unrelated, which
    is the opposite of what we want it to learn. Choosing the negative
    ourselves, from a product that shares no type at all, avoids that.
    """
    rng = np.random.default_rng(seed)
    pool = list(train_index)
    types = {i: products['types'].iloc[i] for i in pool}

    triplets = []
    for anchor in pool:
        same = [i for i in pool if i != anchor and types[i] & types[anchor]]
        different = [i for i in pool if not (types[i] & types[anchor])]
        if not same or not different:
            continue
        take = min(positives, len(same))
        for positive in rng.choice(same, size=take, replace=False):
            negative = int(rng.choice(different))
            triplets.append((anchor, int(positive), negative))
    rng.shuffle(triplets)
    return triplets


def train(products, triplets, output_dir=OUTPUT_DIR, base_model=BASE_MODEL,
          epochs=1, batch_size=16):
    """Train and save. Imported lazily so the rest of the file can be read and
    tested on a machine without the library."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    text = products['text'].tolist()
    examples = [InputExample(texts=[text[a], text[p], text[n]])
                for a, p, n in triplets]

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    # MultipleNegativesRankingLoss with an explicit third text treats it as a
    # hard negative: the model is pushed to rank the positive above it.
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup = max(1, int(0.1 * len(loader) * epochs))
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, show_progress_bar=True)

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--positives', type=int, default=POSITIVES_PER_PRODUCT)
    parser.add_argument('--base-model', default=BASE_MODEL)
    parser.add_argument('--output', default=str(OUTPUT_DIR))
    parser.add_argument('--dry-run', action='store_true',
                        help='Build the split and the triplets, then stop. '
                             'Checks the data without downloading a model.')
    args = parser.parse_args()

    products = load_products()
    split = build_split(products)
    by_barcode = dict(zip(split['barcode'].astype(str), split['split']))
    train_index = [i for i in products.index
                   if by_barcode.get(str(products['barcode'].iloc[i])) == 'train']
    test_index = [i for i in products.index
                  if by_barcode.get(str(products['barcode'].iloc[i])) == 'test']

    triplets = build_triplets(products, train_index, args.positives)

    print(f'Products:              {len(products)}')
    print(f'Training products:     {len(train_index)}')
    print(f'Held out for testing:  {len(test_index)}')
    print(f'Training triplets:     {len(triplets)}')
    print(f'Split saved at:        {SPLIT_PATH}')

    leaked = {a for a, _, _ in triplets} | {p for _, p, _ in triplets} \
        | {n for _, _, n in triplets}
    overlap = leaked & set(test_index)
    print(f'Test products inside training data: {len(overlap)}'
          f'{"  <- PROBLEM" if overlap else "  (none, as it should be)"}')

    if args.dry_run:
        print('\nDry run: nothing was trained.')
        return
    if not triplets:
        raise SystemExit('No training triplets could be built.')

    saved = train(products, triplets, args.output, args.base_model,
                  args.epochs, args.batch_size)
    print(f'\nSaved the finetuned model to {saved}')
    print('Now compare it against everything else:')
    print('    python src/benchmark.py --methods all --split test')


if __name__ == '__main__':
    main()
