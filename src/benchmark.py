"""Run every method over the same products and print what each one scores.

Usage:
    python src/benchmark.py                    # the methods that need no model
    python src/benchmark.py --methods all      # adds the embedding models
    python src/benchmark.py --k 3 --save results.csv

The rule we fixed in Work 3, before seeing any result: we keep the embeddings
if they beat the word-overlap baseline by at least 10 points. Writing that
down in advance is the whole point, because afterwards almost any result looks
like a reason to keep going.

Two numbers are reported for each method:

  precision@3  of all the suggestions made, how many were a reasonable
               alternative. This is the metric from our Work 3 plan, and it is
               what the user experiences: of the three options on screen, how
               many are worth looking at.
  hit@3        how often at least one of the three was reasonable. A softer
               question, but the one that decides whether the tool was any use
               for that search.

Suggestions whose own type is unknown are counted in neither. They are
reported as a separate percentage, because a quarter of the dataset has no
subcategory tag and hiding that would make the methods look more precise than
they are.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from baselines import METHODS as BASELINE_METHODS
from evaluation import coverage_report, judge, load_products
from textprep import looks_spanish

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def available_methods(include_models=False):
    """Baselines always; the embedding methods only when asked for and
    installable, so the benchmark still runs on a machine with no model."""
    methods = dict(BASELINE_METHODS)
    if not include_models:
        return methods
    try:
        from similarity import METHODS as MODEL_METHODS
    except ImportError as exc:
        raise SystemExit(
            f'Could not load the embedding methods: {exc}\n'
            'Install them with: pip install -r requirements.txt') from exc
    methods.update(MODEL_METHODS)
    return methods


def run_method(products, queries, rank, k=3):
    """One row per (query, suggestion) pair, with the verdict on each."""
    rows = []
    for query_index in queries:
        query = products.iloc[query_index]
        for position, suggestion_index in enumerate(rank(products, query_index, k), start=1):
            suggestion = products.iloc[suggestion_index]
            rows.append({
                'query_barcode': query['barcode'],
                'query_name': query['product_name'],
                'query_types': '|'.join(sorted(query['types'])),
                'query_spanish': bool(looks_spanish(query['ingredients_text'])),
                'position': position,
                'suggestion_barcode': suggestion['barcode'],
                'suggestion_name': suggestion['product_name'],
                'suggestion_types': '|'.join(sorted(suggestion['types'])),
                'correct': judge(query['types'], suggestion['types']),
            })
    return pd.DataFrame(rows)


def summarise(results):
    """Turn the per-suggestion verdicts into the numbers we report."""
    judged = results[results['correct'].notna()]
    unknown = len(results) - len(judged)

    by_query = judged.groupby('query_barcode')['correct'].any()
    # Cross-language cases: the query is written in Spanish and the suggestion
    # is not, or the other way round. This is the split that should separate
    # the embeddings from word counting.
    same_language = judged[judged['query_spanish']]

    return {
        'suggestions': len(results),
        'judged': len(judged),
        'unknown_pct': 100 * unknown / len(results) if len(results) else 0.0,
        'precision_at_3': 100 * judged['correct'].mean() if len(judged) else 0.0,
        'hit_at_3': 100 * by_query.mean() if len(by_query) else 0.0,
        'precision_spanish_query': (100 * same_language['correct'].mean()
                                    if len(same_language) else float('nan')),
        'precision_other_query': (100 * judged[~judged['query_spanish']]['correct'].mean()
                                  if len(judged[~judged['query_spanish']]) else float('nan')),
    }


def per_type(results):
    """precision@3 broken down by what the user searched for.

    This is what tells us whether to narrow the demo: a method that works on
    plain yogurts and fails on drinks is still useful, but only if we say so.
    """
    judged = results[results['correct'].notna()].copy()
    if judged.empty:
        return pd.DataFrame()
    judged['first_type'] = judged['query_types'].str.split('|').str[0]
    grouped = judged.groupby('first_type')['correct']
    return pd.DataFrame({'queries': grouped.size(),
                         'precision_at_3': 100 * grouped.mean()}).round(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', default='baselines',
                        help="'baselines' (default), 'all', or a comma-separated list.")
    parser.add_argument('--k', type=int, default=3, help='Suggestions per query.')
    parser.add_argument('--database', default=None, help='DuckDB file to read.')
    parser.add_argument('--split', default='all', choices=['all', 'train', 'test'],
                        help="Which products to use as queries. Use 'test' once "
                             'finetune.py has run: those products were held out '
                             'of training, so every method is measured on products '
                             'none of them was trained on.')
    parser.add_argument('--save', default=None,
                        help='Write the per-suggestion verdicts to this CSV.')
    args = parser.parse_args()

    products = load_products(args.database)
    coverage = coverage_report(products)
    queries = [i for i, t in enumerate(products['types']) if t]

    if args.split != 'all':
        split_path = PROJECT_ROOT / 'data' / 'train_test_split.csv'
        if not split_path.exists():
            raise SystemExit(f'{split_path} does not exist yet. '
                             'Run src/finetune.py first, or use --split all.')
        split = pd.read_csv(split_path, dtype={'barcode': str})
        wanted_barcodes = set(split.loc[split['split'] == args.split, 'barcode'])
        queries = [i for i in queries
                   if str(products['barcode'].iloc[i]) in wanted_barcodes]

    print(f"Products in the corpus: {coverage['products']}")
    print(f"Usable as test queries: {len(queries)}"
          f" ({coverage['without_type_tag']} have no subcategory tag)")
    if args.split != 'all':
        print(f"Restricted to the '{args.split}' split.")
    print(f"Suggestions per query:  {args.k}\n")

    if not queries:
        raise SystemExit('No usable queries. Check the split and the data.')

    wanted = available_methods(include_models=args.methods != 'baselines')
    if args.methods not in ('baselines', 'all'):
        names = [n.strip() for n in args.methods.split(',')]
        missing = [n for n in names if n not in wanted]
        if missing:
            raise SystemExit(f'Unknown method(s): {", ".join(missing)}. '
                             f'Available: {", ".join(sorted(wanted))}')
        wanted = {n: wanted[n] for n in names}

    summaries, everything = {}, []
    for name, rank in wanted.items():
        results = run_method(products, queries, rank, args.k)
        results.insert(0, 'method', name)
        everything.append(results)
        summaries[name] = summarise(results)
        print(f'--- {name} ---')
        print(per_type(results).to_string() or '  (nothing judgeable)')
        print()

    table = pd.DataFrame(summaries).T
    table = table[['precision_at_3', 'hit_at_3', 'precision_spanish_query',
                   'precision_other_query', 'unknown_pct', 'judged']]
    table.columns = ['precision@3', 'hit@3', 'prec. ES query',
                     'prec. other lang', '% unjudgeable', 'judged']
    print('=== Summary (%) ===')
    print(table.round(1).to_string())

    baseline = summaries.get('word_overlap', {}).get('precision_at_3')
    if baseline is not None:
        print(f'\nDeclared baseline (word_overlap) precision@3: {baseline:.1f}%')
        print('Decision rule from Work 3: keep the embeddings only if they beat '
              f'this by 10 points or more, so they need {baseline + 10:.1f}%.')

    if args.save:
        path = Path(args.save)
        pd.concat(everything, ignore_index=True).to_csv(path, index=False,
                                                        encoding='utf-8-sig')
        summary_path = path.with_suffix('.summary.json')
        summary_path.write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        print(f'\nPer-suggestion verdicts: {path}')
        print(f'Summary:                 {summary_path}')


if __name__ == '__main__':
    main()
