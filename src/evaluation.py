"""What counts as a correct suggestion, and the set of products we measure on.

The problem this solves: to say whether a suggested yogurt is a reasonable
alternative to the one the user picked, we need to know what both of them are.
Nothing in the dataset states that directly, so we take it from Open Food
Facts' own subcategory tags.

Why the tags and not labels we write ourselves from the product name: the
word-overlap baseline ranks products by the words they share, most of which
come from the name. If we also derived the ground truth from the name, the
baseline would be scored against its own input and would look better than it
is. The subcategory tags are entered separately by contributors, so they are
independent of the text every method reads.

They are crowdsourced, so they are not perfect. That is why a sample is marked
for the team to check by hand: the disagreement rate tells us how much the
automatic labels can be trusted, and it goes in the log.
"""
from pathlib import Path

import pandas as pd

from textprep import product_text, product_tokens

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
EVAL_SET_PATH = DATA_DIR / 'eval_set.csv'

# The subcategory tags that tell two yogurts apart, with the name we use for
# them. Tags such as en:yogurts or en:dairies sit on all 300 products, so they
# separate nothing and are left out.
#
# A product can carry more than one: 16 are tagged both Greek and plain, 10
# both Greek and fruit. We keep every tag a product has rather than forcing one
# label, because "plain Greek yogurt" is genuinely both and a suggestion that
# matches either reading is a fair suggestion.
TYPE_TAGS = {
    'en:yogurt-drinks': 'drinkable',
    'en:greek-style-yogurts': 'greek',
    'en:fruit-yogurts': 'fruit',
    'en:plain-yogurts': 'plain',
    'en:low-fat-yogurts': 'low_fat',
}

# How many products of each type go into the sample the team checks by hand,
# plus how many untagged products to look at. The untagged ones matter: a
# quarter of the dataset has no subcategory tag at all, and checking a few
# tells us whether those are products Open Food Facts simply missed or products
# that are genuinely hard to classify.
HAND_CHECK_PER_TYPE = 6
HAND_CHECK_UNTAGGED = 10
RANDOM_SEED = 20261123


def types_of(categories):
    """The set of type names a product carries, from its categories field."""
    if not isinstance(categories, str):
        return frozenset()
    tags = {tag.strip() for tag in categories.split(',') if tag.strip()}
    return frozenset(TYPE_TAGS[tag] for tag in tags & TYPE_TAGS.keys())


def judge(query_types, suggestion_types):
    """Is this suggestion a reasonable alternative to the query?

    Returns True, False, or None when the suggestion has no type tag and so
    cannot be judged either way. Unjudgeable suggestions are reported
    separately instead of being quietly counted as wrong, because counting
    missing data as a failure of the model would misread a gap in the dataset
    as a gap in the method.
    """
    if not suggestion_types:
        return None
    return bool(query_types & suggestion_types)


def load_products(database_path=None, table='yogurts'):
    """The product table with the columns every method and the metric need."""
    import duckdb

    database_path = Path(database_path or DATA_DIR / 'yogurts_spain.duckdb')
    if not database_path.exists():
        raise FileNotFoundError(
            f'{database_path} not found. Run src/build_yogurt_dataset.py first.')
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        products = con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()

    products = products.reset_index(drop=True)
    products['types'] = products['categories'].apply(types_of)
    products['text'] = [product_text(n, i) for n, i
                        in zip(products['product_name'], products['ingredients_text'])]
    products['tokens'] = [product_tokens(n, i) for n, i
                          in zip(products['product_name'], products['ingredients_text'])]
    return products


def build_eval_set(products, path=EVAL_SET_PATH):
    """Write the evaluation set to disk, once.

    Every product with at least one type tag becomes a test query. Within that,
    a stratified sample is marked 'core' for the team to check by hand, and a
    few untagged products are marked 'untagged' so somebody looks at the gap.

    The file is written once and then treated as fixed. Rebuilding it after
    seeing a result would let us pick the test that flatters the model, which
    is the thing an evaluation set exists to prevent.
    """
    rows = products.copy()
    rows['type_labels'] = rows['types'].apply(lambda t: '|'.join(sorted(t)))
    rows['is_query'] = rows['types'].apply(bool)

    rng = __import__('numpy').random.default_rng(RANDOM_SEED)
    review = pd.Series('', index=rows.index, dtype=object)

    for type_name in sorted(set(TYPE_TAGS.values())):
        members = rows.index[rows['types'].apply(lambda t: type_name in t)]
        take = min(HAND_CHECK_PER_TYPE, len(members))
        if take:
            picked = rng.choice(members.to_numpy(), size=take, replace=False)
            review.loc[picked] = 'core'

    untagged = rows.index[~rows['is_query']]
    take = min(HAND_CHECK_UNTAGGED, len(untagged))
    if take:
        picked = rng.choice(untagged.to_numpy(), size=take, replace=False)
        review.loc[picked] = 'untagged'

    rows['review_group'] = review
    # Columns the team fills in by hand. Left empty on purpose: an automatic
    # label in a column called "checked by hand" would be a lie.
    rows['hand_label'] = ''
    rows['agrees_with_tags'] = ''
    rows['checked_by'] = ''
    rows['notes'] = ''

    columns = ['barcode', 'product_name', 'brands', 'quantity', 'type_labels',
               'is_query', 'review_group', 'hand_label', 'agrees_with_tags',
               'checked_by', 'notes', 'ingredients_text']
    out = rows[columns]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding='utf-8-sig')
    return out


def coverage_report(products):
    """The numbers that say whether this evaluation is worth running."""
    total = len(products)
    tagged = int(products['types'].apply(bool).sum())
    per_type = {name: int(products['types'].apply(lambda t, n=name: n in t).sum())
                for name in sorted(set(TYPE_TAGS.values()))}
    multi = int(products['types'].apply(lambda t: len(t) > 1).sum())
    return {'products': total, 'with_type_tag': tagged,
            'without_type_tag': total - tagged, 'with_more_than_one_type': multi,
            'per_type': per_type}


if __name__ == '__main__':
    products = load_products()
    report = coverage_report(products)
    print(f"Products:              {report['products']}")
    print(f"With a type tag:       {report['with_type_tag']}"
          f" ({report['with_type_tag'] / report['products']:.0%})")
    print(f"Without a type tag:    {report['without_type_tag']}"
          f" ({report['without_type_tag'] / report['products']:.0%})")
    print(f"With more than one:    {report['with_more_than_one_type']}")
    print('Per type:')
    for name, count in report['per_type'].items():
        print(f'    {name:12s} {count}')
    written = build_eval_set(products)
    core = int((written['review_group'] == 'core').sum())
    untagged = int((written['review_group'] == 'untagged').sum())
    print(f'\nWrote {EVAL_SET_PATH} with {len(written)} rows.')
    print(f'    {int(written.is_query.sum())} usable as test queries')
    print(f'    {core} marked for hand checking, {untagged} untagged ones to look at')
