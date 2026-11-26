"""The demo: pick a yogurt, see which ones are the same thing and what they cost.

Run it from the project root:
    streamlit run app/app.py

The page is laid out like a supermarket shelf label, because that is the
comparison a shopper already knows how to read: the product, the price, and
underneath in small print the price per kilo, which is the only figure that
lets you compare a 125 g pot with a four-pack.

Three things this page refuses to fake, because its whole value is that
someone would act on it while standing in a shop:

  It never invents a price. The products without one say so.
  It never calls two products identical. It says how close their ingredients
  are, which is a smaller and truer claim.
  It says which model produced the suggestions, so a result from the
  pretrained model is never passed off as a result from ours.
"""
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'app'))

from evaluation import load_products                      # noqa: E402
from pricing import price_per_kilo                        # noqa: E402
from similarity import (DEFAULT_MODEL, FINETUNED_DIR,     # noqa: E402
                        embed, finetuned_model_name, rank_with)
from style import CSS, euros                              # noqa: E402

st.set_page_config(page_title='Which yogurts are the same?',
                   page_icon='🥛', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)

TYPE_NAMES = {'plain': 'plain', 'greek': 'Greek style', 'fruit': 'with fruit',
              'drinkable': 'drinkable', 'low_fat': 'low fat'}


@st.cache_data(show_spinner='Reading the products...')
def get_products():
    return load_products()


@st.cache_resource(show_spinner='Loading the model...')
def get_vectors(model_name, texts):
    """The vectors, or the exception that stopped us getting them.

    The first run downloads about 450 MB from Hugging Face, so this is the
    step that fails on a blocked network. Handing the error back lets the page
    explain itself in a sentence rather than show a stack trace to a room.
    """
    try:
        return embed(list(texts), model_name)
    except Exception as exc:  # noqa: BLE001 - the page reports whatever broke
        return RuntimeError(str(exc))


def esc(value, limit=None):
    text = str(value or '').strip()
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + '…'
    return html.escape(text)


def kinds(row):
    types = sorted(row['types'])
    if not types:
        return 'type not recorded'
    return ', '.join(TYPE_NAMES.get(t, t) for t in types)


def price_block(row, reference_per_kilo):
    """The right-hand side of the shelf label."""
    price = row.get('price_eur')
    if price is None or pd.isna(price):
        return '<div class="price"><div class="nil">no price</div></div>'

    store = row.get('price_store')
    store_line = (f'<div class="kg">{esc(store)}</div>'
                  if isinstance(store, str) and store.strip() else '')
    per_kilo = price_per_kilo(price, row.get('quantity'))

    if per_kilo is None:
        return (f'<div class="price"><div class="big">{euros(float(price))}</div>'
                f'<div class="kg">size unknown</div>{store_line}</div>')

    difference = ''
    if reference_per_kilo:
        gap = per_kilo - reference_per_kilo
        if gap < -0.01:
            difference = f'<div class="save">{euros(abs(gap))}/kg less</div>'
        elif gap > 0.01:
            difference = f'<div class="more">{euros(gap)}/kg more</div>'

    return (f'<div class="price"><div class="big">{euros(float(price))}</div>'
            f'<div class="kg">{euros(per_kilo)}/kg</div>'
            f'{store_line}{difference}</div>')


def suggestion(row, score, reference_per_kilo):
    percent = max(0, min(100, round(score * 100)))
    own = '<span class="own">own brand</span>' if row['is_private_label'] else ''
    return f"""
<div class="tag">
  <div class="score">
    <div class="n">{percent}<span>%</span></div>
    <div class="bar"><i style="width:{percent}%"></i></div>
  </div>
  <div>
    <div class="name">{esc(row['product_name']) or '(no name)'}{own}</div>
    <div class="meta">{esc(row['brands']) or 'unknown brand'} · {kinds(row)}</div>
    <div class="ing">{esc(row['ingredients_text'], 180)}</div>
  </div>
  {price_block(row, reference_per_kilo)}
</div>"""


def main():
    products = get_products()
    finetuned = finetuned_model_name()
    have_ours = finetuned is not None

    with st.sidebar:
        st.markdown('**Settings**')
        how_many = st.slider('Suggestions', 1, 10, 3)
        compare = st.toggle('Compare with the pretrained model', value=False,
                            disabled=not have_ours,
                            help='Puts the model we started from next to ours, '
                                 'on the product you picked.')
        priced_only = st.toggle('Only products with a price', value=False)

        with_price = int(products['price_eur'].notna().sum())
        scraped = products.attrs.get('prices_from_mercadona', 0)
        st.markdown(
            f'<div class="note">{len(products)} yogurts sold in Spain. '
            f'{with_price} have a price'
            + (f', {scraped} of them read off the Mercadona site.'
               if scraped else ' from Open Prices.')
            + ('<br><br>Running the model we finetuned on these products, '
               + ('trained on this machine.' if FINETUNED_DIR.exists()
                  else f'from {esc(finetuned)}.')
               if have_ours else
               '<br><br>The finetuned model is not available, so these '
               'suggestions come from the pretrained one and will be worse '
               'than the figures in our log.')
            + '</div>', unsafe_allow_html=True)

    st.markdown('<h1 class="masthead">Which yogurts are the same?</h1>'
                '<p class="standfirst">Pick the one you usually buy. We compare '
                'ingredient lists, across languages, and show what the closest '
                'ones cost per kilo.</p>', unsafe_allow_html=True)

    names = [f"{row['product_name'] or '(no name)'} — {row['brands'] or 'unknown brand'}"
             for _, row in products.iterrows()]
    # Open on a product with a price and a readable ingredient list, so the
    # first thing anyone sees is the case where the tool does its whole job.
    default = next((i for i in range(len(products))
                    if pd.notna(products['price_eur'].iloc[i])
                    and len(str(products['ingredients_text'].iloc[i])) > 40), 0)
    choice = st.selectbox('Your yogurt', options=range(len(names)),
                          format_func=lambda i: names[i], index=default,
                          help='Type to search. All 300 products are here.')

    query = products.iloc[choice]
    reference = price_per_kilo(query.get('price_eur'), query.get('quantity'))
    price_line = (f'{euros(float(query["price_eur"]))}'
                  + (f' · {euros(reference)}/kg' if reference else '')
                  if pd.notna(query.get('price_eur')) else 'no price')

    st.markdown(f"""
<div class="yours">
  <div class="name">{esc(query['product_name']) or '(no name)'}</div>
  <div class="meta">{esc(query['brands']) or 'unknown brand'} · {kinds(query)}
    · {esc(query['quantity']) or 'size not recorded'} · {price_line}</div>
  <div class="ing">{esc(query['ingredients_text'], 320)}</div>
</div>""", unsafe_allow_html=True)

    models = [('Closest by ingredients', finetuned if have_ours else DEFAULT_MODEL)]
    if compare and have_ours:
        models.append(('The model we started from', DEFAULT_MODEL))

    columns = st.columns(len(models), gap='large') if len(models) > 1 else [st.container()]

    for column, (heading, model_name) in zip(columns, models):
        with column:
            st.markdown(f'<div class="sectionhead">{heading}</div>',
                        unsafe_allow_html=True)
            vectors = get_vectors(model_name, tuple(products['text']))
            if isinstance(vectors, Exception):
                st.error('No suggestions: the model did not load.')
                st.caption(f'{vectors}')
                st.caption('The first run downloads it from Hugging Face. '
                           'Check the connection and reload.')
                continue

            # Ask for more than we need, so the price filter still has
            # something left after it removes the products without one.
            wanted = how_many * 20 if priced_only else how_many
            ranked = rank_with(vectors, choice, min(wanted, len(products) - 1))

            cards, shown = [], 0
            for index in ranked:
                row = products.iloc[index]
                if priced_only and pd.isna(row.get('price_eur')):
                    continue
                cards.append(suggestion(row, float(vectors[index] @ vectors[choice]),
                                        reference))
                shown += 1
                if shown >= how_many:
                    break

            if cards:
                st.markdown(''.join(cards), unsafe_allow_html=True)
            else:
                st.info(f'No priced products among the closest matches. '
                        f'Only {with_price} of our {len(products)} have a price, '
                        'so turn that filter off to see the rest.')

    st.markdown(
        '<div class="note">Similarity compares ingredient lists, not taste. '
        'A high figure means the ingredients are close, not that the products '
        'are the same. Prices were observed on different days and in one '
        'region, so read them as a guide rather than today\'s shelf price.'
        '</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
