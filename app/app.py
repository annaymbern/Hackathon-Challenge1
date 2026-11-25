"""The demo: pick a yogurt, get the most similar ones with their prices.

Run it from the project root:
    streamlit run app/app.py

What the user sees is deliberately plain. The interesting part is behind it:
the suggestions come from a sentence-embedding model we finetuned on our own
products, and the "compare the two models" switch lets anyone check the claim
we make in the log, live, on whatever product they pick.

Three things this page refuses to fake, because the whole value of the tool is
that someone would act on it while standing in a supermarket:

  It never invents a price. Only 8 of our 300 yogurts have one, and the rest
  say so plainly.
  It never calls two products identical. It says how similar their ingredients
  are, which is a different and smaller claim.
  It says which model produced the suggestions, so a result from the pretrained
  model is never passed off as a result from ours.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from evaluation import load_products          # noqa: E402
from pricing import describe_price, price_per_kilo  # noqa: E402
from similarity import (DEFAULT_MODEL, FINETUNED_DIR, embed,  # noqa: E402
                        finetuned_model_name, rank_with)

st.set_page_config(page_title='Similar yogurts', page_icon='🥛', layout='wide')

TYPE_NAMES = {
    'plain': 'plain', 'greek': 'Greek style', 'fruit': 'with fruit',
    'drinkable': 'drinkable', 'low_fat': 'low fat',
}


@st.cache_data(show_spinner='Loading products...')
def get_products():
    return load_products()


@st.cache_resource(show_spinner='Loading the model...')
def get_vectors(model_name, texts):
    """The vectors, or None when the model cannot be loaded.

    The first run downloads about 500 MB from Hugging Face, so this is the
    step that fails on a locked-down network or a fresh machine with no
    internet. Returning None lets the page say so in a sentence instead of
    showing a stack trace to whoever is watching the demo.
    """
    try:
        return embed(list(texts), model_name)
    except Exception as exc:  # noqa: BLE001 - the page reports whatever went wrong
        return RuntimeError(str(exc))


def label_for(row):
    types = sorted(row['types'])
    if not types:
        return 'type not recorded'
    return ', '.join(TYPE_NAMES.get(t, t) for t in types)


def show_result(position, row, score, query_per_kilo):
    """One suggestion. Everything on screen is either in the data or derived
    from it; nothing is filled in to make the card look complete."""
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            name = row['product_name'] or '(no name)'
            brand = row['brands'] or 'unknown brand'
            st.markdown(f'**{position}. {name}**')
            st.caption(f'{brand} · {label_for(row)}'
                       + ('  ·  supermarket own brand' if row['is_private_label'] else ''))
            ingredients = str(row['ingredients_text'] or '')
            st.caption(ingredients[:160] + ('...' if len(ingredients) > 160 else ''))
        with right:
            st.metric('Similarity', f'{score:.0%}')
            st.caption(describe_price(row))
            their_per_kilo = price_per_kilo(row.get('price_eur'), row.get('quantity'))
            if query_per_kilo and their_per_kilo:
                difference = their_per_kilo - query_per_kilo
                if difference < 0:
                    st.caption(f'**{abs(difference):.2f} EUR/kg cheaper**')
                elif difference > 0:
                    st.caption(f'{difference:.2f} EUR/kg more expensive')


def main():
    st.title('Find a similar yogurt')
    st.caption('Pick the yogurt you usually buy. We show the ones whose '
               'ingredients are closest to it, with their price when we have it.')

    products = get_products()

    finetuned = finetuned_model_name()
    finetuned_available = finetuned is not None
    with st.sidebar:
        st.header('Settings')
        how_many = st.slider('Suggestions', 1, 10, 3)
        compare = st.toggle('Compare both models', value=False,
                            disabled=not finetuned_available,
                            help='Shows the pretrained model next to the one we '
                                 'finetuned, so you can see the difference.')
        only_with_price = st.toggle('Only products with a price', value=False)

        st.divider()
        if finetuned_available:
            where = ('trained on this machine' if FINETUNED_DIR.exists()
                     else f'downloaded from {finetuned}')
            st.success(f'Using the model we finetuned on our products, {where}.')
        else:
            st.warning('The finetuned model is not available, so this is '
                       'running on the pretrained one. Results will be worse '
                       'than the numbers in our log. Run '
                       '`python src/finetune.py` to build it.')
        st.caption(f'{len(products)} yogurts sold in Spain. '
                   f"{int(products['price_eur'].notna().sum())} have a price.")

    names = [f"{row['product_name'] or '(no name)'} — {row['brands'] or 'unknown brand'}"
             for _, row in products.iterrows()]
    # Open on a product that has a price and a readable ingredient list, so the
    # first thing anyone sees is the case where the tool does its whole job.
    # Every other product is one keystroke away in the same box.
    default = next((i for i in range(len(products))
                    if pd.notna(products['price_eur'].iloc[i])
                    and len(str(products['ingredients_text'].iloc[i])) > 40), 0)
    choice = st.selectbox('Your yogurt', options=range(len(names)),
                          format_func=lambda i: names[i], index=default,
                          help='Type to search. All 300 products are in here.')

    query = products.iloc[choice]
    query_per_kilo = price_per_kilo(query.get('price_eur'), query.get('quantity'))

    with st.container(border=True):
        st.markdown(f"### {query['product_name'] or '(no name)'}")
        st.caption(f"{query['brands'] or 'unknown brand'} · {label_for(query)} · "
                   f"{query['quantity'] or 'size not recorded'} · {describe_price(query)}")
        st.caption(str(query['ingredients_text'] or '')[:300])

    st.divider()

    models = [('Our finetuned model' if finetuned_available else 'Pretrained model',
               finetuned if finetuned_available else DEFAULT_MODEL)]
    if compare and finetuned_available:
        models.append(('Pretrained model, for comparison', DEFAULT_MODEL))

    columns = st.columns(len(models)) if len(models) > 1 else [st.container()]

    for column, (label, model_name) in zip(columns, models):
        with column:
            if len(models) > 1:
                st.subheader(label)
            vectors = get_vectors(model_name, tuple(products['text']))
            if isinstance(vectors, Exception):
                st.error('The model could not be loaded, so there are no '
                         'suggestions to show.')
                st.caption(f'{vectors}')
                st.caption('The first run downloads the model from Hugging '
                           'Face. Check the connection, then reload the page.')
                continue

            # Ask for extra results so the price filter still has something to
            # show after it removes the ones without a price.
            wanted = how_many * 10 if only_with_price else how_many
            ranked = rank_with(vectors, choice, min(wanted, len(products) - 1))

            shown = 0
            for index in ranked:
                row = products.iloc[index]
                if only_with_price and pd.isna(row.get('price_eur')):
                    continue
                shown += 1
                show_result(shown, row, float(vectors[index] @ vectors[choice]),
                            query_per_kilo)
                if shown >= how_many:
                    break

            if shown == 0:
                st.info('Nothing to show with those settings. Turn off the '
                        'price filter: only 8 of our products have one.')

    st.divider()
    st.caption(
        'Similarity compares ingredient lists, not taste. A high score means '
        'the ingredients are close, not that the products are the same. '
        'Prices come from Open Prices and were observed on different days, so '
        'treat them as a guide rather than today\'s shelf price.')


if __name__ == '__main__':
    main()
