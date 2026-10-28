"""Retrieve Spanish yogurt products and prices directly from the internet.

Install: pip install pandas pyarrow duckdb
Run (from anywhere): python src/build_yogurt_dataset.py
Optional: python src/build_yogurt_dataset.py --max-products 100 --no-save
Full source: python src/build_yogurt_dataset.py --max-products 0

Project layout (paths are resolved relative to this file, so the current
working directory does not matter):
    Hackathon-Challenge1/
        src/build_yogurt_dataset.py   <- this script
        data/yogurts_spain.duckdb     <- DuckDB database, table "yogurts"
        data/yogurts_spain.csv        <- same data as CSV

No input CSVs are needed. Source files are never saved to disk. By default,
the prepared result is saved in data/ as yogurts_spain.csv and as a DuckDB
database (yogurts_spain.duckdb, table "yogurts") for SQL analysis.
Read it from other code with:  from build_yogurt_dataset import load_yogurts
Prices are queried with DuckDB directly from the remote Parquet export, so
only the columns and rows needed are transferred instead of the whole file.
The default is a first subset of up to 300 eligible products, not a census.
If filtered product search is unavailable, the script streams the official
compressed export; this fallback can transfer a large amount of data.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import pandas as pd

# Public sources require no account, API key or manually downloaded files.
PRODUCT_API = 'https://world.openfoodfacts.org/api/v2/search'
PRODUCT_EXPORT = 'https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz'
PRICE_EXPORT = 'https://huggingface.co/datasets/openfoodfacts/open-prices/resolve/main/prices.parquet'
USER_AGENT = 'PAIBS-Group1-Yogurt-Prototype/1.0'

# Project paths: <root>/src/build_yogurt_dataset.py and <root>/data/...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'yogurts_spain.csv'
DUCKDB_PATH = DATA_DIR / 'yogurts_spain.duckdb'
DUCKDB_TABLE = 'yogurts'
CHUNK = 50_000
PAGE_SIZE = 100

# These are candidate labels from the team mapping, not verified ground truth.
OWN_BRANDS = {'hacendado', 'carrefour', 'dia', 'eroski', 'milbona', 'pilos',
              'auchan', 'alcampo', 'alipende', 'el corte ingles', 'consum'}
PRODUCT_FIELDS = ['code', 'product_name', 'product_name_es', 'brands',
                  'ingredients_text', 'ingredients_text_es', 'countries_tags',
                  'countries', 'categories_tags', 'categories', 'quantity', 'obsolete']
PRICE_FIELDS = ['product_code', 'type', 'price', 'currency', 'date', 'created', 'id',
                'location_osm_address_country_code', 'location_osm_display_name',
                'price_is_discounted']
OUTPUT_FIELDS = ['barcode', 'product_name', 'brands', 'ingredients_text',
                 'countries_tags', 'categories', 'quantity', 'is_private_label']


# Open a response with timeouts and bounded retries; keep it as a stream.
def open_url(url):
    for attempt in range(3):
        try:
            return urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=45)
        except HTTPError as exc:
            if attempt == 2 or exc.code not in {429, 500, 502, 503, 504}:
                raise
            # Respect a numeric Retry-After value. Stop rather than retrying too early
            # if the server requests a longer pause than this small prototype allows.
            wait = exc.headers.get('Retry-After', '') if exc.headers else ''
            if exc.code == 429 and (not wait.isdigit() or int(wait) > 60):
                raise
            time.sleep(int(wait) if wait.isdigit() else 2 ** (attempt + 1))
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


# Treat absent optional fields as missing text, not as evidence of a negative label.
def text_column(frame, field):
    if field not in frame:
        return pd.Series('', index=frame.index, dtype='string')
    return frame[field].astype('string').fillna('').str.strip()


def normalise_label(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value.casefold())
                   if not unicodedata.combining(c)).strip()


# Match complete brand labels to avoid substring matches such as HacendadoXYZ.
def own_brand_candidate(value):
    return any(normalise_label(brand) in OWN_BRANDS for brand in value.split(','))


# Apply the same eligibility checks to API responses and streamed export batches.
def clean_products(frame):
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_FIELDS)
    barcode = text_column(frame, 'code')
    name = text_column(frame, 'product_name_es')
    name = name.mask(name.eq(''), text_column(frame, 'product_name'))
    ingredients = text_column(frame, 'ingredients_text_es')
    ingredients = ingredients.mask(ingredients.eq(''), text_column(frame, 'ingredients_text'))
    countries = text_column(frame, 'countries_tags')
    if 'countries_tags' in frame:
        # Require the exact country tag when structured tags are available.
        spanish = countries.str.split(',').apply(lambda tags: 'en:spain' in [t.strip() for t in tags])
    else:
        countries = text_column(frame, 'countries')
        spanish = countries.str.contains(r'\b(?:spain|españa|espana)\b', case=False, regex=True)
    categories = text_column(frame, 'categories_tags')
    if 'categories_tags' in frame:
        # Exclude the broader fermented-milk category unless the yogurt tag exists.
        yogurt = categories.str.split(',').apply(lambda tags: 'en:yogurts' in [t.strip() for t in tags])
    else:
        categories = text_column(frame, 'categories')
        yogurt = categories.str.contains(r'\b(?:yogurts?|yoghurts?|yogures?|yaourts?)\b', case=False, regex=True)
    obsolete = text_column(frame, 'obsolete').str.lower().isin(['1', 'true'])
    # Length is only an availability heuristic; it does not verify the recipe.
    usable = ingredients.str.len().ge(8) & name.ne('')
    valid_code = barcode.str.fullmatch(r'(?:\d{8}|\d{12}|\d{13}|\d{14})', na=False)
    keep = spanish & yogurt & usable & valid_code & ~obsolete
    result = pd.DataFrame({'barcode': barcode, 'product_name': name,
                           'brands': text_column(frame, 'brands'),
                           'ingredients_text': ingredients, 'countries_tags': countries,
                           'categories': categories, 'quantity': text_column(frame, 'quantity')})
    result = result.loc[keep].copy()
    # False means no mapped own-brand label was found, not verified manufacturer status.
    result['is_private_label'] = result['brands'].apply(own_brand_candidate)
    return result


# Request only yogurt products tagged as sold in Spain and the fields we use.
def api_product_batches():
    page = 1
    while True:
        query = urlencode({'categories_tags': 'en:yogurts', 'countries_tags': 'en:spain',
                           'fields': ','.join(PRODUCT_FIELDS), 'page_size': PAGE_SIZE, 'page': page})
        with closing(open_url(PRODUCT_API + '?' + query)) as response:
            payload = json.load(response)
        if 'products' not in payload or not isinstance(payload['products'], list):
            raise ValueError('Product search returned an unexpected response schema.')
        products = payload['products']
        if not products:
            return
        # API list fields are converted to the comma-separated form used in exports.
        for product in products:
            for field in ['brands', 'countries_tags', 'categories_tags']:
                if isinstance(product.get(field), list):
                    product[field] = ','.join(product[field])
        yield pd.DataFrame(products)
        count = payload.get('count')
        if len(products) < PAGE_SIZE or (count is not None and page * PAGE_SIZE >= int(count)):
            return
        page += 1
        # Search has a lower rate limit than single-product requests.
        time.sleep(6.5)


# Read and decompress the official TSV export as it arrives over the network.
# No raw CSV, gzip archive or temporary source file is written to the computer.
def streamed_product_batches():
    with closing(open_url(PRODUCT_EXPORT)) as response:
        with gzip.GzipFile(fileobj=response) as source:
            with pd.read_csv(source, sep='\t', usecols=lambda col: col in PRODUCT_FIELDS,
                             chunksize=CHUNK, dtype='string', encoding='utf-8-sig',
                             low_memory=False, on_bad_lines='error') as reader:
                yield from reader


def collect_products(max_products=300, mode='auto'):
    if max_products < 0:
        raise ValueError('--max-products must be zero (all products) or a positive number.')
    # Automatic mode tries the small filtered API first, then the streaming export.
    modes = ['api', 'stream'] if mode == 'auto' else [mode]
    for method in modes:
        selected, scanned, retained = [], 0, 0
        batches = api_product_batches() if method == 'api' else streamed_product_batches()
        try:
            for batch in batches:
                scanned += len(batch)
                clean = clean_products(batch)
                if not clean.empty:
                    selected.append(clean)
                    combined = pd.concat(selected, ignore_index=True).drop_duplicates('barcode')
                    retained = len(combined)
                print(f'    {method}: inspected {scanned:,} rows; retained {retained:,} yogurts.', flush=True)
                if max_products and retained >= max_products:
                    print(f'    Sample limit: first {max_products:,} eligible products; not representative.', flush=True)
                    return combined.head(max_products).reset_index(drop=True), method
            if selected:
                return pd.concat(selected, ignore_index=True).drop_duplicates('barcode').reset_index(drop=True), method
            raise ValueError('No eligible Spanish yogurts with usable ingredient text were found.')
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            if mode != 'auto' or method != 'api':
                raise
            # Restart from the export rather than silently returning partial API results.
            print(f'    Filtered search unavailable: {exc}', flush=True)
            print('    Switching to the official streaming export. It is large and may take time;', flush=True)
            print('    source files will still not be saved to disk.', flush=True)
        finally:
            # Close the HTTP stream even if the sample limit stops iteration early.
            batches.close()


# Query the price export with DuckDB over HTTP. Parquet is columnar and DuckDB
# uses range requests, so only the selected columns and matching row groups travel
# over the network; nothing is written to disk.
PRICE_SQL = """
    WITH spanish AS (
        SELECT product_code AS barcode,
               CAST(price AS DOUBLE) AS price_eur,
               currency,
               date AS price_date,
               created,
               id AS price_observation_id,
               location_osm_address_country_code AS price_country,
               location_osm_display_name AS price_store,
               price_is_discounted
        FROM read_parquet(?)
        -- Country and currency are mandatory checks, not optional assumptions.
        WHERE type = 'PRODUCT'
          AND currency = 'EUR'
          AND location_osm_address_country_code = 'ES'
          AND price > 0
          AND date IS NOT NULL
          AND date <= current_date
          AND product_code IN (SELECT barcode FROM wanted)
    ),
    ranked AS (
        -- Latest dated observation; ties resolved reproducibly by creation and ID.
        SELECT *, row_number() OVER (
                   PARTITION BY barcode
                   ORDER BY price_date DESC, created DESC NULLS LAST, price_observation_id DESC
               ) AS rn
        FROM spanish
    )
    SELECT barcode, price_eur, currency,
           strftime(price_date, '%Y-%m-%d') AS price_date,
           price_country, price_store, price_observation_id, price_is_discounted
    FROM ranked
    WHERE rn = 1
    ORDER BY barcode
"""


def duckdb_connection(database=':memory:'):
    con = duckdb.connect(database)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET http_timeout = 45000;")
    con.execute("SET http_retries = 3;")
    return con


def latest_spanish_prices(barcodes):
    wanted = pd.DataFrame({'barcode': pd.Series(sorted(barcodes), dtype='string')})
    with closing(duckdb_connection()) as con:
        con.register('wanted', wanted)
        prices = con.execute(PRICE_SQL, [PRICE_EXPORT]).df()
    prices['barcode'] = prices['barcode'].astype('string')
    return prices


# Persist the result as a DuckDB table so it can be queried with SQL.
def save_duckdb(products, database_path=DUCKDB_PATH):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(duckdb.connect(str(database_path))) as con:
        con.register('products_df', products)
        con.execute(f'CREATE OR REPLACE TABLE {DUCKDB_TABLE} AS SELECT * FROM products_df')


# Read the prepared dataset back from data/yogurts_spain.duckdb.
# `query` may be any SQL over the "yogurts" table; by default the whole table is returned.
def load_yogurts(query=None, database_path=DUCKDB_PATH):
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(f'{database_path} not found. Run build_yogurt_dataset.py first.')
    with closing(duckdb.connect(str(database_path), read_only=True)) as con:
        return con.execute(query or f'SELECT * FROM {DUCKDB_TABLE}').df()


# Return the prepared dataframe so notebooks can work entirely in memory as well.
def run(output_path=None, max_products=300, product_mode='auto'):
    print('1/3 Retrieving eligible Open Food Facts products...', flush=True)
    products, source_mode = collect_products(max_products, product_mode)
    print('2/3 Querying Open Prices with DuckDB...', flush=True)
    prices = latest_spanish_prices(set(products['barcode']))
    # Preserve ingredient-matching candidates even when their price is missing.
    products = products.merge(prices, on='barcode', how='left', validate='one_to_one')
    products['product_source_mode'] = source_mode
    products['retrieved_at_utc'] = datetime.now(timezone.utc).isoformat()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        products.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'3/3 Saved {len(products):,} products to {output_path}', flush=True)
        duckdb_path = output_path.with_suffix('.duckdb')
        save_duckdb(products, duckdb_path)
        print(f'    DuckDB database: {duckdb_path} (table "yogurts")', flush=True)
    else:
        print(f'3/3 Prepared {len(products):,} products in memory; no CSV was saved.', flush=True)
    print(f'    Products with observed Spanish EUR prices: {products.price_eur.notna().sum():,}')
    print(f'    Own-brand candidates requiring review: {products.is_private_label.sum():,}')
    print('    Prices are observed package prices, not current guarantees or EUR/kg.')
    return products


# Input file paths are no longer required: running the script retrieves both sources.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=CSV_PATH,
                        help='CSV path; the DuckDB database is written next to it with a .duckdb suffix.')
    parser.add_argument('--max-products', type=int, default=300, help='Eligible products to retain; 0 scans the full source.')
    parser.add_argument('--product-mode', choices=['auto', 'api', 'stream'], default='auto')
    parser.add_argument('--no-save', action='store_true', help='Keep the result in memory without writing a CSV.')
    args = parser.parse_args()
    try:
        run(None if args.no_save else args.output, args.max_products, args.product_mode)
    except (HTTPError, URLError, TimeoutError, ValueError, ImportError, duckdb.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
