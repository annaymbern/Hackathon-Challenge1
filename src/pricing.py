"""Work out what a product costs per kilo.

Comparing prices between yogurts is harder than it sounds. One is sold as a
single 500 g tub, another as a pack of four 125 g pots, a third as "1 kg (8 x
125 g)". A price tag on its own says nothing: 2.79 euros is expensive for one
pot and cheap for a pack of eight. The only honest comparison is per kilo, and
to get there we have to read the quantity field, which contributors type by
hand in whatever format they feel like.

When the quantity cannot be read, we return nothing and the app says so. An
invented weight would turn into an invented saving, and a saving that is not
real is worse than no saving at all: the point of the tool is that the user
trusts the number enough to act on it in the supermarket.
"""
import re

# "4 x 125 g", "2 x 130g", "8 x 125 g": a pack. The total is the product of
# the two numbers, and it is checked before the single-quantity patterns
# because "500 g (4 x 125 g)" contains both.
_PACK = re.compile(r'(\d+)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*(kg|g|gr|gram|gramos|ml|cl|l|litro)\b',
                   re.IGNORECASE)

# The same pack written the other way round: "125 g x 4".
_PACK_REVERSED = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(kg|g|gr|gram|gramos|ml|cl|l|litro)\s*[x×]\s*(\d+)\b',
    re.IGNORECASE)

# "500 g", "1 kg", "150g", "500 gram", "1,5 l"
_SINGLE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(kg|g|gr|gram|gramos|ml|cl|l|litro)\b',
                     re.IGNORECASE)

# Grams per unit. Millilitres are treated as grams: yogurt is mostly water, so
# the difference is a few percent, well inside the error we already carry from
# prices observed on different days. Drinkable yogurts are the ones this
# affects, and the app labels those prices as approximate.
_TO_GRAMS = {
    'kg': 1000.0, 'g': 1.0, 'gr': 1.0, 'gram': 1.0, 'gramos': 1.0,
    'ml': 1.0, 'cl': 10.0, 'l': 1000.0, 'litro': 1000.0,
}


def parse_quantity_grams(quantity):
    """Total grams in the package, or None when the field cannot be read.

    "1 kg (8 x 125 g)" and "500 g (4 x 125 g)" both describe the same thing
    twice, and the two readings agree, so either is fine. We take the pack
    form because it is the one that appears on its own ("4 x 125 g" with no
    total), while a bare total never comes with a pack that contradicts it.
    """
    if not isinstance(quantity, str) or not quantity.strip():
        return None

    pack = _PACK.search(quantity)
    if pack:
        count = float(pack.group(1))
        size = float(pack.group(2).replace(',', '.'))
        unit = _TO_GRAMS.get(pack.group(3).lower())
        if unit and 0 < count <= 100:
            return count * size * unit

    reversed_pack = _PACK_REVERSED.search(quantity)
    if reversed_pack:
        size = float(reversed_pack.group(1).replace(',', '.'))
        unit = _TO_GRAMS.get(reversed_pack.group(2).lower())
        count = float(reversed_pack.group(3))
        if unit and 0 < count <= 100:
            return count * size * unit

    single = _SINGLE.search(quantity)
    if single:
        size = float(single.group(1).replace(',', '.'))
        unit = _TO_GRAMS.get(single.group(2).lower())
        if unit:
            grams = size * unit
            # A yogurt under 20 g or over 5 kg is a typo, not a product.
            if 20 <= grams <= 5000:
                return grams
    return None


def price_per_kilo(price_eur, quantity):
    """Euros per kilo, or None when either half is missing."""
    if price_eur is None:
        return None
    try:
        price = float(price_eur)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    grams = parse_quantity_grams(quantity)
    if not grams:
        return None
    return price * 1000.0 / grams


def describe_price(row):
    """One short line about the price, honest about what is missing.

    Three cases, and the user can tell them apart: we know the price per kilo,
    we know the price but not the size, or we know neither.
    """
    price = row.get('price_eur')
    if price is None or (isinstance(price, float) and price != price):
        return 'no price'
    per_kilo = price_per_kilo(price, row.get('quantity'))
    if per_kilo is None:
        return f'{float(price):.2f} EUR (size unknown, cannot compare)'
    return f'{float(price):.2f} EUR  ({per_kilo:.2f} EUR/kg)'
