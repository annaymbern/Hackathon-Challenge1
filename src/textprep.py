"""Turn a product's name and ingredient list into comparable text.

Every method we compare (word overlap, pretrained embeddings, fine-tuned
embeddings) reads the text produced here, so none of them gets an advantage
from a different cleaning step. That is the only reason this lives in its own
module instead of inside each method.

Open Food Facts ingredient text is written by volunteers in whatever language
the packaging uses, with the site's own markup mixed in: _leche_ marks an
allergen, asterisks point at footnotes, and percentages, storage instructions
and batch codes are common. None of that says anything about what the product
is, so it is stripped before anything else looks at the text.
"""
import re
import unicodedata

# Words that carry no information about what a product is. They are frequent in
# Spanish, Catalan, English, French, German, Italian and Portuguese ingredient
# lists, which is roughly the language mix in the Spanish subset. Removing them
# makes the word-overlap baseline stronger, not weaker: without this, two
# unrelated products score a match for sharing "de" and "y".
STOPWORDS = {
    # Spanish and Catalan
    'de', 'del', 'la', 'el', 'los', 'las', 'y', 'e', 'con', 'sin', 'en', 'al',
    'un', 'una', 'o', 'u', 'que', 'por', 'para', 'su', 'sus', 'mas', 'muy',
    'amb', 'sense', 'i', 'les', 'dels', 'als',
    # English
    'the', 'and', 'or', 'with', 'without', 'of', 'in', 'from', 'a', 'an',
    'may', 'contain', 'contains', 'traces',
    # French, German, Italian, Portuguese
    'et', 'avec', 'sans', 'aux', 'des', 'les', 'le', 'du', 'ou',
    'und', 'mit', 'ohne', 'von', 'aus', 'kann', 'enthalten',
    'con', 'senza', 'di', 'da', 'il', 'lo', 'gli', 'per',
    'com', 'sem', 'dos', 'das', 'na', 'no',
    # Packaging and storage boilerplate that appears inside ingredient fields
    'conservar', 'consumir', 'preferentemente', 'antes', 'fecha', 'lote',
    'refrigerado', 'nevera', 'agitar', 'envase', 'producto', 'informacion',
}

# OFF markup and anything that is a measurement rather than an ingredient.
_MARKUP = re.compile(r'[_*]')
_MEASURES = re.compile(r'\d+[\d.,]*\s*(?:%|g|kg|ml|l|mg|kcal|kj|º|°)?', re.IGNORECASE)
_NON_LETTER = re.compile(r'[^a-z\s]')
_SPACES = re.compile(r'\s+')


def strip_accents(value):
    """cafés -> cafes. Volunteers are inconsistent with accents, so folding them
    stops the same word counting as two different ones."""
    return ''.join(c for c in unicodedata.normalize('NFKD', value)
                   if not unicodedata.combining(c))


def clean(value):
    """Lowercase, remove accents, OFF markup, numbers and punctuation."""
    if value is None:
        return ''
    text = strip_accents(str(value).casefold())
    text = _MARKUP.sub(' ', text)
    text = _MEASURES.sub(' ', text)
    text = _NON_LETTER.sub(' ', text)
    return _SPACES.sub(' ', text).strip()


def tokens(value, drop_stopwords=True, min_length=3):
    """The cleaned text as a list of words.

    Words shorter than three letters are dropped with the stopwords: they are
    almost all articles and units that survived the steps above.
    """
    words = clean(value).split()
    if not drop_stopwords:
        return words
    return [w for w in words if len(w) >= min_length and w not in STOPWORDS]


# Words that appear in almost any Spanish or Catalan ingredient list for a
# dairy product and in almost no other language's. Used only to split the
# benchmark results by language, never to rank anything.
_SPANISH_MARKERS = {
    'leche', 'llet', 'azucar', 'sucre', 'fermentos', 'ferments', 'nata',
    'desnatada', 'entera', 'semidesnatada', 'lacticos', 'lactics', 'cultivos',
    'edulcorante', 'almidon', 'midons', 'aroma', 'yogur', 'iogurt',
}


def looks_spanish(ingredients):
    """A rough guess at whether the ingredient list is in Spanish or Catalan.

    This is a keyword heuristic, not language detection, and it will get some
    products wrong. It is good enough for its only job: splitting the benchmark
    into same-language and cross-language cases, which is where we expect the
    embeddings and the word count to differ most.
    """
    return bool(set(tokens(ingredients)) & _SPANISH_MARKERS)


def product_text(name, ingredients):
    """The single string every method reads for one product.

    The name is kept because around a quarter of the products carry almost no
    ingredient text, and the name still says what the thing is.
    """
    return f'{clean(name)} {clean(ingredients)}'.strip()


def product_tokens(name, ingredients):
    """The same text as a set of words, for the word-overlap baseline."""
    return set(tokens(name)) | set(tokens(ingredients))
