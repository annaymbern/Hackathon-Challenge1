"""The methods the AI has to beat.

A number on its own says nothing. "The model puts a valid alternative in the
top 3 seventy percent of the time" is only good news if something simpler does
worse, so before measuring the model we measure two things that are not AI at
all:

  random      pick three products at random. The floor. If the model barely
              beats this, the dataset has so few distinct products that the
              choice does not mean much.
  word_overlap  count the words two products share and return the three with
              the most in common. No model, no training, about fifteen lines.

word_overlap is the one that matters. It is a real method, not a straw man:
two natural yogurts written in Spanish do share most of their ingredient
words, and on those cases it is hard to beat. Where it should fall apart is
across languages, because "leche entera" and "latte intero" share no words at
all while being the same thing. That is exactly where the embeddings are
supposed to earn their place, and the per-language split in the benchmark is
what shows whether they do.
"""
import numpy as np

SEED = 20261123


def _ranked(scores, query_index, k):
    """Top k products by score, never returning the query itself."""
    scores = np.asarray(scores, dtype=float)
    scores[query_index] = -np.inf
    # argsort is ascending, so take the tail and reverse it.
    top = np.argsort(scores, kind='stable')[::-1][:k]
    return [int(i) for i in top if np.isfinite(scores[i])]


def rank_word_overlap(products, query_index, k=3):
    """Shared-word count: the baseline as described in our Work 3 plan.

    Counting raw shared words favours products with long ingredient lists,
    which is a real weakness of this method and one reason we also report
    Jaccard below. We keep this version as the declared baseline because it is
    the one we committed to before seeing any result.
    """
    query = products['tokens'].iloc[query_index]
    scores = [len(query & other) for other in products['tokens']]
    return _ranked(scores, query_index, k)


def rank_jaccard(products, query_index, k=3):
    """Shared words divided by total distinct words across the two products.

    The same idea as word_overlap, corrected for length so a product with a
    forty-word ingredient list does not win every comparison. Reported
    alongside the declared baseline so that if the embeddings win, they are
    winning against the stronger of the two simple methods.
    """
    query = products['tokens'].iloc[query_index]
    scores = []
    for other in products['tokens']:
        union = len(query | other)
        scores.append(len(query & other) / union if union else 0.0)
    return _ranked(scores, query_index, k)


def rank_random(products, query_index, k=3, seed=SEED):
    """Three products picked at random. The floor the others have to clear."""
    rng = np.random.default_rng(seed + query_index)
    choices = [i for i in range(len(products)) if i != query_index]
    size = min(k, len(choices))
    return [int(i) for i in rng.choice(choices, size=size, replace=False)]


#: Methods the benchmark knows about. The embedding methods register themselves
#: in similarity.py, so this file stays free of any AI dependency and the
#: baseline can always be run, even on a machine with no model downloaded.
METHODS = {
    'random': rank_random,
    'word_overlap': rank_word_overlap,
    'jaccard': rank_jaccard,
}
