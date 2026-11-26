# Find a similar yogurt

**Anna Ymbern, Pati Cerda, Helena Collart, Jana Biosca — Team 3, PAIBS**

Pick the yogurt you usually buy, and the app shows you the ones whose
ingredients are closest to it, with their price per kilo when we have it.

## Run the demo

```bash
git clone https://github.com/annaymbern/Hackathon-Challenge1.git
cd Hackathon-Challenge1
pip install -r requirements.txt
streamlit run app/app.py
```

That is everything. The data is in the repository, and the model downloads
itself from Hugging Face the first time you run it (about 450 MB, a minute or
two). Nothing to configure, no API key, no account.

It opens in your browser. Pick a yogurt from the box at the top, and turn on
**Compare both models** in the sidebar to see the model we trained next to the
one we started from, on whatever product you choose.

## What it does, and why it needs AI

Deciding that two yogurts are the same kind of thing is harder than it sounds.
Ingredient lists are free text written by volunteers, in a different order
every time, and half of the Spanish catalogue has them in Italian, English,
German or Croatian. "Leche entera, fermentos lácticos" and "latte intero,
fermenti lattici" describe the same product and share no word at all.

So we turn each ingredient list into a vector with a sentence-embedding model,
and the nearest vectors are the most similar products. Then we finetuned that
model on our own products, teaching it which distinctions matter in a
supermarket, using the Open Food Facts subcategory tags as the training signal.

## Does it work

Measured on 66 products held out of training, percentage of the three
suggestions that are the same type as the one searched:

| Method | Overall | Query in Spanish | Query in another language |
|---|---|---|---|
| Random pick | 42.4% | 46.3% | 21.7% |
| Shared word count | 65.4% | 67.1% | 54.5% |
| Jaccard | 74.5% | 77.4% | 58.3% |
| Pretrained embeddings | 81.9% | 84.1% | 68.2% |
| **Our finetuned model** | **88.6%** | **89.1%** | **86.2%** |

The last column is the result we care about. Counting words loses about 19
points when the ingredients are not in Spanish. Our model loses 3.

Reproduce it:

```bash
python src/finetune.py                              # trains, about 5 minutes
python src/benchmark.py --methods all --split test  # prints the table
```

The seed is fixed, so you should get the same numbers. Full results and the
caveats are in [results/](results/) and [LOG.md](LOG.md).

## What is honest about it

- **It never invents a price.** Only 8 of our 300 yogurts have one. The rest
  say "no price" instead of guessing.
- **It never says two products are identical.** It says how similar the
  ingredients are, which is a smaller claim.
- **It says which model it is using.** A result from the pretrained model is
  never passed off as a result from ours.
- **77 of 300 products have no subcategory tag**, so we cannot score the
  suggestions that land on them. We report those separately rather than
  counting them as mistakes.

## The code

| File | What it does |
|---|---|
| `src/build_yogurt_dataset.py` | Downloads products from Open Food Facts and prices from Open Prices. Category and country are parameters |
| `src/textprep.py` | Cleans names and ingredient text. Every method reads the same text, so none gets an advantage |
| `src/evaluation.py` | Decides what counts as a correct suggestion, and builds the evaluation set |
| `src/baselines.py` | The methods with no AI in them: shared word count, Jaccard, random |
| `src/similarity.py` | The embedding model and the nearest-neighbour search |
| `src/finetune.py` | Trains the model on our products, holding the test set out first |
| `src/benchmark.py` | Runs every method over the same products and prints the comparison |
| `src/pricing.py` | Reads the quantity field and works out euros per kilo |
| `src/mercadona_yogures.py` | Scrapes Mercadona prices. Written, not yet connected to the app |
| `app/app.py` | The demo |

Rebuild the dataset from scratch, or try another category:

```bash
python src/build_yogurt_dataset.py                       # Spanish yogurts
python src/build_yogurt_dataset.py --category en:cheeses # anything else
```

## Train it in the browser instead

If you would rather not install anything, the whole pipeline runs in Colab:

[02_finetune_and_benchmark.ipynb](https://colab.research.google.com/github/annaymbern/Hackathon-Challenge1/blob/main/notebooks/02_finetune_and_benchmark.ipynb)

## Data

- [Open Food Facts](https://world.openfoodfacts.org/), ODbL. 300 yogurts sold
  in Spain, in `data/yogurts_spain.parquet`.
- [Open Prices](https://prices.openfoodfacts.org/), joined on the barcode.
  Coverage outside France is thin, which is why only 8 of our products have a
  price.
- The finetuned model:
  [helenacollart/yogurt-similarity](https://huggingface.co/helenacollart/yogurt-similarity)

## What we would do next

Connect the Mercadona scraper so more than 8 products have a price, check the
29 products marked `core` in `data/eval_set.csv` by hand to see how often the
Open Food Facts tags are wrong, and widen beyond yogurts.

Our decisions, including the ones that turned out wrong, are in
[LOG.md](LOG.md).
