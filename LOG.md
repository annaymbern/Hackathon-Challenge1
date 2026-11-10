# Project Log – Group 1

Patricia Cerdà, Helena Collart, Jana Biosca and Anna Ymbern
This is our logbook. Every time we make an important decision, change direction or find a problem, we write it here: what we believed, what we found, what we decided and why. Newest entries go at the top.

## 10 Nov 2026
### Change of approach: from own-brand dupes to similar products with prices
- **What we believed:** the most useful tool was a private-label dupe finder: pick a branded yogurt and get its own-brand equivalent.
- **What we found:** limiting the results to own-brand products is a very big restriction. The dataset has no column that marks a product as private label, so we depend on our own hand-made brand list. Only 86 of our 300 yogurts are own-brand candidates, and none of them are verified. It also hides cheaper or better options that happen to be branded.
- **Decided:** the user enters a yogurt (barcode) and the system shows the most similar yogurts by ingredients, of any brand, with their prices next to them. The user decides which one to buy.
- **Why:** it is more useful for someone on a budget, it uses all our data instead of a third of it, and it no longer depends on labels we cannot verify.
- **What stays the same:** the AI part. We still use embeddings of the ingredient lists to decide which products are similar.
- **What changes:** own-brand becomes an optional tag next to each result, not a filter. Prices become critical.
**Our line:** we are building a tool that finds yogurts with similar ingredients and shows their prices, for shoppers on a budget, so they can choose a cheaper alternative to the product they usually buy.

- ### Work 1
- Riskiest question: 

“Can our AI find the yogurts that are really most similar to a given one from its ingredients, and do it better than simply counting shared words?”

- Real/Faked:

Ingestion of Open Food Facts subset (Spanish yogurts): Real. Our script loads 300 products into DuckDB. It is the first 300 eligible, not a representative sample.
Cleaning and filtering: Real. Filters by country, yogurt category, valid barcode and usable ingredient text.
Price join with Open Prices (by barcode): Real. The join works, but only 8 of 300 products have a Spanish price.
Mercadona price scraper: Real, not connected. The script exists, but its output is not joined to the dataset yet.
Own-brand tag: Faked (by hand). A hand-made list of 11 supermarket brands gives 86 candidates, not verified. It is now only an optional tag.
Ingredient text cleaning: Stub. Not built yet. We use the raw text for now.
Embedding model (our AI): Stub. Not built yet. It will return a fixed answer in Work 2 and become real in Work 4. 
Similarity search (top 3 similar yogurts): Stub. Not built yet. It depends on the embeddings.
Environmental impact (Green-Score): Stub. Not built yet. The pipeline does not retrieve this column.
Evaluation set (30 yogurts labelled by type): Stub. Not built yet. Needed to answer the riskiest question.
Baseline (word-overlap rule): Stub. Not built yet. Needed to answer the riskiest question.
Output (Streamlit page): Stub. Not built yet.

- ### Work 2: Our stack

| Component | Tool | Level | Used before? | Alternative |
|---|---|---|---|---|
| Data: yogurts from Open Food Facts | Python script + DuckDB | 1 | 1 member (Jana), DuckDB first used on 28 Oct | Hugging Face `datasets` in streaming mode |
| Data: prices | DuckDB on the Open Prices Parquet file + `requests` scraper for Mercadona | 1 | 1 member (Jana) | A small price CSV filled in by hand |
| Processing: cleaning and joining | pandas | 1–2 | Yes, all 4 | DuckDB SQL |
| AI: embeddings of name + ingredients | `sentence-transformers`, multilingual pretrained model | 3 | Yes, all 4 (embeddings studied in Deep Learning class) | TF-IDF with scikit-learn |
| Similarity search: top 3 | scikit-learn cosine similarity | 2–3 | Yes, all 4 | FAISS or ChromaDB |
| Baseline | Word-overlap rule in plain Python | 3 | Yes, all 4 | TF-IDF similarity |
| Evaluation | CSV with 30 yogurts labelled by type + a script that prints % correct | 3 | Yes, all 4 have built evaluation code | Google Sheet checked by hand |
| Output | Streamlit page: barcode in, 3 similar yogurts with prices out | 2 | Yes, all 4 (Data Visualization class) | Gradio |
| Sharing the demo | Hugging Face Spaces | 4 | No one | Streamlit Community Cloud |
| Team | GitHub repo + GitHub Projects | 0–4 | Yes, all 4 | Trello |

- **Tools nobody has used:** Hugging Face Spaces. We can ask for coaching when we reach Level 4.
- **Why no vector database:** with 300 products, comparing all against all with scikit-learn is instant.
- **Why a pretrained multilingual model:** ingredient lists come in several languages and we have no data to train our own.

### Work 3 – Our benchmark plan

- **Metric:** % of suggested yogurts that are the same type as the yogurt the user picked. For each test yogurt the system suggests 3, so with 30 test yogurts we check 90 suggestions. It matters to the user because it answers: "of the 3 alternatives I get, how many are real alternatives?"
- **Evaluation set:** 30 yogurts from our dataset, 5 of each type: natural, Greek, fruit-flavoured, low-fat / 0%, drinkable and high protein. We include some with ingredients not written in Spanish. We choose them before running any model, save them in `data/eval_set.csv` and do not change them afterwards.
- **How we know the right answer:** we label the type of each yogurt by hand, reading its name and ingredients. A suggestion is correct when its type is the same as the test yogurt's. Two of us check each suggestion without knowing which method produced it.
- **Baseline:** a word overlap rule. It counts the words that two products share in their name and ingredients and suggests the 3 with the most shared words. No AI.
- **Fair comparison:** the baseline and the embedding model get the same text and the same 30 yogurts.
- **Decision rule, fixed in advance:** we keep the embeddings if they beat the baseline by at least 10 points. If not, we log it as a result and decide whether to keep the simple rule or move the AI to another component.

### Work 4 – From stub to real

**Key component:** the similarity model that finds the 3 most similar yogurts. It was a stub at the start of the session. Now it is real and runs inside the skeleton, but it is not measured on the full benchmark yet.

| Component | Today | How it becomes real | How we test it |
|---|---|---|---|
| Similarity model | Real, checked by hand on 3 yogurts | Pretrained multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) on name + ingredients, then the 3 closest by cosine similarity | Benchmark from Work 3: beat the word-overlap baseline by at least 10 points on our 30 test yogurts |

**What we checked by hand:** we ran the baseline and the model on 3 yogurts.

| Test yogurt | Word-overlap baseline | AI (embeddings) |
|---|---|---|
| Greek-style yogurt, ingredients in English | 0 of 3 Greek: a natural, a Skyr and a 0% yogurt | 3 of 3 Greek, written in German, Spanish and Catalan |
| Greek yogurt, in Spanish | 3 of 3 Greek | 2 of 3 Greek, the third is a natural yogurt |
| Low-fat natural yogurt, in Catalan | 2 natural yogurts and 1 protein product | 3 natural yogurts, but we cannot tell from the name whether they are low-fat |

- **What we learned:** the model helps most when the ingredients are in different languages, which happens in about half of our dataset. When both products are in Spanish, counting words works just as well. Neither method seems to capture "low-fat" well.
- **What could still fail:** three cases are not a result. The model may not beat the baseline by 10 points on the 30 test yogurts. Almost none of the suggested yogurts has a price.
- **What blocked us:** the DuckDB file could not be opened with a different DuckDB version, so we exported the data to CSV and the code now reads the CSV.
- **Level:** we started the session at Level 1 and we are now at Level 2 (the skeleton runs end to end). Level 3 needs the benchmark result.
- **Next steps:** check the 30 test yogurts by hand, decide whether low-fat counts as its own type, run the benchmark for both methods and log the result. Then connect the Mercadona prices.
  
## 28 Oct 2026
- **Tasks:** merging and cleaning the data.
- **Realizations:** we realized that we wanted our solution to be more environmentally oriented, so we are considering joining an additional CSV with Green Scores.
- **Data retrieval:** developed a Python pipeline to retrieve Spanish yogurt products from Open Food Facts and match them with Open Prices using barcodes.
- **Technical challenges:** the Open Food Facts API returned errors, so we switched to processing the dataset in chunks. We also introduced DuckDB to handle large price datasets more efficiently.
- **Initial results:** retrieved 300 yogurt products, including 86 potential private-label products, but only 8 had available Spanish prices.
- **Decision:** keep products without prices for ingredient-based similarity and explore additional price sources if needed. Next, we will expand the dataset and validate the DuckDB integration.

## 25 Oct 2026
- **Found:** the dataset has no prices and doesn't say which products are own-brand.
- **Decided:** we'll join it with Open Prices using the barcode, and mark own brands by hand.


## 30 Sep 2026
- **Decided:** out of 5 ideas, we chose Idea 3 (dupe finder) and Idea 4 (best quality per €), because they work with the data and the AI does the hard part.


