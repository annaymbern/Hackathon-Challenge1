# Project Log – Group 1

Patricia Cerdà, Helena Collart, Jana Biosca and Anna Ymbern
This is our logbook. Every time we make an important decision, change direction or find a problem, we write it here: what we believed, what we found, what we decided and why. Newest entries go at the top.

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


