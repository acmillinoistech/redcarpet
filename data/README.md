# Data Preparation

## 1. Scrape Data from Web

There are almost 9,000 talent pages to scrape, each taking roughly one second to process. This step can take as long as 150 minutes. Instead of repeating this task, use the scraper results saved in `all_output.json`.

**Note:** This data snapshot was taken on Monday, March 11th, 2019. A list of the pages visited by this scraper run can be found in `scraper_log.txt`. The number of reactions for each talent changes rapidly, so that feature will not be up to date with the current total on the live website.

The scraper may be run again if:

- The data is not 

If the scraper must be run again, ensure that the dependencies are installed and, in this directory, run:

```bash
$ casperjs cameo_scraper.js
```

After completion, the scraper will write the JSON data to `output.json`. If there are any errors, the script will stop and will need to be restarted after the logic has been adjusted.

Possible scraper enhancements:

- Avoid scraping the same talent twice if they occur in multiple categories. For the current snapshot, this would reduce the number of unique talent items from less than 9000 to over 5000.
- Write each category of talent data to a separate file, so that some results are saved if the scraper fails during execution.
- Expand the description section of each talent page and add their full description as a feature.
- Replace this entire step by asking Cameo to share a snapshot of their data.

## 2. Prepare Item Matrix

Run the Jupyter notebook `Prepare Item Matrix.ipynb`, which will perform the following tasks:

- Create a multi-label matrix to represent which categories (labels) a talent (item) belongs to (103 unique categories).
- Drop duplicate item records for talent that belong to multiple categories.

The notebook will write the dataframe to `talent.csv`.

The notebook will also write the useful string constants to `strings.py`:

- `READABLE_LABELS`: the pretty names of each category, in the order of the columns in the multi-label matrix.
- `COLUMN_LABELS`: the actual names of each column in the multi-label matrix, in order.
- `ATTRIBUTES`: the dataframe columns that are not part of the multi-label matrix.

## 3. Simulate User x Item Matrix

...
