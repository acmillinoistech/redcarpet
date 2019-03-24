# Data Preparation

## 1. Scrape Data from Web

There are almost 9,000 talent pages to scrape, each taking roughly one second to process. This step can take as long as 150 minutes. Instead of repeating this task, use the scraper results saved in `all_output.json`.

**Note:** This data snapshot was taken on Monday, March 11th, 2019. A list of the pages visited by this scraper run can be found in `scraper_log.txt`. The number of reactions for each talent changes rapidly, so that feature will not be up to date with the current total on the live website.

The scraper may need to be run again if:

- New talent are added
- Ratings data (reactions and stars) is no longer representative of user trends
- New features must be added to the item model

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

The attributes for each talent are:

- `id`: a unique identifier for the talent, also the URL of their webpage: `cameo.com/${id}`.
- `name`: display name for the talent.
- `price`: price in dollars to book the talent.
- `reactions`: the number of users who have reacted to videos from the talent (non-negative integer).
- `stars`: the cumulative rating from users who booked the talent (float between 0 and 5, inclusive).
- `joined`: the month and year in which the talent first became available on the platform.
- `categories`: a list of readable category names the talent belongs to, more convenient to view than the item matrix.

## 3. Simulate Transactions Matrix

The transactions matrix has rows for each user and columns for each item. A "hit" in the matrix indicates a transaction (purchase, like, rating) from the user concerning the item.

In this step, simulate a transaction matrix that students will use along with the item matrix to train their recommender systems.
