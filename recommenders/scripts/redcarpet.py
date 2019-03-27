import numpy as np
import pandas as pd
import ml_metrics
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from IPython.display import HTML


"""
redcarpet: Module for recommender systems using sets
"""

"""
HELPER METHODS
"""


def nonzero_index_set(arr):
    """
    Returns a set of indices corresponding to non-zero
    entries in a numpy array (or other list-like).
    """
    res = set()
    for i, val in enumerate(arr):
        if val > 0:
            res.add(i)
    return res


def mat_to_sets(mat):
    """
    Converts a numpy matrix into a list of sets of column
    indices corresponding to non-zero row entries.
    """
    return [nonzero_index_set(row) for row in mat]


def get_recs(user_recs, k=None):
    """
    Extracts recommended item indices, leaving out their scores.
    params:
        user_recs: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the
            list of tuples sorted in order of decreasing relevance
        k: maximumum number of recommendations to include for each
            user, if None, include all recommendations
    returns:
        list of lists of recommendations where each
            list has the column indices of recommended items
            sorted in order they appeared in user_recs
    """
    recs = [[item for item, score in recs][0:k] for recs in user_recs]
    return recs


def write_kaggle_recs(recs_list, filename=None, headers=["Id", "Predicted"]):
    """
    Writes recommendations to file in Kaggle submission format.
    params:
        recs_list: list of lists of recommendations where each
            list has the column indices of recommended items
            sorted in order of decreasing relevance
        filename: path to file for writing output
        headers: list of strings of output columns, defaults to
            submission columns: ["Id", "Predicted"]
    returns:
        int: number of non-header lines, where each line represents
            a user and the recommendations given to them
    """
    if filename is None:
        raise ValueError("Must provide a filename.")
    lines = [",".join(headers)]
    for i, recs in enumerate(recs_list):
        lines.append("{},{}".format(i, " ".join([str(v) for v in recs])))
    text = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(text)
    return len(lines) - 1


def download_kaggle_recs(recs_list, filename=None, headers=["Id", "Predicted"]):
    """
    Writes recommendations to file in Kaggle submission format.
    params:
        recs_list: list of lists of recommendations where each
            list has the column indices of recommended items
            sorted in order of decreasing relevance
        filename: path to file for writing output
        headers: list of strings of output columns, defaults to
            submission columns: ["Id", "Predicted"]
    returns:
        html: HTML download link to display in a notebook, click
            to download the submission file
    """
    # Based on: https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel
    if filename is None:
        raise ValueError("Must provide a filename.")
    rec_df = pd.DataFrame(
        [(i, " ".join([str(r) for r in recs])) for i, recs in enumerate(recs_list)],
        columns=headers,
    )
    csv = rec_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = """<a download="{filename}"
                href="data:text/csv;base64,{payload}"
                target="_blank">Download ({lines} lines): {filename}</a>"""
    html = html.format(payload=payload, filename=filename, lines=len(rec_df))
    return HTML(html)


def check_list_of_sets(s_data, var_name):
    if not isinstance(s_data, list):
        raise ValueError(
            "{} must be a list of sets. Got: {}".format(var_name, type(s_data))
        )
    if len(s_data) > 0:
        entry = s_data[0]
        if not isinstance(entry, set):
            raise ValueError(
                "{} must be a list of sets. Got list of: {}".format(
                    var_name, type(entry)
                )
            )


"""
EVALUATION METRICS
"""


def mapk_score(s_hidden, recs_pred, k=10):
    """
    Computes the mean average precision at k (MAP@K) of recommendations.
    MAP@K = mean AP@K score over all users
    AP@K = (1 / min(m, k)) * sum from 1 to k of (precision at i * relevance of ith item)
    Where m is the number of items in a user's hidden set
    Where k is the number of items recommended to each user
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        float, range [0, 1]
    """
    check_list_of_sets(s_hidden, "s_hidden")
    return ml_metrics.mapk(s_hidden, recs_pred, k)


def uhr_score(s_hidden, recs_pred, k=10):
    """
    Computes the user hit rate (UHR) score of recommendations.
    UHR = the fraction of users whose top list included at
    least one item also in their hidden set.
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        float, range [0, 1]
    """
    check_list_of_sets(s_hidden, "s_hidden")
    if len(s_hidden) != len(recs_pred):
        note = "Length of true list {} does not match length of recommended list {}."
        raise ValueError(note.format(len(s_hidden), len(recs_pred)))
    scores = []
    for r_true, r_pred_orig in zip(s_hidden, recs_pred):
        r_pred = list(r_pred_orig)[0:k]
        intersect = set(r_true).intersection(set(r_pred))
        scores.append(1 if len(intersect) > 0 else 0)
    return np.mean(scores)


def get_apk_scores(s_hidden, recs_pred, k=10):
    """
    Returns the average precision at k (AP@K) for each user.
    AP@K = (1 / min(m, k)) * sum from 1 to k of (precision at i * relevance of ith item)
    Where m is the number of items in a user's hidden set
    Where k is the number of items recommended to each user
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        list of floats, each float in the range [0, 1]
    """
    check_list_of_sets(s_hidden, "s_hidden")
    apks = []
    for r_true, r_pred in zip(s_hidden, recs_pred):
        apk = mapk_score([r_true], [r_pred], k=k)
        apks.append(apk)
    return apks


def get_hit_counts(s_hidden, recs_pred, k=10):
    """
    Returns the number of successful recommendations for each user.
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        list of integers, each integer in the range [0, k]
    """
    check_list_of_sets(s_hidden, "s_hidden")
    hits = []
    for r_true, r_pred in zip(s_hidden, recs_pred):
        ix = r_true.intersection(set(r_pred[0:k]))
        hits.append(len(ix))
    return hits


def get_all_scores(rec_scores, k=10):
    """
    Get scores of all items in the list of lists of recommendations.
    """
    all_scores = []
    for recs in rec_scores:
        for (item, score) in recs[0:k]:
            all_scores.append(score)
    return all_scores


"""
ANALYSIS TOOLS
"""


def show_apk_dist(s_hidden, models, k=10, bin_size=0.1):
    """
    Plot a histogram of average precision scores for all users.
    """
    bins = np.arange(0, 1 + bin_size, bin_size)
    pal = sns.color_palette("hls", len(models))
    for ((rec_scores, name), color) in zip(models, pal):
        apks = get_apk_scores(s_hidden, get_recs(rec_scores), k=k)
        sns.distplot(apks, kde=False, label=name, bins=bins, color=color)
    plt.xticks(bins)
    plt.xlabel("Average Precision in Top {}".format(k))
    plt.ylabel("Number of Users")
    plt.title("AP@K Score Distribution")
    plt.gcf().set_size_inches((8, 5))
    plt.grid()
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Models", frameon=False
    )
    plt.show()


def show_hit_dist(s_hidden, models, k=10):
    """
    Plot a histogram of hit counts for all users.
    """
    bins = range(k + 1)
    pal = sns.color_palette("hls", len(models))
    for ((rec_scores, name), color) in zip(models, pal):
        hits = get_hit_counts(s_hidden, get_recs(rec_scores), k=k)
        sns.distplot(hits, kde=False, label=name, bins=bins, color=color)
    plt.xticks(bins)
    plt.xlabel("Number of Successful Recommendations in Top {}".format(k))
    plt.ylabel("Number of Users")
    plt.title("Hit Count Distribution")
    plt.gcf().set_size_inches((8, 5))
    plt.grid()
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Models", frameon=False
    )
    plt.show()


def show_score_dist(models, k=10, bins=None):
    """
    Plot a histogram of item recommendation scores for all users.
    """
    pal = sns.color_palette("hls", len(models))
    for ((rec_scores, name), color) in zip(models, pal):
        scores = get_all_scores(rec_scores, k=k)
        if bins is not None:
            sns.distplot(scores, kde=False, label=name, color=color, bins=bins)
        else:
            sns.distplot(scores, kde=False, label=name, color=color)
    if bins is not None:
        plt.xticks(bins)
    plt.xlabel("Score for Recommended Item in Top {}".format(k))
    plt.ylabel("Number of Items")
    plt.title("Item Score Distribution")
    plt.gcf().set_size_inches((8, 5))
    plt.grid()
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Models", frameon=False
    )
    plt.show()


def show_user_detail(s_input, s_hidden, rec_scores, uid, name_fn=None, k=10):
    """
    Show the detailed results of recommendations to a user.
    """
    s_pred = get_recs(rec_scores)
    print("User: {}".format(uid))
    print("Given:       {}".format(sorted(s_input[uid])))
    print("Recommended: {}".format(sorted(s_pred[uid])))
    print("Actual:      {}".format(sorted(s_hidden[uid])))
    set_intersect = set(s_pred[uid]).intersection(set(s_hidden[uid]))
    n_intersect = len(set_intersect)
    apk = mapk_score([s_hidden[uid]], [s_pred[uid]], k)
    print()
    print("Recommendation Hits = {}".format(n_intersect))
    print("Average Precision   = {0:.3f}".format(apk))
    print()
    print("All Recommendation Scores:")
    for i, (item_id, score) in enumerate(rec_scores[uid]):
        hit = "Y" if item_id in s_hidden[uid] else " "
        item_name = "Item {}".format(item_id)
        if name_fn is not None:
            item_name = name_fn(item_id)
        print(
            "{0}. [{3}] ({2:.3f}) {1}".format(
                str(i + 1).zfill(2), item_name, score, hit
            )
        )


def show_user_recs(s_hidden, rec_scores, k=10):
    """
    Show a table of recommendation results by user.
    """
    apks = get_apk_scores(s_hidden, get_recs(rec_scores), k=k)
    hits = get_hit_counts(s_hidden, get_recs(rec_scores), k=k)
    cols = ["User", "APK", "Hits"]
    data = {"User": range(len(rec_scores)), "APK": apks, "Hits": hits}
    return pd.DataFrame(data)[cols]


def show_item_recs(s_hidden, rec_scores, k=10):
    """
    Show a table of recommendation results by item.
    """
    item_res = {}
    for (user, likes) in zip(rec_scores, s_hidden):
        for (i, score) in user:
            if i not in item_res:
                item_res[i] = {"Item": i, "Results": [], "Scores": []}
            item_res[i]["Results"].append(1 if i in likes else 0)
            item_res[i]["Scores"].append(score)
    res = []
    for i in item_res:
        record = item_res[i]
        total = len(record["Results"])
        hits = sum(record["Results"])
        res.append(
            {
                "Item": i,
                "Recommended": total,
                "Hits": hits,
                "Hit Rate": hits / total,
                "Avg Score": np.mean(record["Scores"]),
            }
        )
    cols = ["Item", "Recommended", "Hits", "Hit Rate", "Avg Score"]
    return pd.DataFrame(res)[cols]


"""
SIMILARITY MEASURES
"""


def jaccard_sim(u, v):
    """
    Computes the Jaccard similarity between sets u and v.
    sim = intersection(u, v) / union(u, v)
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    intersection = len(u.intersection(v))
    union = len(u.union(v))
    zero = 1e-10
    # Add small value to denominator to avoid divide by zero
    sim = intersection / (union + zero)
    return sim


def cosine_sim(u, v):
    """
    Computes the Cosine similarity between sets u and v.
    sim = intersection(u, v) / sqrt(|u| * |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    intersection = len(u.intersection(v))
    mag_u = len(u)
    mag_v = len(v)
    zero = 1e-10
    # Add small value to denominator to avoid divide by zero
    sim = intersection / (np.sqrt(mag_u * mag_v) + zero)
    return sim


def forbes_sim(u, v):
    """
    Computes the Forbes similarity between sets u and v.
    sim = a/((a+b)*(a+c))
    Where a = # of items in intersection(u, v)
          b = # of items only in u
          c = # of items only in v
    Note: n is omitted since it is constant for all vectors.
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    a = len(u.intersection(v))
    b = len(u) - a
    c = len(v) - a
    zero = 1e-10
    sim = a / (((a + b) * (a + c)) + zero)
    return sim


def mcconnoughy_sim(u, v):
    """
    Computes the McConnoughy similarity between sets u and v.
    sim = (a*a - b*c) / sqrt((a+b)*(a+c))
    Where a = # of items in intersection(u, v)
          b = # of items only in u
          c = # of items only in v
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    a = len(u.intersection(v))
    b = len(u) - a
    c = len(v) - a
    zero = 1e-10
    sim = ((a * a) - (b * c)) / (np.sqrt((a + b) * (a + c)) + zero)
    return sim


def simpson_sim(u, v):
    """
    Computes the Simpson similarity coefficient between sets u and v.
    sim = intersection(u, v) / min(|u|, |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    ix = len(u.intersection(v))
    zero = 1e-10
    sim = ix / (min(len(u), len(v)) + zero)
    return sim


def first_kulczynski_sim(u, v):
    """
    Computes the first Kulczynski similarity between sets u and v.
    sim = a / (b + c)
    Where a = # of items in intersection(u, v)
          b = # of items only in u
          c = # of items only in v
    Note: If (b + c) is zero, this measure is undefined. In this
        implementation, a small value (1e-4) is added to the
        denominator to avoid division by zero. Consequently, the
        similarity between two sets with `a` matches will be equal
        to a / 1e-4, which is equivalent to a * 1000
    params:
        u, v: sets to compare
    returns:
        float from zero to infinity, where higher scores represents
            greater similarity and zero represents no similarity
    """
    a = len(u.intersection(v))
    b = len(u) - a
    c = len(v) - a
    zero = 1e-4
    sim = a / (b + c + zero)
    return sim


def second_kulczynski_sim(u, v):
    """
    Computes the second Kulczynski similarity between sets u and v.
    sim = (1/2) * ((a / (a + b)) + (a / (a + c)) )
    Where a = # of items in intersection(u, v)
          b = # of items only in u
          c = # of items only in v
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    a = len(u.intersection(v))
    b = len(u) - a
    c = len(v) - a
    zero = 1e-10
    sim = ((a / (a + b + zero)) + (a / (a + c + zero))) / 2
    return sim


def sorenson_dice_sim(u, v):
    """
    Computes the Sørensen-Dice similarity coefficient between sets u and v.
    sim = (2 * intersection(u, v)) / (|u| + |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
    """
    ix = len(u.intersection(v))
    zero = 1e-10
    sim = (2 * ix) / (len(u) + len(v) + zero)
    return sim


"""
ASSOCIATION RULE METRICS
Based on: https://github.com/resumesai/resumesai.github.io/blob/master/analysis/Rule%20Mining.ipynb
"""


def sets_to_contingency(a, b, N):
    """
    Creates a contingency table from two sets.
    params:
        a, b: sets to compare
        N: total number of possible items
    returns:
        (f11, f10, f01, f00) tuple of contingency table entries:
        f11 = # of items both in a and b
        f10 = # of items only in a
        f01 = # of items only in b
        f00 = # of items not in either a or b
    """
    f11 = len(a.intersection(b))
    f10 = len(a) - f11
    f01 = len(b) - f11
    f00 = N - (f11 + f10 + f01)
    return (f11, f10, f01, f00)


def rule_support(f11, f10, f01, f00):
    """
    Computes the support for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum support and
            0 indicates no support
    """
    N = f11 + f10 + f01 + f00
    zero = 1e-10
    return f11 / (N + zero)


def rule_confidence(f11, f10, f01, f00):
    """
    Computes the confidence for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum confidence and
            0 indicates no confidence
    """
    zero = 1e-10
    return f11 / (f11 + f10 + zero)


def rule_lift(f11, f10, f01, f00):
    """
    Computes the lift for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float ranging from zero to infinity where 1 implies independence, greater
        than 1 implies positive association, less than 1 implies negative association
    """
    N = f11 + f10 + f01 + f00
    zero = 1e-10
    supp_ab = f11 / N
    supp_a = f10 / N
    supp_b = f01 / N
    return supp_ab / ((supp_a * supp_b) + zero)


def rule_conviction(f11, f10, f01, f00):
    """
    Computes the conviction for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1]
    """
    N = f11 + f10 + f01 + f00
    zero = 1e-10
    supp_b = f01 / N
    conf = rule_confidence(f11, f10, f01, f00)
    return (1 - supp_b) / ((1 - conf) + zero)


def rule_power_factor(f11, f10, f01, f00):
    """
    Computes the rule power factor (RPF) for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1]
    """
    N = f11 + f10 + f01 + f00
    zero = 1e-10
    supp_ab = f11 / N
    supp_a = f10 / N
    return (supp_ab * supp_ab) / (supp_a + zero)


def rule_interest_factor(f11, f10, f01, f00):
    """
    Computes the interest factor for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum interest and
            0 indicates no interest
    """
    N = f11 + f10 + f01 + f00
    zero = 1e-10
    f1p = f11 + f10
    fp1 = f11 + f01
    return (N * f11) / ((f1p * fp1) + zero)


def rule_phi_correlation(f11, f10, f01, f00):
    """
    Computes the phi correlation for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [-1, 1] where 1 indicates perfect positive correlation and
            -1 indicates perfect negative correlation.
    """
    f1p = f11 + f10
    f0p = f01 + f00
    fp1 = f11 + f01
    fp0 = f10 + f00
    num = (f11 * f00) - (f01 * f10)
    denom = np.sqrt(f1p * fp1 * f0p * fp0)
    if denom == 0:
        return 0.0
    return num / denom


def rule_is_score(f11, f10, f01, f00):
    """
    Computes the IS score for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum int3erest and
            0 indicates no int3erest
    """
    intfac = rule_interest_factor(f11, f10, f01, f00)
    supp = rule_support(f11, f10, f01, f00)
    return np.sqrt(intfac * supp)


def mine_association_rules(m_train, min_support=0.5):
    """
    Finds association rules using the Apriori algorithm. Produces rules of the form `a -> b`,
    which suggests that if a user likes item `a`, then they may also like item `b`.
    params:
        m_train: matrix of train data, rows = users, columns = items, 1 = like, 0 otherwise
        min_support: A float between 0 and 1 for minumum support of the itemsets returned.
          The support is computed as the fraction:
            transactions_where_item(s)_occur / total_transactions
    returns:
        rule_df: Pandas dataframe of association rules, with columns:
            "a": antecedent (LHS) item index
            "b": consequent (RHS) item index
            "ct": tuple of contingency table entries for the rule
            "support": support for the rule `a -> b` in m_train
    """
    freq_is = apriori(pd.DataFrame(m_train), max_len=2, min_support=min_support)
    freq_is["len"] = freq_is["itemsets"].apply(lambda s: len(s))
    freq_is = freq_is.query("len == 2")
    if len(freq_is) == 0:
        return pd.DataFrame([], columns=["a", "b", "ct", "support"])
    item_counts = m_train.sum(axis=0)
    rules = []
    for record in freq_is.to_dict(orient="records"):
        fset = record["itemsets"]
        a = min(fset)
        b = max(fset)
        n = len(m_train)
        supp = record["support"]
        all_a = item_counts[a]
        all_b = item_counts[b]
        both = supp * n
        f11 = int(both)
        f10 = int(all_a - both)
        f01 = int(all_b - both)
        f00 = int(n - (f11 + f10 + f01))
        rules.append({"a": a, "b": b, "ct": (f11, f10, f01, f00), "support": supp})
        rules.append({"a": b, "b": a, "ct": (f11, f01, f10, f00), "support": supp})
    rule_df = pd.DataFrame(rules)
    return rule_df


def rank_association_rules(mined_rules_df, score_fn, score_name="score"):
    """
    Ranks association rules according to a given score function.
    params:
        mined_rules_df: Pandas dataframe of association rules to rank, with columns:
            "a": antecedent (LHS) item index
            "b": consequent (RHS) item index
            "ct": tuple of contingency table entries for the rule
        score_fn(f11, f10, f01, f11): function for scoring rules, takes the entries
            of the contingency table as parameters
        score_name: label to name column with result of the score function
    returns:
        rule_df: copy of mined_rules_df, with the additional column:
            score_name: result of the score function
            Sorted in descending order by score_name
    """
    rule_df = pd.DataFrame(mined_rules_df.copy())
    rule_df[score_name] = rule_df["ct"].apply(lambda ct: score_fn(*ct))
    return rule_df.sort_values(by=score_name, ascending=False)


"""
RECOMMENDATION ALGORITHMS
"""


def collaborative_filter(s_train, s_input, j=3, sim_fn=None, threshold=0.01, k=10):
    """
    Collaborative filtering recommender system.
    params:
        s_train: list of sets of liked item indices for train data
        s_input: list of sets of liked item indices for input data
        j: number of similar users to base recommendations on
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
        threshold: minimum similarity required to consider a similar user
        k: number of items to recommend for each user
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    if sim_fn is None:
        raise ValueError("Must specify a similarity function.")
    check_list_of_sets(s_train, "s_train")
    check_list_of_sets(s_input, "s_input")
    recs_pred = []
    for src in s_input:
        users = []
        for vec in s_train:
            sim = sim_fn(src, vec)
            if sim >= threshold:
                users.append((sim, vec))
        j_users = min(len(users), j)
        if j_users > 0:
            top_users = sorted(users, key=lambda p: p[0], reverse=True)
            sim_user_sets = [vec for (sim, vec) in top_users[0:j_users]]
            opts = dict()
            for user_set in sim_user_sets:
                for item in user_set:
                    if item not in src:
                        if item not in opts:
                            opts[item] = 0
                        opts[item] += 1
            ranks = []
            for item in opts:
                ranks.append((item, opts[item] / len(sim_user_sets)))
            top_ranks = sorted(ranks, key=lambda p: p[1], reverse=True)
            k_recs = min(len(top_ranks), k)
            recs = top_ranks[0:k_recs]
            recs_pred.append(recs)
        else:
            recs_pred.append([])
    return recs_pred


def content_filter(items_train, s_input, sim_fn=None, threshold=0.01, k=10):
    """
    Content-based filtering recommender system.
    params:
        items_train: list of sets of non-zero attribute indices for items
        s_input: list of sets of liked item indices for input data
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
        threshold: minimum similarity required to consider a similar item
        k: number of items to recommend for each user
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    if sim_fn is None:
        raise ValueError("Must specify a similarity function.")
    check_list_of_sets(items_train, "items_train")
    check_list_of_sets(s_input, "s_input")
    recs_pred = []
    for src in s_input:
        sim_items = []
        for item_new_idx, item_new in enumerate(items_train):
            total_sim = 0
            for item_liked_idx in src:
                item_liked = items_train[item_liked_idx]
                sim = sim_fn(item_new, item_liked)
                total_sim += sim
            mean_sim = total_sim / len(src)
            if mean_sim > 0:
                sim_items.append((item_new_idx, mean_sim))
        top_ranks = sorted(sim_items, key=lambda p: p[1], reverse=True)
        k_recs = min(len(top_ranks), k)
        recs = top_ranks[0:k_recs]
        recs_pred.append(recs)
    return recs_pred


def svd_filter(m_train, m_input, n_factors=2, baselines=None, threshold=0.01, k=10):
    """
    Matrix factorization recommender system using Singular Value Decomposition (SVD).
    params:
        m_train: matrix of train data, rows = users, columns = items, 1 = like, 0 otherwise
        m_input: matrix of input data, rows = users, columns = items, 1 = like, 0 otherwise
        n_factors: number of latent factors to estimate (default: 2)
        baselines: numpy array of values to fill empty input entries, array length
            equal to the number of columns in `m_train` and `m_input` (default: None)
        threshold: minimum score to qualify as a recommended item
        k: number of items to recommend for each user
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
        (u, s, vt): tuple of matrix factors produced by SVD:
            u: latent user matrix, rows = users, columns = latent factors
            s: array of latent factor weights
            vt: transposed latent item matrix, rows = latent factors, columns = items
            To estimate the matrix, compute: `B = u.dot(np.diag(s)).dot(vt)`
    """
    m_input_filled = m_input
    if baselines is not None:
        m_input_filled = m_input + np.vstack([baselines for i in range(len(m_input))])
        m_input_filled = np.clip(m_input_filled, a_min=0, a_max=1)
    m_all = np.vstack([m_train, m_input_filled])
    A = csc_matrix(m_all, dtype=float)
    u, s, vt = svds(A, k=n_factors)
    B = u.dot(np.diag(s)).dot(vt)
    s_b_test = []
    row_base = len(m_train)
    for row_add in range(len(m_input)):
        inp = m_input[row_add]
        row_id = row_base + row_add
        row = B[row_id]
        rec_scores = []
        for col, (score, orig) in enumerate(zip(row, inp)):
            if orig < 1:
                if score >= threshold:
                    rec_scores.append((col, score))
        ranks = sorted(rec_scores, key=lambda p: p[1], reverse=True)
        s_b_test.append(ranks[0:k])
    return s_b_test, (u, s, vt)


def association_filter(rule_df, m_train, s_input, score_fn=None, min_score=0.01, k=10):
    """
    Association rule recommender system using frequent itemsets.
    params:
        rule_df: Pandas dataframe of association rules to use, with columns:
            "a": antecedent (LHS) item index
            "b": consequent (RHS) item index
            "ct": tuple of contingency table entries for the rule
        m_train: matrix of train data, rows = users, columns = items, 1 = like, 0 otherwise
        s_input: list of sets of liked item indices for input data
        score_fn(f11, f10, f01, f11): function for scoring rules, takes the entries
            of the contingency table as parameters
        min_score: minimum result from score function required to use an association rule
        k: number of items to recommend for each user
    returns:
        top_rules_df: copy of rule_df, with the additional column:
            "score": result of the score function
            Sorted in descending order by "score"
    """
    if score_fn is None:
        raise ValueError("Must specify a scoring function for association rules.")
    score_name = "score"
    ranked_rules = rank_association_rules(
        rule_df, score_fn=score_fn, score_name=score_name
    )
    top_rules_df = ranked_rules.query("{} >= {}".format(score_name, min_score))
    rule_records = top_rules_df.to_dict(orient="records")
    all_recs = []
    for likes in s_input:
        rec_map = {}
        for rule in rule_records:
            if rule["a"] in likes and rule["b"] not in likes:
                if rule["b"] not in rec_map:
                    rec_map[rule["b"]] = 0
                rec_map[rule["b"]] += rule[score_name]
        ranks = sorted(rec_map.items(), key=lambda p: p[1], reverse=True)
        all_recs.append(ranks[0:k])
    return all_recs, top_rules_df


def weighted_hybrid(components, use_ranks=False, k=10):
    """
    Hybrid recommender system using weights.
    params:
        components: list of tuples where each tuple has (recs list, weight)
            where recs_list is a list of tuples where each tuple has
            (item index, relevance score) and weight is the factor used to
            scale the relevance scores for this component
        use_ranks: boolean (default: False), if True, apply weights to the
            inverse rank of each item instead of its score, for example:
            the first recommended item will earn 1 * weight, the second will
            earn 1/2 * weight, the third will earn 1/3 * weight and so on
        k: number of items to recommend for each user
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    if len(components) < 1:
        raise ValueError("Must provide at least one component.")
    n_users = len(components[0][0])
    user_recs = []
    user_item_maps = [{} for i in range(n_users)]
    for (recs_list, weight) in components:
        if len(recs_list) != n_users:
            raise ValueError(
                "Length of components do not match. Should be equal to number of users."
            )
        for user_id, recs in enumerate(recs_list):
            item_map = user_item_maps[user_id]
            for i, (item, score) in enumerate(recs):
                if item not in item_map:
                    item_map[item] = 0
                if use_ranks:
                    rank = i + 1
                    inv_rank = 1 / rank
                    item_map[item] += weight * inv_rank
                else:
                    item_map[item] += weight * score
    for item_map in user_item_maps:
        item_scores = item_map.items()
        ranks = sorted(item_scores, key=lambda p: p[1], reverse=True)
        top_recs = ranks[0:k]
        user_recs.append(top_recs)
    return user_recs
