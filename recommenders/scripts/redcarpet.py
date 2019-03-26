import numpy as np
import pandas as pd
import ml_metrics
import base64
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
    Computes the user hit tate (UHR) score of recommendations.
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
RECOMMENDATION ALGORITHMS
"""


def collaborative_filter(s_train, s_input, k=10, j=3, sim_fn=jaccard_sim):
    """
    Collaborative filtering recommender system.
    params:
        s_train: list of sets of liked item indices for train data
        s_input: list of sets of liked item indices for input data
        k: number of items to recommend for each user
        j: number of similar users to base recommendations on
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    check_list_of_sets(s_train, "s_train")
    check_list_of_sets(s_input, "s_input")
    recs_pred = []
    for src in s_input:
        users = []
        for vec in s_train:
            sim = sim_fn(src, vec)
            if sim > 0:
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
                count = opts[item]
                score = count / len(sim_user_sets)
                ranks.append((item, score))
            top_ranks = sorted(ranks, key=lambda p: p[1], reverse=True)
            k_recs = min(len(top_ranks), k)
            recs = top_ranks[0:k_recs]
            recs_pred.append(recs)
        else:
            recs_pred.append([])
    return recs_pred


def content_filter(items_train, s_input, k=10, sim_fn=jaccard_sim):
    """
    Content-based filtering recommender system.
    params:
        items_train: list of sets of non-zero attribute indices for items
        s_input: list of sets of liked item indices for input data
        k: number of items to recommend for each user
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
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


def svd_filter(m_train, m_input, n_factors=2, baselines=None, threshold=0, k=10):
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


def weighted_hybrid(components, k=10, use_ranks=False):
    """
    Hybrid recommender system using weights.
    params:
        components: list of tuples where each tuple has (recs list, weight)
            where recs_list is a list of tuples where each tuple has
            (item index, relevance score) and weight is the factor used to
            scale the relevance scores for this component
        k: number of items to recommend for each user
        use_ranks: boolean (default: False), if True, apply weights to the
            inverse rank of each item instead of its score, for example:
            the first recommended item will earn 1 * weight, the second will
            earn 1/2 * weight, the third will earn 1/3 * weight and so on
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
