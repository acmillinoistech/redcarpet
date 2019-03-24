import numpy as np
import ml_metrics


"""
redcarpet: Module for recommender systems using sets
"""

"""
HELPER METHODS
"""


def nonzero_index_set(arr):
    """
    Returns a set of  indices corresponding to non-zero
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


"""
EVALUATION METRICS
"""


def mapk_score(recs_true, recs_pred, k=10):
    """
    Computes the mean average precision at k (MAP@K) of recommendations.
    MAP@K = mean AP@K score over all users
    AP@K = (1 / m) * sum from 1 to k of (precision at i * relevance of ith item)
    Where m is the number of items in a user's hidden set
    Where k is the number of items recommended to each user
    params:
        recs_true: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        float, range [0, 1], though score of 1 will be impossible if
        recs_true includes users who have more than k hidden items
    """
    return ml_metrics.mapk(recs_true, recs_pred, k)


def uhr_score(recs_true, recs_pred, k=10):
    """
    Computes the user hit tate (UHR) score of recommendations.
    UHR = the fraction of users whose top list included at
    least one item also in their hidden set.
    params:
        recs_true: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        float, range [0, 1]
    """
    if len(recs_true) != len(recs_pred):
        note = "Length of true list {} does not match length of recommended list {}."
        raise ValueError(note.format(len(recs_true), len(recs_pred)))
    scores = []
    for r_true, r_pred_orig in zip(recs_true, recs_pred):
        r_pred = list(r_pred_orig)[0:k]
        intersect = set(r_true).intersection(set(r_pred))
        scores.append(1 if len(intersect) > 0 else 0)
    return np.mean(scores)


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


"""
RECOMMENDATION ALGORITHMS
"""


def collaborative_filter(recs_train, recs_input, k=10, j=3, sim_fn=jaccard_sim):
    """
    Collaborative filtering recommender system.
    params:
        recs_train: list of sets of liked item indices for train data
        recs_input: list of sets of liked item indices for input data
        k: number of items to recommend for each user
        j: number of similar users to base recommendations on
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    recs_pred = []
    for src in recs_input:
        users = []
        for vec in recs_train:
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


def content_filter(items_train, recs_input, k=10, sim_fn=jaccard_sim):
    """
    Content-based filtering recommender system.
    params:
        items_train: list of sets of non-zero attribute indices for items
        recs_input: list of sets of liked item indices for input data
        k: number of items to recommend for each user
        sim_fn(u, v): function that returns a float value representing
            the similarity between sets u and v
    returns:
        recs_pred: list of lists of tuples of recommendations where
            each tuple has (item index, relevance score) with the list
            of tuples sorted in order of decreasing relevance
    """
    recs_pred = []
    for src in recs_input:
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


def weighted_hybrid(components, k=10):
    """
    Hybrid recommender system using weights.
    params:
        components: list of tuples where each tuple has (recs list, weight)
            where recs_list is a list of tuples where each tuple has
            (item index, relevance score) and weight is the factor used to
            scale the relevance scores for this component
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
            for (item, score) in recs:
                if item not in item_map:
                    item_map[item] = 0
                item_map[item] += weight * score
    for item_map in user_item_maps:
        item_scores = item_map.items()
        ranks = sorted(item_scores, key=lambda p: p[1], reverse=True)
        top_recs = ranks[0:k]
        user_recs.append(top_recs)
    return user_recs
