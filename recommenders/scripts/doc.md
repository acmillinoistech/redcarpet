# Red Carpet

Module for recommender systems using binary data.

## Helper Methods

### nonzero_index_set(arr)

<pre>
    Returns a set of indices corresponding to non-zero
    entries in a numpy array (or other list-like).
</pre>

### mat_to_sets(mat)

<pre>
    Converts a numpy matrix into a list of sets of column
    indices corresponding to non-zero row entries.
</pre>

### get_recs(user_recs, k=None)

<pre>
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
</pre>

### write_kaggle_recs(recs_list, filename=None, headers=["Id", "Predicted"])

<pre>
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
</pre>


### download_kaggle_recs(recs_list, filename=None, headers=["Id", "Predicted"])

<pre>
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
</pre>

## Evaluation Metrics


### mapk_score(s_hidden, recs_pred, k=10)

<pre>
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
</pre>


### uhr_score(s_hidden, recs_pred, k=10)

<pre>
    Computes the user hit rate (UHR) score of recommendations.
    UHR = the fraction of users whose top list included at
    least one item also in their hidden set.
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        float, range [0, 1]
</pre>


### get_apk_scores(s_hidden, recs_pred, k=10)

<pre>
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
</pre>


### get_hit_counts(s_hidden, recs_pred, k=10)

<pre>
    Returns the number of successful recommendations for each user.
    params:
        s_hidden: list of sets of hidden items for each user
        recs_pred: list of lists of recommended items, with each list
        k: number of recommendations to use in top set
    returns:
        list of integers, each integer in the range [0, k]
</pre>


### get_all_scores(rec_scores, k=10)

<pre>
    Get scores of all items in the list of lists of recommendations.
</pre>


## Analysis Tools


### show_apk_dist(s_hidden, models, k=10, bin_size=0.1)

<pre>
    Plot a histogram of average precision scores for all users.
</pre>


### show_hit_dist(s_hidden, models, k=10)

<pre>
    Plot a histogram of hit counts for all users.
</pre>


### show_score_dist(models, k=10, bins=None)

<pre>
    Plot a histogram of item recommendation scores for all users.
</pre>


### show_user_detail(s_input, s_hidden, rec_scores, uid, name_fn=None, k=10)

<pre>
    Show the detailed results of recommendations to a user.
</pre>


### show_user_recs(s_hidden, rec_scores, k=10)

<pre>
    Show a table of recommendation results by user.
</pre>


### show_item_recs(s_hidden, rec_scores, k=10)

<pre>
    Show a table of recommendation results by item.
</pre>


## Similarity Measures


### jaccard_sim(u, v)

<pre>
    Computes the Jaccard similarity between sets u and v.
    sim = intersection(u, v) / union(u, v)
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
</pre>


### cosine_sim(u, v)

<pre>
    Computes the Cosine similarity between sets u and v.
    sim = intersection(u, v) / sqrt(|u| * |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
</pre>


### forbes_sim(u, v)

<pre>
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
</pre>


### mcconnoughy_sim(u, v)

<pre>
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
</pre>


### simpson_sim(u, v)

<pre>
    Computes the Simpson similarity coefficient between sets u and v.
    sim = intersection(u, v) / min(|u|, |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
</pre>


### first_kulczynski_sim(u, v)

<pre>
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
</pre>


### second_kulczynski_sim(u, v)

<pre>
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
</pre>


### sorenson_dice_sim(u, v)

<pre>
    Computes the Sørensen-Dice similarity coefficient between sets u and v.
    sim = (2 * intersection(u, v)) / (|u| + |v|)
    Where |s| is the number of items in set s
    params:
        u, v: sets to compare
    returns:
        float between 0 and 1, where 1 represents perfect
            similarity and 0 represents no similarity
</pre>


## Association Rule Metrics


### sets_to_contingency(a, b, N)

<pre>
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
</pre>


### rule_support(f11, f10, f01, f00)

<pre>
    Computes the support for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum support and
            0 indicates no support
</pre>


### rule_confidence(f11, f10, f01, f00)

<pre>
    Computes the confidence for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum confidence and
            0 indicates no confidence
</pre>


### rule_lift(f11, f10, f01, f00)

<pre>
    Computes the lift for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float ranging from zero to infinity where 1 implies independence, greater
        than 1 implies positive association, less than 1 implies negative association
</pre>


### rule_conviction(f11, f10, f01, f00)

<pre>
    Computes the conviction for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1]
</pre>


### rule_power_factor(f11, f10, f01, f00)

<pre>
    Computes the rule power factor (RPF) for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1]
</pre>


### rule_interest_factor(f11, f10, f01, f00)

<pre>
    Computes the interest factor for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum interest and
            0 indicates no interest
</pre>


### rule_phi_correlation(f11, f10, f01, f00)

<pre>
    Computes the phi correlation for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [-1, 1] where 1 indicates perfect positive correlation and
            -1 indicates perfect negative correlation.
</pre>


### rule_is_score(f11, f10, f01, f00)

<pre>
    Computes the IS score for a rule `a -> b` based on the contingency table.
    params:
        f11 = count a and b appearing together
        f10 = count of a appearing without b
        f01 = count of b appearing without a
        f00 = count of neither a nor b appearing
    returns:
        float in range [0, 1] where 1 indicates maximum int3erest and
            0 indicates no int3erest
</pre>


### mine_association_rules(m_train, min_support=0.5)

<pre>
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
</pre>


### rank_association_rules(mined_rules_df, score_fn, score_name="score")

<pre>
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
</pre>


## Recommendation Algorithms


### collaborative_filter(s_train, s_input, j=3, sim_fn=None, threshold=0.01, k=10)

<pre>
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
</pre>


### content_filter(items_train, s_input, sim_fn=None, threshold=0.01, k=10)

<pre>
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
</pre>


### svd_filter(m_train, m_input, n_factors=2, baselines=None, threshold=0.01, k=10)

<pre>
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
</pre>


### association_filter(rule_df, m_train, s_input, score_fn=None, min_score=0.01, k=10)

<pre>
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
</pre>


### weighted_hybrid(components, use_ranks=False, k=10)

<pre>
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
</pre>
