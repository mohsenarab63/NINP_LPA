from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def evaluate(ground_truth, pred_communities):
    # 1) Build a mapping from node to ground-truth community label.
    # ground_truth is expected to be a list of communities (e.g., list of sets/lists),
    # where each community contains node IDs.
    true_label_dict = {}
    for label, community in enumerate(ground_truth):
        for node in community:
            true_label_dict[node] = label

    # 2) Build a mapping from node to predicted community label.
    # pred_communities has the same expected format as ground_truth.
    pred_label_dict = {}
    for label, community in enumerate(pred_communities):
        for node in community:
            pred_label_dict[node] = label

    # 3) Determine the set of nodes that appear in both ground truth and prediction.
    # Only the intersection is used for evaluation to avoid missing-node issues.
    nodes_in_gt = set(true_label_dict.keys())
    nodes_in_pred = set(pred_label_dict.keys())
    nodes_eval = sorted(nodes_in_gt & nodes_in_pred)

    # 4) Construct aligned label vectors for metric computation.
    # The same node order (nodes_eval) is used for both y_true and y_pred.
    y_true = [true_label_dict[n] for n in nodes_eval]
    y_pred = [pred_label_dict[n] for n in nodes_eval]

    # 5) Compute clustering quality metrics:
    # NMI: Normalized Mutual Information
    # ARI: Adjusted Rand Index
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return nmi, ari