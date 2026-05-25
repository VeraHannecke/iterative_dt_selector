import numpy as np
from sklearn.tree import DecisionTreeClassifier


def build_single_tree(X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeClassifier(max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0  # fixed seed 
    )
    model.fit(X, y)

    # DEBUG only
    # tree_text = export_text(
    #     model,
    #     features=list(X.columns),
    #     max_depth=6  # display limit, not tree
    # )
    # print("\n TREE STRUCTURE:")
    # print(tree_text)
    accuracy = model.score(X, y)

    importances = model.feature_importances_

    return model, importances, accuracy


def get_tree_paths(model, features):
    # feat order = left first DFS
    tree = model.tree_

    paths = []
    feat_order = []

    def traverse(node_id, path_current):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right: 
            counts = tree.value[node_id][0]
            pred_class = int(np.argmax(counts))
            n = int(tree.n_node_samples[node_id])
            paths.append((list(path_current), pred_class, n))
            return

        index = tree.feature[node_id]
        featname = features[index]
        thresh = tree.threshold[node_id]

        if featname not in feat_order:
            feat_order.append(featname)

        #binary feats always split at 0.5
        # not ideal but works for data for now
        if thresh == 0.5:
            leftv = 0
            rightv = 1
        else:
            leftv = thresh
            rightv = thresh

        path_current.append((featname, leftv, '↓'))
        traverse(left, path_current)
        path_current.pop()

        path_current.append((featname, rightv, '↑'))
        traverse(right, path_current)
        path_current.pop()

    traverse(0, [])
    return paths, feat_order


def print_tree_matrix(model, features, log=print):
    paths, feat_order = get_tree_paths(model, features)

    if not paths:
        return

    widths = []
    for feat in feat_order:
        widths.append(len(feat) + 2)

    header = "  "
    for feat, w in zip(feat_order, widths):
        header += f"{'[' + feat + ']':<{w + 2}}"
    header += f"{'Result':<12}Samples"
    separator = "  " + "-" * (len(header) - 2)

    c1_paths = []

    c0_paths = []
    for p, c, n in paths:
        if c == 1:
            c1_paths.append((p, c, n))
        else:
            c0_paths.append((p, c, n))

    for label, class_paths in [("Class 1", c1_paths), ("Class 0", c0_paths)]:
        log(f"\n  Tree Matrix - {label} paths:")
        log(header)
        log(separator)

        for path, pred_class, n in class_paths:
            path_dict = {}
            for feat, val, direction in path:
                path_dict[feat] = (val, direction)
            row = "  "

            for feat, w in zip(feat_order, widths):
                if feat in path_dict:
                    val, direction = path_dict[feat]
                    if isinstance(val, float) and val not in (0.0, 1.0):
                        cell = f"{feat}{direction}{val:.2f}"
                    else:
                        cell = f"{feat}={'present' if int(val) == 1 else 'absent'}"
                else:
                    cell = "---"
                row += f"{cell:<{w + 2}}"

            row += f"{'class ' + str(pred_class):<12}{n}"
            log(row)

    log("")


#

def get_split_features(model, features, importances, threshold):
    # split features + importance über threshold
    tree = model.tree_
    used = set()

    for nodeid in range(tree.node_count):
        if tree.children_left[nodeid] != tree.children_right[nodeid]:
            used.add(features[tree.feature[nodeid]])

    result = []
    for feat, imp in zip(features, importances):
        if feat in used and imp >= threshold:
            result.append((feat, imp))
    return result


def _impurity_decrease(X_node, y_node, feat_idx):
    n = len(y_node)
    if n == 0:
        return 0.0

    values = X_node[:, feat_idx]
    vals = np.unique(values)

    if len(vals) <= 1:
        return 0.0

    thresholds = (vals[:-1] + vals[1:]) / 2

    # parent gini
    p_counts = np.bincount(y_node, minlength=2)
    p_probs = p_counts / n
    gini_p = 1.0 - float(np.sum(p_probs ** 2))

    best = 0.0

    for thresh in thresholds:
        left_mask = values <= thresh
        n_left = left_mask.sum()
        n_right = n - n_left
        if n_left == 0 or n_right == 0:
            continue

        l_counts = np.bincount(y_node[left_mask], minlength=2)
        l_probs = l_counts / n_left
        gini_l = 1.0 - float(np.sum(l_probs ** 2))


        r_counts = np.bincount(y_node[~left_mask], minlength=2)
        r_probs = r_counts / n_right
        gini_r = 1.0 - float(np.sum(r_probs ** 2))

        decrease = gini_p - (n_left / n) * gini_l - (n_right / n) * gini_r
        if decrease > best:
            best = decrease

    return best


def find_tied_features(model, X, y, features, threshold, tol=1e-7):
    # same gini, not picked bc of seed
    tree = model.tree_
    X_arr = X.values
    y_arr = np.asarray(y)

    already_split = set()
    for node in range(tree.node_count):
        if tree.children_left[node] != tree.children_right[node]:
            already_split.add(features[tree.feature[node]])

    n_ind = model.decision_path(X)
    tied = {}

    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            continue

        sample = n_ind[:, node].nonzero()[0]
        X_node = X_arr[sample]
        y_node = y_arr[sample]

        chosen_index = tree.feature[node]
        chosen_decrease = _impurity_decrease(X_node, y_node, chosen_index)

        if chosen_decrease < threshold:
            continue

        for feat_i, feat_name in enumerate(features):
            if feat_name in already_split or feat_i == chosen_index:
                continue
            decrease = _impurity_decrease(X_node, y_node, feat_i)
            if abs(decrease - chosen_decrease) <= tol:
                tied[feat_name] = max(tied.get(feat_name, 0.0), decrease)

    out = []
    for f, imp in tied.items():
        if imp >= threshold:
            out.append((f, imp))
    return out
