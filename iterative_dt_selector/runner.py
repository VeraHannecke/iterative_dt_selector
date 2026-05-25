import os
import numpy as np
from sklearn.tree import export_text

from .data_loader import prepare_dataset
from .tree_builder import (build_single_tree, get_tree_paths, print_tree_matrix, get_split_features, find_tied_features
)
from .visualization import visualize_matrices, visualize_matrices_2


def run_single_iteration(X, y, max_depth, min_samples_split, min_samples_leaf):
    # build one tree
    features = list(X.columns)

    model, importances, _ = build_single_tree(
        X, y, max_depth, min_samples_split, min_samples_leaf
    )

    best_idx = int(np.argmax(importances))
    best_feat = features[best_idx]
    best_import = float(importances[best_idx])

    pairs = list(zip(features, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_feats = pairs[:10]

    results = {
        'best_feature': best_feat,
        #'accuracy': accuracy,
        'best_importance': best_import,
        'top_features': top_feats,
        'model': model
        #'num_remaining': len(features)
    }
    return results


def print_iteration_results(results, importance_threshold, log):
    top3 = results['top_features'][:3]

    top3_str = []
    for feat, imp in top3:
        if importance_threshold and imp >= importance_threshold:
            top3_str.append(f"{feat} ({imp:.4f}) [BRANCH]")
        else:
            top3_str.append(f"{feat} ({imp:.4f})")
    while len(top3_str) < 3:
        top3_str.append("-")

    log(f"Top 3: {top3_str[0]:<38} {top3_str[1]:<38} {top3_str[2]:<38}")


# toggle: branch on features that tie with chosen split features
# large branching on binary data
BRANCH_ON_TIED = True


def build_branch(X, y, max_depth, min_samples_split, min_samples_leaf,
                 importance_threshold, n_iterations, depth,
                 rm_path, tree_counter, leaf_counter, log, visited=None):
    # build one tree and recursively spawn children for each qualifying split feature
    # visited:frozenset(rm_path) already explored

    if visited is None:
        visited = set()

    rm_key = frozenset(rm_path)
    if rm_key in visited:
        return None
    visited.add(rm_key)

    if X.shape[1] == 0 or depth > n_iterations:
        return None

    path_str = " -> ".join(rm_path) if rm_path else "root"
    log("\n" + "=" * 100)
    log(f"Branch [{path_str}]  |  Depth {depth}")
    log("=" * 100)

    results = run_single_iteration(X, y, max_depth, min_samples_split, min_samples_leaf)
    tree_counter[0] += 1
    leaf_counter[0] += results['model'].tree_.n_leaves

    print_iteration_results(results, importance_threshold, log)

    tree_text = export_text(results['model'], feature_names=list(X.columns),
                            max_depth=max_depth if max_depth is not None else 6)
    log("\nTREE STRUCTURE:")
    log(tree_text)
    print_tree_matrix(results['model'], list(X.columns), log)


    if importance_threshold is not None and results['best_importance'] < importance_threshold:
        log(f"Best importance ({results['best_importance']:.4f}) below threshold ({importance_threshold}). Branch stops.")
        return None

    paths, feat_order = get_tree_paths(results['model'], list(X.columns))

    binary_feats = set()
    for col in X.columns:
        if set(X[col].unique()).issubset({0.0, 1.0}):
            binary_feats.add(col)

    # which samples reach which leaf
    path_samples = {}
    for path_conds, cls, n_samples in paths:
        key = frozenset(path_conds)
        if key in path_samples:
            continue

        mask = np.ones(len(X), dtype=bool)
        for featname, val, direction in path_conds:
            col_vals = X[featname].values
            if featname in binary_feats:
                mask &= col_vals == float(val)
            elif direction == '↓':
                mask &= col_vals <= val
            else:
                mask &= col_vals > val
        path_samples[key] = [str(idx) for idx in X.index[mask]]

    node = {
        'removed_feature': rm_path[-1] if rm_path else None,
        'depth': depth,
        'removed_path': list(rm_path),
        'results': results,
        'paths': paths,
        'feature_order': feat_order,
        'binary_features': binary_feats,
        'path_samples': path_samples,
        'children': []
    }

    cutoff = importance_threshold if importance_threshold is not None else 0.0
    branch_feats = get_split_features(results['model'], list(X.columns),
        results['model'].feature_importances_, cutoff)

    if BRANCH_ON_TIED:
        tied = find_tied_features(results['model'], X, y, list(X.columns), cutoff)

        existing = {f for f, _ in branch_feats}
        for feat, imp in tied:
            if feat not in existing:
                branch_feats.append((feat, imp))

    for feat, imp in branch_feats:
        c_key = frozenset(rm_path + [feat])
        if c_key in visited:
            continue
        X_child = X.drop(columns=[feat])
        child = build_branch(
            X_child, y, max_depth, min_samples_split, min_samples_leaf,
            importance_threshold, n_iterations,
            depth + 1, rm_path + [feat], tree_counter, leaf_counter, log, visited
        )
        if child is not None:
            node['children'].append(child)

    return node


def run_iterative_feature_selection(
    file_path,
    threshold=-60,
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=2,
    importance_threshold=0.19,
    num_iterations=10,
    overlap_threshold=0.30,
    output_dir=".",
    log_file=None,
    class1_label="responder",
    class0_label="non-responder"
):

    output_dir = os.path.normpath(output_dir)
    log_f = None

    if log_file:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file)
        log_f = open(log_path, 'w', encoding='utf-8')

    def log(msg=""):
        try:
            print(msg)
        except Exception:
            print(str(msg).encode('ascii', errors='replace').decode('ascii'))
        if log_f:
            log_f.write(str(msg) + '\n')

    try:
        X, y = prepare_dataset(file_path, threshold)
        counts = np.bincount(y)
        n0 = int(counts[0])
        n1 = int(counts[1])

        log("=" * 100)
        log("Dataset Summary:")
        log(f"  Number of samples: {X.shape[0]}")
        log(f"  Number of features: {X.shape[1]}")
        log(f"  Classification threshold: {threshold}")
        if len(counts) > 1:
            log(f"  Class 1 ({class1_label}): {counts[1]}  |  Class 0 ({class0_label}): {counts[0]}")

        depth_str = str(max_depth) if max_depth is not None else "unlimited"
        imp_str = str(importance_threshold) if importance_threshold is not None else "disabled"
        log(f"Configuration: max_depth={depth_str}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        log(f"               importance_threshold={imp_str}, max_branch_depth={num_iterations}")
        log("=" * 100)

        tree_counter = [0]
        leaf_counter = [0]

        root = build_branch(X, y, max_depth, min_samples_split, min_samples_leaf,
                            importance_threshold, num_iterations,
                            depth=1, rm_path=[], tree_counter=tree_counter, leaf_counter=leaf_counter, log=log)

        log("\n" + "=" * 100)
        log(f"Total trees built: {tree_counter[0]}  |  Total leaves: {leaf_counter[0]}")
        log("=" * 100)

    finally:
        if log_f:
            log_f.close()

    visualize_matrices(
        root,
        threshold=threshold,
        overlap_threshold=overlap_threshold,
        output_dir=output_dir,
        class1_label=class1_label,
        class0_label=class0_label,
        class1_n=n1,
        class0_n=n0
    )

    visualize_matrices_2(root,
        threshold=threshold,
        overlap_threshold=overlap_threshold,
        output_dir=output_dir,
        class1_label=class1_label,
        class0_label=class0_label,
        class1_n=n1,
        class0_n=n0)

    return root
