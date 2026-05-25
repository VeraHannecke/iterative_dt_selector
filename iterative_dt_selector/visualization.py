import os
import altair as alt
import pandas as pd


REMOVE_DUPLICATE_PATHS = True

# toggle split into multiple plots when >20 feats
# if False: one plot, first 20 feats get color rest shown without colour
SPLIT_20 = True


def make_heatmap(df, feat_order, title, class_n=None):
    # single heatmap for one class of one group

    if df.empty:
        return alt.Chart(pd.DataFrame({'text': [f'{title}: no paths']})).mark_text().encode(
            text='text:N'
        ).properties(title=title, width=120, height=60)

    gene_order = []
    for g in feat_order:
        if g in df['gene'].values:
            gene_order.append(g)

    tooltips = [alt.Tooltip('gene:N', title='feature'), 'path:N', 'value:Q', 'samples:Q']
    if 'branch' in df.columns:
        tooltips.append('branch:N')

    x_enc = alt.X('gene:N', sort=gene_order, scale=alt.Undefined, title='Gene / Feature',
                  axis=alt.Axis(labelAngle=-45))
    y_enc = alt.Y('path:N', sort=alt.EncodingSortField(field='samples', order='descending'), title='Tree path')

    chart_width = alt.Step(45)

    if 'is_binary' in df.columns:
        bin_df = df[df['is_binary']]
        cont_df = df[~df['is_binary']]
    else:
        bin_df = df[df['value'].isin([0.0, 1.0])]
        cont_df = df[~df['value'].isin([0.0, 1.0])]


    layers = []

    if not bin_df.empty:
        bin_layer = alt.Chart(bin_df).transform_filter(
            'datum.samples >= min_n'
        ).mark_rect(stroke='white', strokeWidth=1).encode(
            x=x_enc, y=y_enc,
            color=alt.Color('value:Q',
                scale=alt.Scale(domain=[0, 1], range=['#e0e0e0', '#c0392b']),
                legend=alt.Legend(title='Split value', values=[0, 1], orient='left')
            ),
            tooltip=tooltips
        ).properties(title=title, width=chart_width, height=alt.Step(30))
        layers.append(bin_layer)

    if not cont_df.empty:
        cont_min = float(cont_df['value'].min())
        cont_max = float(cont_df['value'].max())

        cont_layer = alt.Chart(cont_df).transform_filter(
            'datum.samples >= min_n'
        ).mark_rect(stroke='white', strokeWidth=1).encode(
            x=x_enc, y=y_enc,
            color=alt.Color('value:Q',
                scale=alt.Scale(domain=[cont_min, cont_max], scheme='viridis'),
                legend=alt.Legend(title='Split value', orient='left')
            ),
            tooltip=tooltips
        ).properties(title=title if not layers else '', width=chart_width, height=alt.Step(30))
        layers.append(cont_layer)

        if 'direction' in cont_df.columns:
            dir_text = alt.Chart(cont_df).transform_filter(
                'datum.samples >= min_n'
            ).transform_calculate(
                cmp_sym="datum.direction === '↓' ? '<=' : '>'"
            ).mark_text(fontSize=9, color='white').encode(
                x=x_enc, y=y_enc,
                text=alt.Text('cmp_sym:N')
            )
            layers.append(dir_text)

    if len(layers) > 1:
        base = alt.layer(*layers).resolve_scale(color='independent')
    else:
        base = layers[0]

    n_col = alt.Chart(df).transform_filter(
        'datum.samples >= min_n'
    ).transform_aggregate(
        n='max(samples)', groupby=['path']
    ).mark_text(align='center', baseline='middle').encode(
        y=alt.Y('path:N', sort=alt.EncodingSortField(field='n', order='descending'), axis=None),
        text=alt.Text('n:Q', format='d')
    ).properties(title=f'n/{class_n}' if class_n is not None else 'n', width=35, height=alt.Step(30))

    if 'sample_names' in df.columns:
        path_df = df[['path', 'sample_names', 'samples']].drop_duplicates(subset=['path'])
        samples_col = alt.Chart(path_df).transform_filter(
            'datum.samples >= min_n'
        ).mark_text(align='left', baseline='middle', fontSize=10).encode(
            y=alt.Y('path:N', sort=alt.EncodingSortField(field='samples', order='descending'), axis=None),
            text=alt.Text('sample_names:N')
        ).properties(title='Samples', width=250, height=alt.Step(30))
        return alt.hconcat(base, n_col, samples_col)

    return alt.hconcat(base, n_col)


def save_chart(chart, output_file, title, threshold, class1_label, class0_label):
    html = chart.to_html()

    legend = f"""
    <div style="font-family: sans-serif; padding: 16px 0 12px 0;">
        <div style="color:#555; margin-bottom:4px;">
            Class 1: regression &le; {threshold} ({class1_label})
            <span style="display:inline-block; width:48px;"></span>
            Class 0: regression &gt; {threshold} ({class0_label})
        </div>
        <div style="color:#555; margin-bottom:4px;">
            n = number of samples at the leaf &nbsp;|&nbsp; path label: d=branch depth, p=path index within tree
        </div>
        <div style="color:#555;">
            &lt;= = feature value below split threshold &nbsp;|&nbsp; &gt; = feature value above split threshold
        </div>
    </div>
    """

    html_title = f'<h2 style="font-family:sans-serif; margin:16px 0 0 0;">{title}</h2>'

    #fix the slider so it's always visible
    slider_css = """<style>
    .vega-bindings {
        position: fixed;
        top: 12px;
        right: 16px;
        z-index: 1000;
        background: white;
        padding: 6px 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-family: sans-serif;
        font-size: 13px;
    }
    </style>"""

    html = html.replace('<body>', f'<body>\n{html_title}\n{legend}\n{slider_css}', 1)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def flatten_tree(root):
    if root is None:
        return []
    nodes = [root]
    for child in root.get('children', []):
        nodes.extend(flatten_tree(child))
    return nodes


def _group_by_overlap(nodes, threshold, class_val=None):
    # collect paths from all nodes, group by jaccard of first group

    path_items = []
    for node in nodes:
        for path, cv, n in node['paths']:
            if class_val is not None and cv != class_val:
                continue
            feat_set = []
            for feat, val, direction in path:
                feat_set.append(feat)
            features = frozenset(feat_set)
            path_items.append((path, cv, n, features, node))
    groups = [] 

    for item in path_items:
        path, class_val, n, features, node = item
        placed = False
        for anchor_feats, g_items in groups:
            union = anchor_feats | features
            if not union:
                continue
            jaccard = len(anchor_feats & features) / len(union)
            if jaccard >= threshold:
                g_items.append(item)
                placed = True
                break

        if not placed:
            groups.append((features, [item]))

    result = {}
    for i, (_, items) in enumerate(groups):
        result[f'Group {i + 1}'] = items
    return result


def build_overlap_dataframe(path_items, class_val, gene_order=None):
    # gene pass order so both classes use the same columns per group
    records = []
    seen = set()

    class_items = []
    for p, c, n, f, node in path_items:
        if c == class_val:
            class_items.append((p, c, n, f, node))
    class_items = sorted(class_items, key=lambda x: x[2], reverse=True)

    if gene_order is None:
        gene_order = []
        for _, _, _, _, node in class_items:
            for g in node['feature_order']:
                if g not in gene_order:
                    gene_order.append(g)

    for idx, (path, pred_class, n, features, node) in enumerate(class_items, 1):
        path_key = frozenset(path)
        if REMOVE_DUPLICATE_PATHS and path_key in seen:
            continue
        seen.add(path_key)

        p_dict = {}
        d_dict = {}
        for feat, val, direction in path:
            p_dict[feat] = val
            d_dict[feat] = direction
        depth = node['depth']
        branch_str = ' -> '.join(node['removed_path']) if node['removed_path'] else 'root'
        label = f"d{depth} p{idx}"

        bin_feats = node.get('binary_features', set())
        paths = node.get('path_samples', {})
        sample_names = ', '.join(paths.get(path_key, []))

        for gene in gene_order:
            if gene in p_dict:
                records.append({
                    'gene': gene,
                    'path': label,
                    'value': p_dict[gene],
                    'direction': d_dict[gene],
                    'samples': n,
                    'branch': branch_str,
                    'is_binary': gene in bin_feats,
                    'sample_names': sample_names,
                })

    return pd.DataFrame(records), gene_order


def visualize_matrices(root, threshold=-60, overlap_threshold=0.30, output_dir=".",
                       class1_label="responder", class0_label="non-responder",
                       class1_n=None, class0_n=None):
    # save class 1 and  0 to separate html

    if root is None:
        print("No data to visualize.")
        return

    os.makedirs(output_dir, exist_ok=True)

    nodes = flatten_tree(root)

    overlap = _group_by_overlap(nodes, overlap_threshold)
    c1_data = {}
    c0_data = {}
    samples = []

    for gname, items in overlap.items():
        # comb from both classes so columns match ac plots
        shared_gene_ord = []
        for _, _, _, _, node in items:
            for g in node['feature_order']:
                if g not in shared_gene_ord:
                    shared_gene_ord.append(g)

        df_c1, go_c1 = build_overlap_dataframe(items, class_val=1, gene_order=shared_gene_ord)
        df_c0, go_c0 = build_overlap_dataframe(items, class_val=0, gene_order=shared_gene_ord)
        c1_data[gname] = (df_c1, go_c1)
        c0_data[gname] = (df_c0, go_c0)
        if not df_c1.empty:
            samples.extend(df_c1['samples'].tolist())
        if not df_c0.empty:
            samples.extend(df_c0['samples'].tolist())

    max_n = max(samples)

    n_param = alt.param(
        name='min_n', value=1,
        bind=alt.binding_range(min=1, max=max_n, step=1, name='Min n: ')
    )

    c1_charts = []
    for gn, (df, go) in c1_data.items():
        c1_charts.append(make_heatmap(df, go, f"{gn} - Class 1 ({class1_label})", class_n=class1_n))

    c0_charts = []
    for gn, (df, go) in c0_data.items():
        c0_charts.append(make_heatmap(df, go, f"{gn} - Class 0 ({class0_label})", class_n=class0_n))

    c1 = alt.hconcat(*c1_charts).resolve_scale(color='shared')
    c0 = alt.hconcat(*c0_charts).resolve_scale(color='shared')

    c1 = c1.add_params(n_param).configure_view(stroke=None)
    c0 = c0.add_params(n_param).configure_view(stroke=None)

    out_c1 = os.path.join(output_dir, "tree_matrices_class1.html")
    out_c0 = os.path.join(output_dir, "tree_matrices_class0.html")

    save_chart(c1, out_c1, f'Decision Tree Feature Selection — Class 1 ({class1_label})',
               threshold, class1_label, class0_label)
    save_chart(c0, out_c0,
               f'Decision Tree Feature Selection — Class 0 ({class0_label})',
               threshold, class1_label, class0_label)

    print(f"\nVisualization saved to:")
    print(f"  {out_c1}")
    print(f"  {out_c0}")


#  depth visual

def build_depth_df(path_items, class_val):
    records = []
    seen = set()

    class_items = []
    for p, c, n, f, node in path_items:
        if c == class_val:
            class_items.append((p, c, n, f, node))
    class_items = sorted(class_items, key=lambda x: x[2], reverse=True)

    for idx, (path, pred_class, n, features, node) in enumerate(class_items, 1):
        path_key = frozenset(path)
        if REMOVE_DUPLICATE_PATHS and path_key in seen:
            continue
        seen.add(path_key)

        depth = node['depth']
        branch = ' -> '.join(node['removed_path']) if node['removed_path'] else 'root'
        label = f"d{depth} p{idx}"

        bin_feats = node.get('binary_features', set())
        path_s = node.get('path_samples', {})
        sample_n = ', '.join(path_s.get(path_key, []))

        for pos, (featname, val, direction) in enumerate(path):
            is_binary = featname in bin_feats

            if is_binary:
                text = str(int(val))
            else:
                sym = '<=' if direction == '↓' else '>'
                text = f"{sym} {val:.2f}"

            records.append({
                'depth_pos': pos,
                'path': label,
                'gene': featname,
                'value': float(val),
                'direction': direction,
                'cell_text': text,
                'samples': n,
                'branch': branch,
                'is_binary': is_binary,
                'sample_names': sample_n,
            })

    return pd.DataFrame(records)


def _group_by_feat_limit(nodes, max_features=20):
    # greedy assign each path to the first group if space
    path_items = []
    for node in nodes:
        for path, cv, n in node['paths']:
            feat_set = []
            for feat, val, direction in path:
                feat_set.append(feat)
            features = frozenset(feat_set)
            path_items.append((path, cv, n, features, node))

    groups = [] 

    for item in path_items:
        path, cv, n, features, node = item
        placed = False

        for group_feats, group_i in groups:
            if len(group_feats | features) <= max_features:
                group_feats |= features
                group_i.append(item)
                placed = True
                break

        if not placed:
            groups.append((set(features), [item]))

    result = {}
    for j, (_, items) in enumerate(groups):
        result[f'Group {j + 1}'] = items
    return result


def make_depth_heatmap(df, genes, title, class_n=None):
    if df.empty:
        return alt.Chart(pd.DataFrame({'text': [f'{title}: no paths']})).mark_text().encode(
            text='text:N'
        ).properties(title=title, width=120, height=60)

    tooltips = [alt.Tooltip('gene:N', title='feature'), 'path:N', 'value:Q', 'samples:Q']
    if 'branch' in df.columns:
        tooltips.append('branch:N')

    x_enc = alt.X('depth_pos:O', title='Split depth (0 = root)', axis=alt.Axis(labelAngle=0))
    y_enc = alt.Y('path:N', sort=alt.EncodingSortField(field='samples', order='descending'), title='Tree path')

    color_enc = alt.Color('gene:N',
                          scale=alt.Scale(domain=genes[:20], scheme='category20'),
                          legend=alt.Legend(title='Feature', orient='left'))

    filtered = alt.Chart(df).transform_filter('datum.samples >= min_n')

    rects = filtered.mark_rect(stroke='white', strokeWidth=1).encode(
        x=x_enc, y=y_enc, color=color_enc, tooltip=tooltips
    ).properties(title=title, width=alt.Step(60), height=alt.Step(30))

    texts = filtered.mark_text(fontSize=9).encode(
        x=x_enc, y=y_enc,
        text=alt.Text('cell_text:N'),

        tooltip=tooltips
    )

    base = alt.layer(rects, texts)

    n_col = filtered.transform_aggregate(
        n='max(samples)', groupby=['path']
    ).mark_text(align='center', baseline='middle').encode(
        y=alt.Y('path:N', sort=alt.EncodingSortField(field='n', order='descending'), axis=None),
        text=alt.Text('n:Q', format='d')
    ).properties(title=f'n/{class_n}' if class_n is not None else 'n', width=35, height=alt.Step(30))

    if 'sample_names' in df.columns:
        path_df = df[['path', 'sample_names', 'samples']].drop_duplicates(subset=['path'])
        samples_col = alt.Chart(path_df).transform_filter(
            'datum.samples >= min_n'
        ).mark_text(align='left', baseline='middle', fontSize=10).encode(
            y=alt.Y('path:N', sort=alt.EncodingSortField(field='samples', order='descending'), axis=None),
            text=alt.Text('sample_names:N')
        ).properties(title='Samples', width=250, height=alt.Step(30))
        return alt.hconcat(base, n_col, samples_col)

    return alt.hconcat(base, n_col)


def visualize_matrices_2(root, threshold=-60, overlap_threshold=0.30, output_dir=".",
                              class1_label="responder", class0_label="non-responder",
                              class1_n=None, class0_n=None):
    os.makedirs(output_dir, exist_ok=True)

    nodes = flatten_tree(root)

    #all unique features in first appear
    genes = []
    for node in nodes:
        for g in node.get('feature_order', []):
            if g not in genes:
                genes.append(g)

    if SPLIT_20:
        # each group capped at 20 features
        feat_groups = _group_by_feat_limit(nodes)
    else:
        # one plot for all, coloring limited to first 20
        feat_groups = _group_by_overlap(nodes, threshold=0)

    c1_data = {}
    c0_data = {}
    samples = []

    for gname, items in feat_groups.items():
        df_c1 = build_depth_df(items, class_val=1)
        df_c0 = build_depth_df(items, class_val=0)
        if not df_c1.empty:
            samples.extend(df_c1['samples'].tolist())
        if not df_c0.empty:
            samples.extend(df_c0['samples'].tolist())


        group_featset = set()
        for path, cv, n, features, node in items:
            group_featset |= features

        g_genes = []
        for g in genes:
            if g in group_featset:
                g_genes.append(g)

        c1_data[gname] = (df_c1, g_genes)
        c0_data[gname] = (df_c0, g_genes)

    max_n = max(samples) if samples else 1
    n_param = alt.param(name='min_n', value=1,
                        bind=alt.binding_range(min=1, max=max_n, step=1, name='Min n: '))

    c1_charts = []
    for gn, (df, g_genes) in c1_data.items():
        c1_charts.append(make_depth_heatmap(df, g_genes, f"{gn} - Class 1 ({class1_label})", class_n=class1_n))
    
    c0_charts = []
    for gn, (df, g_genes) in c0_data.items():
        c0_charts.append(make_depth_heatmap(df, g_genes, f"{gn} - Class 0 ({class0_label})", class_n=class0_n))

    c1 = alt.hconcat(*c1_charts).resolve_scale(color='independent')
    c0 = alt.hconcat(*c0_charts).resolve_scale(color='independent')

    c1 = c1.add_params(n_param).configure_view(stroke=None)
    c0 = c0.add_params(n_param).configure_view(stroke=None)


    out_c1 = os.path.join(output_dir, "tree_matrices_depth_class1.html")
    out_c0 = os.path.join(output_dir, "tree_matrices_depth_class0.html")

    save_chart(c1, out_c1,
               f'Decision Tree Feature Selection Class 1 ({class1_label})',
               threshold, class1_label, class0_label)
    save_chart(c0, out_c0,
               f'Decision Tree Feature Selection Class 0 ({class0_label})',
               threshold, class1_label, class0_label)
    print(f"  {out_c1}")
    print(f"  {out_c0}")