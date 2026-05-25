iterative-dt-selector

Iterative feature selection using decision trees on binary or continuous data. Builds a branching tree of decision trees, starting from the full feature set, it identifies important split features and recursively spawns a separate branch for each by removing that feature and rebuilding. Produces interactive HTML heatmap visualizations of tree paths per class.

Install with `pip install iterative-dt-selector` (not uploaded yet)

Dependencies: pandas>=2.0.0, numpy>=1.26.4, scikit-learn>=1.3.0, altair>=6.0.0, openpyxl>=3.1.5


Usage

```
iterative_dt_selector data.xlsx

iterative_dt_selector data.xlsx --classification-threshold -60 --importance-threshold 0.21 --iterations 5 --output-dir ./results --log-file run.log
```

If the package is not installed, run it directly from the source directory:

```
python -m iterative_dt_selector.cli data.xlsx
```


Data format

Accepts .xlsx, .xls, or .csv. The file must follow this layout:

- Row 0: sample IDs (first cell ignored)
- Row 1: continuous regression/response values used for classification
- Rows 2+: one feature per row, values binary (0/1) or continuous


symbol      sample_1   sample_2   ...
regression  -45.2      -78.1      ...
GENE_A      0          1          ...
GENE_B      1          0          ...



CLI flags

--classification-threshold (default -60)
Threshold for classification. Samples <= threshold are assigned to class 1.

--max-depth (default 6)
Maximum depth of decision trees.

--importance-threshold (default 0.21)
Stop a branch when the best feature importance drops below this value. Set to 0 to disable.

--iterations (default 10)
Maximum number of features to remove (branch recursion depth).

--overlap-threshold (default 0)
Min Jaccard overlap for paths to share a group in the visualization. Set to 0 to show all paths in one group.

--output-dir (default .)
Location of output files, created if it does not exist.

--log-file (default off)
Enable logging to a file inside --output-dir. Pass a filename.

--class1-label (default responder)
Display label for class 1 (regression <= threshold).

--class0-label (default non-responder)
Display label for class 0 (regression > threshold).

--min-samples-split (default 2)
Minimum samples required to split a node.

--min-samples-leaf (default 2)
Minimum samples required in each leaf node.
