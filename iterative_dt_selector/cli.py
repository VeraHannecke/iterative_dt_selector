import argparse
import importlib.metadata

from .runner import run_iterative_feature_selection


def get_version():
    try:
        return importlib.metadata.version("iterative-dt-selector")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def build_parser():
    parser = argparse.ArgumentParser(
        prog="iterative_dt_selector",
        description="Iterative feature selection using decision trees on binary genomic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {get_version()}"
    )

    parser.add_argument(
        "file",
        help="Path of input data file"
    )

    parser.add_argument(
        "--classification-threshold", type=float, default=-60,
        help="Threshold for classification. Samples <= threshold -> class 1."
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Maximum depth of decision trees."
    )
    parser.add_argument(
        "--importance-threshold", type=float, default=0.21,
        help="Stop when best feature importance drops below this value. 0 = disabled."
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Maximum number of features to remove (iterations)."
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0,
        help="Min overlap for paths to share a group. Set to 0 to show all paths in one group."
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Location of output files, created if it doesnt exist."
    )
    parser.add_argument(
        "--log-file", default=None,
        metavar="FILENAME",
        help="Enable logging to a file inside --output-dir."
    )
    parser.add_argument(
        "--class1-label", default="responder",
        help="Label name for class 1 (regression <= threshold)."
    )
    parser.add_argument(
        "--class0-label", default="non-responder",
        help="Label name for class 0 (regression > threshold)."
    )

    advanced = parser.add_argument_group("advanced tree options")
    advanced.add_argument(
        "--min-samples-split", type=int, default=2,
        help="Minimum samples required to split a node."
    )
    advanced.add_argument(
        "--min-samples-leaf", type=int, default=2,
        help="Minimum samples required in each leaf node."
    )


    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # treat 0 as disabled for importance threshold
    imp_threshold = args.importance_threshold if args.importance_threshold > 0 else None

    run_iterative_feature_selection(
        file_path=args.file,
        threshold=args.classification_threshold,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        importance_threshold=imp_threshold,
        num_iterations=args.iterations,
        #
        overlap_threshold=args.overlap_threshold,
        output_dir=args.output_dir,
        log_file=args.log_file,
        class1_label=args.class1_label,
        class0_label=args.class0_label
    )


if __name__ == "__main__":
    main()
