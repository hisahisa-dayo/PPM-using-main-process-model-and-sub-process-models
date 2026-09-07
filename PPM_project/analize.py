"""Export exception rules and their process models for each dataset.

The filename ``analize.py`` follows the requested spelling.

Examples:
    python analize.py
    python analize.py --data-set Helpdesk.csv
    python analize.py --error-threshold 0.50 --min-support-ratio 0.005

Outputs:
    analyze/<dataset>/exception_rules.txt
    analyze/<dataset>/main_process_model.png
    analyze/<dataset>/sub_process_models/sub_process_model_XXX.png
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import pm4py


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pm4py_dfg import ProcessModelGenerator
from pm_evaluator import evaluate_pm_on_train
from dt_router import train_decision_tree_router


DEFAULT_DATASETS = ("Helpdesk.csv", "Hospital.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save exception rules, the main DFG, and one sub-DFG per rule. "
            "Models are learned from all cases in each dataset."
        )
    )
    parser.add_argument(
        "--data-set",
        action="append",
        dest="data_sets",
        help=(
            "Dataset filename in data/. May be specified multiple times. "
            "If omitted, Helpdesk and Hospital are analyzed."
        ),
    )
    parser.add_argument("--error-threshold", type=float, default=0.50)
    parser.add_argument("--min-support-ratio", type=float, default=0.005)
    parser.add_argument("--categorical-min-frequency", type=int, default=10)
    parser.add_argument(
        "--edge-min-ratio",
        type=float,
        default=0.05,
        help=(
            "Hide DFG edges whose relative frequency is at or below this "
            "value for their source activity (default: 0.05). This affects "
            "PNG visualization only."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "analyze"
    )
    args = parser.parse_args()
    if not 0.0 <= args.error_threshold <= 1.0:
        parser.error("--error-threshold must be between 0 and 1")
    if not 0.0 < args.min_support_ratio <= 1.0:
        parser.error("--min-support-ratio must be greater than 0 and at most 1")
    if args.categorical_min_frequency < 0:
        parser.error("--categorical-min-frequency must be at least 0")
    if not 0.0 <= args.edge_min_ratio < 1.0:
        parser.error("--edge-min-ratio must be at least 0 and less than 1")
    args.data_sets = tuple(args.data_sets or DEFAULT_DATASETS)
    args.output_dir = args.output_dir.resolve()
    return args


def load_clean_cases(data_path: Path) -> list[tuple[str, list[str]]]:
    """Load traces using the CSV row order, as main.py does.

    The Helpdesk time column contains string values such as ``50:17.0``.
    Sorting those values lexicographically can change the event sequence, so
    rows must remain in their original order. ``groupby(sort=False)`` keeps
    both the first-seen case order and the row order within each case.
    """
    frame = pd.read_csv(data_path)
    required = {"case", "event", "time"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {data_path.name}: {', '.join(missing)}"
        )
    frame["case"] = frame["case"].astype(str)
    cases = []
    for case_id, group in frame.groupby("case", sort=False):
        trace = [
            activity for activity in group["event"].tolist()
            if activity not in {"!", "_END_"}
        ]
        if len(trace) > 1:
            cases.append((case_id, trace))
    return cases


def activity_dictionary(cases: list[tuple[object, list[str]]]) -> dict[str, int]:
    activities = sorted({activity for _, trace in cases for activity in trace})
    return {activity: index for index, activity in enumerate(activities)}


def filter_dfg_for_visualization(dfg, starts, ends, min_ratio: float):
    """Remove low-frequency edges without changing prediction models.

    A DFG edge is measured against all outgoing transitions from its source
    activity. Start/end edges are measured against all start/end counts.
    Edges with ratio <= min_ratio are hidden.
    """
    outgoing_totals = {}
    for (source, _), count in dfg.items():
        outgoing_totals[source] = outgoing_totals.get(source, 0) + count

    filtered_dfg = {
        edge: count
        for edge, count in dfg.items()
        if (
            outgoing_totals.get(edge[0], 0) > 0
            and count / outgoing_totals[edge[0]] > min_ratio
        )
    }
    total_starts = sum(starts.values())
    filtered_starts = {
        activity: count
        for activity, count in starts.items()
        if total_starts > 0 and count / total_starts > min_ratio
    }
    total_ends = sum(ends.values())
    filtered_ends = {
        activity: count
        for activity, count in ends.items()
        if total_ends > 0 and count / total_ends > min_ratio
    }
    return filtered_dfg, filtered_starts, filtered_ends


def save_dfg(dfg, starts, ends, output_path: Path, min_ratio: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visible_dfg, visible_starts, visible_ends = filter_dfg_for_visualization(
        dfg, starts, ends, min_ratio
    )
    pm4py.save_vis_dfg(
        visible_dfg, visible_starts, visible_ends, str(output_path)
    )


def write_rules(
    path: Path,
    dataset: str,
    rules: list[dict],
    error_threshold: float,
    min_support_ratio: float,
    edge_min_ratio: float,
) -> None:
    lines = [
        f"Dataset: {dataset}",
        f"Error threshold (theta_err): {error_threshold:.2f}",
        f"Minimum support ratio: {min_support_ratio:.4f}",
        f"Visualization edge cutoff: <= {edge_min_ratio * 100:.2f}% hidden",
        f"Number of selected exception rules: {len(rules)}",
        "",
    ]
    if not rules:
        lines.append("No exception rule satisfied the specified conditions.")
    for rank, rule in enumerate(rules, 1):
        model_file = f"sub_process_models/sub_process_model_{rank:03d}.png"
        lines.extend([
            "=" * 78,
            f"Rule {rank}",
            f"Target activity: {rule['Activity']}",
            f"Error rate: {rule['Error_Rate']:.6f} "
            f"({rule['Error_Rate'] * 100:.2f}%)",
            f"Error events: {int(rule['Error_Count'])}",
            f"Total events in leaf: {int(rule['Total_Cases'])}",
            f"Number of unique cases: {len(rule['Case_IDs'])}",
            f"Decision-tree leaf ID: {rule['Leaf_ID']}",
            f"Sub-process model: {model_file}",
            "Condition:",
            f"IF {rule['Rule']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def analyze_dataset(dataset_file: str, args: argparse.Namespace) -> None:
    data_path = PROJECT_DIR / "data" / dataset_file
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {data_path}")

    dataset_name = data_path.stem
    dataset_dir = args.output_dir / dataset_name
    submodel_dir = dataset_dir / "sub_process_models"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    submodel_dir.mkdir(parents=True, exist_ok=True)
    # Avoid retaining obsolete sub-model images when rerunning with a
    # different threshold. Only files generated by this script are removed.
    for old_image in submodel_dir.glob("sub_process_model_*.png"):
        old_image.unlink()

    print(f"\n=== Analyzing {dataset_name} ===")
    cases = load_clean_cases(data_path)
    if not cases:
        raise ValueError(f"No usable cases were found in {data_path}")

    act_to_id = activity_dictionary(cases)
    traces = [trace for _, trace in cases]
    generator = ProcessModelGenerator(traces, act_to_id)
    matrix, matrix_act_to_id, dfg, starts, ends, second_order = (
        generator.calculate_transition_matrix()
    )
    save_dfg(
        dfg,
        starts,
        ends,
        dataset_dir / "main_process_model.png",
        args.edge_min_ratio,
    )

    evaluation = evaluate_pm_on_train(
        matrix, matrix_act_to_id, cases, second_order
    )
    with contextlib.redirect_stdout(io.StringIO()):
        _, rules, _ = train_decision_tree_router(
            df_eval=evaluation,
            log_csv_path=str(data_path),
            min_support_ratio=args.min_support_ratio,
            total_train_cases=len(cases),
            error_threshold=args.error_threshold,
            csv_sep=",",
            verbose=False,
            min_category_frequency=args.categorical_min_frequency,
        )

    for rank, rule in enumerate(rules, 1):
        case_ids = {str(case_id) for case_id in rule["Case_IDs"]}
        sub_traces = [
            trace for case_id, trace in cases if str(case_id) in case_ids
        ]
        if not sub_traces:
            raise ValueError(
                f"Rule {rank} in {dataset_name} has no matching traces."
            )
        sub_generator = ProcessModelGenerator(sub_traces, act_to_id)
        _, _, sub_dfg, sub_starts, sub_ends, _ = (
            sub_generator.calculate_transition_matrix()
        )
        save_dfg(
            sub_dfg,
            sub_starts,
            sub_ends,
            submodel_dir / f"sub_process_model_{rank:03d}.png",
            args.edge_min_ratio,
        )

    write_rules(
        dataset_dir / "exception_rules.txt",
        dataset_name,
        rules,
        args.error_threshold,
        args.min_support_ratio,
        args.edge_min_ratio,
    )
    print(f"Cases: {len(cases)}")
    print(f"Selected rules / sub-models: {len(rules)}")
    print(f"Saved to: {dataset_dir}")


def main() -> None:
    args = parse_args()
    failures = []
    for dataset_file in args.data_sets:
        try:
            analyze_dataset(dataset_file, args)
        except Exception as exc:
            failures.append((dataset_file, exc))
            print(f"ERROR: {dataset_file}: {exc}", file=sys.stderr)
    if failures:
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        raise RuntimeError(f"Analysis failed for {len(failures)} dataset(s): {details}")


if __name__ == "__main__":
    main()
