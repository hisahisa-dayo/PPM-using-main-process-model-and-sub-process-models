"""Plot proposed_attr_transition accuracy while varying one threshold.

Examples:
    python plot.py --vary error --fixed 0.75
    python plot.py --vary max --fixed 0.50
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
except ImportError as exc:
    raise ImportError(
        "plot.py requires matplotlib. Install it with "
        "'python -m pip install matplotlib'."
    ) from exc

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "figures"
THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))
X_TICKS = tuple(round(i / 10, 1) for i in range(11))
DATASETS = (
    ("BPIC2012", "BPIC12"),
    ("BPIC2017", "BPIC17"),
    ("PrepaidTravelCost", "BPIC20PC"),
    ("RequestForPayment", "BPIC20RP"),
    ("Helpdesk", "Helpdesk"),
    ("Hospital", "Hospital"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy while varying theta_err or theta_max."
    )
    parser.add_argument(
        "--vary",
        choices=("error", "max", "error_threshold", "pm_confidence_threshold"),
        default="error",
        help="Threshold varied from 0.1 to 0.9 (default: error).",
    )
    parser.add_argument(
        "--fixed", type=float, default=None,
        help=(
            "Value of the other threshold. Defaults to 0.75 for --vary error "
            "and 0.50 for --vary max."
        ),
    )
    args = parser.parse_args()
    args.vary = {
        "error_threshold": "error",
        "pm_confidence_threshold": "max",
    }.get(args.vary, args.vary)
    if args.fixed is None:
        args.fixed = 0.75 if args.vary == "error" else 0.50
    if not 0.0 <= args.fixed <= 1.0:
        parser.error("--fixed must be between 0.0 and 1.0")
    return args


def read_accuracy(path: Path) -> float:
    """Return mean_total improved_acc on a 0--1 scale."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "improved_acc" not in rows[0]:
        raise ValueError(f"improved_acc was not found in {path}")
    row = next((r for r in rows if r.get("fold") == "mean_total"), None)
    if row is None or not row.get("improved_acc"):
        raise ValueError(f"mean_total improved_acc was not found in {path}")
    return float(row["improved_acc"]) / 100.0


def result_path(
    dataset: str, vary: str, threshold: float, fixed: float
) -> Path:
    varying_value = f"{threshold:.2f}"
    fixed_value = f"{fixed:.2f}"
    if vary == "error":
        error_value, max_value = varying_value, fixed_value
    else:
        error_value, max_value = fixed_value, varying_value
    return (
        RESULTS_DIR / dataset /
        f"summary_{dataset}_proposed_attr_{error_value}_{max_value}.csv"
    )


def load_values(vary: str, fixed: float) -> dict[str, list[float]]:
    values = {}
    for dataset, _ in DATASETS:
        accuracies = []
        for threshold in THRESHOLDS:
            path = result_path(dataset, vary, threshold, fixed)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required result was not found: {path}\n"
                    "Run main_loop.py for this threshold setting first."
                )
            accuracies.append(read_accuracy(path))
        values[dataset] = accuracies
    return values


def axis_limits(values: list[float]) -> tuple[float, float]:
    step = 0.05
    low = math.floor((min(values) - 0.004) / step) * step
    high = math.ceil((max(values) + 0.004) / step) * step
    if high - low < 0.10:
        middle = (min(values) + max(values)) / 2
        low = math.floor((middle - 0.05) / step) * step
        high = low + 0.10
    return max(0.0, low), min(1.0, high)


def create_figure(
    values: dict[str, list[float]], vary: str, fixed: float
) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 17,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 7.5), squeeze=False)
    for axis, (dataset, display_name) in zip(axes.flat, DATASETS):
        accuracies = values[dataset]
        axis.plot(
            THRESHOLDS, accuracies, color="#1f77b4", marker="o",
            markersize=6.0, linewidth=2.2,
        )
        axis.set_title(display_name, pad=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(X_TICKS)
        axis.set_xticklabels(["0"] + [f"{x:.1f}" for x in X_TICKS[1:]])
        axis.set_ylim(*axis_limits(accuracies))
        axis.yaxis.set_major_locator(MultipleLocator(0.05))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(direction="in", length=4, width=0.9)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    if vary == "error":
        x_label = r"Error threshold $\theta_{\mathrm{err}}$"
        stem = f"accuracy_theta_err_fixed_theta_max_{fixed:.2f}"
    else:
        x_label = r"Maximum-probability threshold $\theta_{\mathrm{max}}$"
        stem = f"accuracy_theta_max_fixed_theta_err_{fixed:.2f}"
    figure.supylabel("Accuracy", fontsize=18, x=0.018)
    figure.supxlabel(x_label, fontsize=18, y=0.018)
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.105, top=0.94,
        wspace=0.20, hspace=0.30,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    values = load_values(args.vary, args.fixed)
    png_path, pdf_path = create_figure(values, args.vary, args.fixed)
    for dataset, display_name in DATASETS:
        print(display_name + ": " + ", ".join(
            f"{threshold:.1f}={accuracy:.4f}"
            for threshold, accuracy in zip(THRESHOLDS, values[dataset])
        ))
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
