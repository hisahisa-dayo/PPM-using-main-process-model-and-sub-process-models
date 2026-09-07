"""Sweep one threshold for proposed_attr_transition across paper datasets.

Edit the configuration block below, then run ``python main_loop.py``.
``SWEEP_PARAMETER`` selects the threshold to vary. The other stays fixed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Configuration -------------------------------------------------------------
DATASETS = (
    "Helpdesk.csv",
    "BPIC2012.csv",
    "BPIC2017.csv",
    "Hospital.csv",
    "RequestForPayment.csv",
    "PrepaidTravelCost.csv",
)

# Choose: "error_threshold" or "pm_confidence_threshold"
#SWEEP_PARAMETER = "error_threshold"
SWEEP_PARAMETER = "pm_confidence_threshold"

SWEEP_VALUES = tuple(value / 10 for value in range(1, 10))

FIXED_ERROR_THRESHOLD = 0.50
FIXED_PM_CONFIDENCE_THRESHOLD = 0.75
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"
DATA_DIR = PROJECT_DIR / "data"
VALID_SWEEP_PARAMETERS = {"error_threshold", "pm_confidence_threshold"}


def validate_inputs() -> None:
    if SWEEP_PARAMETER not in VALID_SWEEP_PARAMETERS:
        raise ValueError(
            f"Unknown SWEEP_PARAMETER: {SWEEP_PARAMETER!r}. "
            f"Choose one of: {sorted(VALID_SWEEP_PARAMETERS)}"
        )
    if not MAIN_PATH.is_file():
        raise FileNotFoundError(f"main.py was not found: {MAIN_PATH}")

    missing = [name for name in DATASETS if not (DATA_DIR / name).is_file()]
    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "The following dataset files were not found:\n" + missing_text
        )

    values = (
        *SWEEP_VALUES,
        FIXED_ERROR_THRESHOLD,
        FIXED_PM_CONFIDENCE_THRESHOLD,
    )
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("All threshold values must be between 0.0 and 1.0.")


def thresholds_for(sweep_value: float) -> tuple[float, float]:
    if SWEEP_PARAMETER == "error_threshold":
        return sweep_value, FIXED_PM_CONFIDENCE_THRESHOLD
    return FIXED_ERROR_THRESHOLD, sweep_value


def run_experiment(
    dataset: str, error_threshold: float, confidence_threshold: float
) -> None:
    command = [
        sys.executable,
        str(MAIN_PATH),
        "--data-set", dataset,
        "--experiment-method", "proposed_attr_transition",
        "--error-threshold", f"{error_threshold:.2f}",
        "--pm-confidence-threshold", f"{confidence_threshold:.2f}",
    ]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    validate_inputs()
    total = len(DATASETS) * len(SWEEP_VALUES)
    completed = 0

    fixed_name = (
        "pm_confidence_threshold"
        if SWEEP_PARAMETER == "error_threshold"
        else "error_threshold"
    )
    fixed_value = (
        FIXED_PM_CONFIDENCE_THRESHOLD
        if SWEEP_PARAMETER == "error_threshold"
        else FIXED_ERROR_THRESHOLD
    )

    print(f"Starting {total} proposed_attr_transition experiments.")
    print(f"Sweeping: {SWEEP_PARAMETER} = {SWEEP_VALUES}")
    print(f"Fixed: {fixed_name} = {fixed_value:.2f}")
    print(f"Python: {sys.executable}")

    for dataset in DATASETS:
        for sweep_value in SWEEP_VALUES:
            error_threshold, confidence_threshold = thresholds_for(sweep_value)
            completed += 1
            print("\n" + "=" * 80, flush=True)
            print(
                f"Experiment {completed}/{total}: dataset={dataset}, "
                f"error_threshold={error_threshold:.2f}, "
                f"pm_confidence_threshold={confidence_threshold:.2f}",
                flush=True,
            )
            print("=" * 80, flush=True)
            run_experiment(dataset, error_threshold, confidence_threshold)

    print(f"\nAll {total} experiments completed successfully.")
    print(f"Results directory: {PROJECT_DIR / 'results'}")


if __name__ == "__main__":
    main()
