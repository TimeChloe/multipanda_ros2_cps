#!/usr/bin/env python3
"""Analyze runtime-versus-prediction inertia diagnostics from one or more runs."""

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PREDICTION_LOG = "shield_prediction_trajectory.csv"


def as_bool(value: str) -> bool:
    try:
        return float(value) != 0.0
    except ValueError:
        return value.strip().lower() in {"true", "yes"}


def quantile(values: List[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def require_columns(
    fieldnames: Iterable[str], required: Iterable[str], path: Path
) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(required) - available)
    if missing:
        raise RuntimeError(
            f"{path} does not contain the inertia-comparison schema; "
            f"missing: {', '.join(missing)}. Rebuild and collect a new run."
        )


def summary(name: str, values: List[float]) -> None:
    print(
        f"{name}: median={quantile(values, 0.5):.9f}, "
        f"p99={quantile(values, 0.99):.9f}, "
        f"min={min(values):.9f}, max={max(values):.9f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the runtime robot inertia and prediction-provider inertia "
            "evaluated at exactly the same measured q/dq."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    required = {
        "wall_time_sec",
        "source",
        "monitor_input_sequence",
        "inertia_model_comparison_valid",
        "runtime_model_joint_kinetic_energy",
        "prediction_model_joint_kinetic_energy",
        "inertia_model_kinetic_energy_error",
        "inertia_model_difference_frobenius_norm",
        "inertia_model_difference_relative_frobenius_norm",
        "inertia_model_difference_max_abs",
        "inertia_model_difference_max_abs_row",
        "inertia_model_difference_max_abs_col",
        "inertia_model_energy_ratio_valid",
        "inertia_model_min_energy_ratio",
        "inertia_model_max_energy_ratio",
    }

    comparisons: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    invalid = 0
    for run_dir in args.run_dirs:
        path = run_dir / PREDICTION_LOG
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require_columns(reader.fieldnames, required, path)
            for row in reader:
                if not as_bool(row["inertia_model_comparison_valid"]):
                    invalid += 1
                    continue
                source = row["source"]
                sequence = row["monitor_input_sequence"]
                # A prediction CSV contains one row per horizon endpoint, so
                # keep one comparison per monitor input. Synchronous records
                # have no async input sequence and are keyed by wall time.
                sync_time = row["wall_time_sec"] if source == "sync" else ""
                comparisons[(str(run_dir), sequence, sync_time)] = row

    if not comparisons:
        raise RuntimeError("No valid inertia-model comparisons were found.")

    rows = list(comparisons.values())
    runtime_energy = [
        float(row["runtime_model_joint_kinetic_energy"]) for row in rows
    ]
    prediction_energy = [
        float(row["prediction_model_joint_kinetic_energy"]) for row in rows
    ]
    kinetic_error = [
        float(row["inertia_model_kinetic_energy_error"]) for row in rows
    ]
    frobenius = [
        float(row["inertia_model_difference_frobenius_norm"]) for row in rows
    ]
    relative_frobenius = [
        float(row["inertia_model_difference_relative_frobenius_norm"])
        for row in rows
    ]
    maximum_entry = [
        float(row["inertia_model_difference_max_abs"]) for row in rows
    ]
    ratio_rows = [
        row for row in rows if as_bool(row["inertia_model_energy_ratio_valid"])
    ]
    minimum_ratio = [
        float(row["inertia_model_min_energy_ratio"]) for row in ratio_rows
    ]
    maximum_ratio = [
        float(row["inertia_model_max_energy_ratio"]) for row in ratio_rows
    ]
    maximum_locations = Counter(
        (
            int(round(float(row["inertia_model_difference_max_abs_row"]))),
            int(round(float(row["inertia_model_difference_max_abs_col"]))),
        )
        for row in rows
    )

    print(f"unique_monitor_inputs: {len(rows)}")
    print(f"invalid_csv_rows: {invalid}")
    summary("runtime_model_kinetic_energy_joule", runtime_energy)
    summary("prediction_model_kinetic_energy_joule", prediction_energy)
    summary("runtime_minus_prediction_kinetic_energy_joule", kinetic_error)
    summary("inertia_difference_frobenius_norm", frobenius)
    summary("inertia_difference_relative_frobenius_norm", relative_frobenius)
    summary("inertia_difference_max_abs", maximum_entry)
    if ratio_rows:
        summary("all_direction_min_kinetic_energy_ratio", minimum_ratio)
        summary("all_direction_max_kinetic_energy_ratio", maximum_ratio)
    print(
        "most_common_max_difference_entries_zero_based:",
        maximum_locations.most_common(5),
    )
    print(
        "observed_positive_model_energy_error_joule:",
        f"{max(0.0, max(kinetic_error)):.9f}",
    )
    print(
        "Note: this isolates inertia-model mismatch at identical q/dq; it is "
        "not a complete rollout kinetic-energy error bound."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
