#!/usr/bin/env python3
"""Estimate one-sided prediction energy-error bounds from matched CSV logs."""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List


CONTROL_LOG = "reachable_cartesian_impedance_validation.csv"
PREDICTION_LOG = "shield_prediction_trajectory.csv"


def as_bool(value: str) -> bool:
    return float(value) > 0.5


def quantile(values: List[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def require_columns(fieldnames: Iterable[str], required: Iterable[str], path: Path) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(required) - available)
    if missing:
        raise RuntimeError(
            f"{path} does not use the calibration log schema; missing: "
            + ", ".join(missing)
        )


def load_controls(path: Path) -> Dict[int, dict]:
    required = {
        "control_loop_sequence",
        "runtime_energy_scaling_enabled",
        "previous_applied_energy_valid",
        "previous_applied_joint_kinetic_energy",
        "previous_applied_cartesian_potential_energy",
    }
    controls: Dict[int, dict] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, required, path)
        for row in reader:
            controls[int(round(float(row["control_loop_sequence"])))] = row
    return controls


def state_error(prediction: dict, actual: dict, prefix: str) -> float:
    squared = 0.0
    for joint in range(1, 8):
        predicted = float(prediction[f"pred_{prefix}{joint}"])
        measured = float(actual[f"measured_{prefix}{joint}"])
        squared += (measured - predicted) ** 2
    return math.sqrt(squared)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Match guaranteed committed prediction samples to measured control "
            "cycles and estimate direct one-sided beta_K/beta_V bounds."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--guard-factor",
        type=float,
        default=1.2,
        help="multiplicative guard applied to the observed positive maximum",
    )
    args = parser.parse_args()
    if args.guard_factor < 1.0:
        parser.error("--guard-factor must be at least 1.0")

    kinetic_residuals: List[float] = []
    potential_residuals: List[float] = []
    q_errors: List[float] = []
    dq_errors: List[float] = []
    unmatched = 0
    rejected = 0

    prediction_required = {
        "async_timing_valid",
        "source_plan_matches_at_handoff",
        "guaranteed_committed_sample",
        "pred_energy_valid",
        "expected_control_sequence_valid",
        "expected_control_loop_sequence",
        "pred_joint_kinetic_energy",
        "pred_cartesian_potential_energy",
    }
    prediction_required.update(f"pred_q{joint}" for joint in range(1, 8))
    prediction_required.update(f"pred_dq{joint}" for joint in range(1, 8))

    for run_dir in args.run_dirs:
        control_path = run_dir / CONTROL_LOG
        prediction_path = run_dir / PREDICTION_LOG
        controls = load_controls(control_path)
        with prediction_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require_columns(reader.fieldnames, prediction_required, prediction_path)
            for prediction in reader:
                trusted = (
                    as_bool(prediction["async_timing_valid"])
                    and as_bool(prediction["source_plan_matches_at_handoff"])
                    and as_bool(prediction["guaranteed_committed_sample"])
                    and as_bool(prediction["pred_energy_valid"])
                    and as_bool(prediction["expected_control_sequence_valid"])
                )
                if not trusted:
                    rejected += 1
                    continue
                sequence = int(round(float(
                    prediction["expected_control_loop_sequence"]
                )))
                actual = controls.get(sequence)
                if actual is None:
                    unmatched += 1
                    continue
                if (
                    as_bool(actual["runtime_energy_scaling_enabled"])
                    or not as_bool(actual["previous_applied_energy_valid"])
                ):
                    rejected += 1
                    continue

                kinetic_residuals.append(
                    float(actual["previous_applied_joint_kinetic_energy"])
                    - float(prediction["pred_joint_kinetic_energy"])
                )
                potential_residuals.append(
                    float(actual[
                        "previous_applied_cartesian_potential_energy"
                    ])
                    - float(prediction["pred_cartesian_potential_energy"])
                )
                q_errors.append(state_error(prediction, actual, "q"))
                dq_errors.append(state_error(prediction, actual, "dq"))

    if not kinetic_residuals:
        raise RuntimeError(
            "No trusted matched samples were found. Run with "
            "enable_runtime_energy_scaling=false and the new log schema."
        )

    def print_summary(name: str, values: List[float]) -> None:
        positive_max = max(0.0, max(values))
        print(
            f"{name}: median={quantile(values, 0.5):.9f}, "
            f"p99={quantile(values, 0.99):.9f}, max={max(values):.9f}, "
            f"one_sided_max={positive_max:.9f}"
        )

    beta_k = args.guard_factor * max(0.0, max(kinetic_residuals))
    beta_v = args.guard_factor * max(0.0, max(potential_residuals))
    print(f"matched_samples: {len(kinetic_residuals)}")
    print(f"unmatched_samples: {unmatched}")
    print(f"filtered_samples: {rejected}")
    print_summary("kinetic_residual_joule", kinetic_residuals)
    print_summary("potential_residual_joule", potential_residuals)
    print_summary("joint_position_error_norm_rad", q_errors)
    print_summary("joint_velocity_error_norm_rad_s", dq_errors)
    print("\nPreliminary calibration values (validate on independent runs):")
    print(f"kinetic_energy_error_bound_joule: {beta_k:.9f}")
    print(f"potential_energy_error_bound_joule: {beta_v:.9f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
