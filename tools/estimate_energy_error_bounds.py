#!/usr/bin/env python3
"""Estimate energy errors from dedicated monitored-trajectory experiments."""

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CONTROL_LOG = "reachable_cartesian_impedance_validation.csv"
PREDICTION_LOG = "shield_prediction_trajectory.csv"
STAGE_CODE = {"intended": 1, "failsafe": 2}


def as_bool(value: str) -> bool:
    return float(value) > 0.5


def as_int(row: dict, name: str) -> int:
    return int(round(float(row[name])))


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


def require_columns(
    fieldnames: Iterable[str], required: Iterable[str], path: Path
) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(required) - available)
    if missing:
        raise RuntimeError(
            f"{path} does not use the passive calibration log schema; missing: "
            + ", ".join(missing)
        )


def load_controls(path: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    required = {
        "control_loop_sequence",
        "runtime_energy_scaling_enabled",
        "previous_applied_energy_valid",
        "previous_applied_joint_kinetic_energy",
        "previous_applied_cartesian_potential_energy",
        "previous_applied_verified_plan_valid",
        "previous_applied_verified_plan_generation",
        "previous_applied_verified_command_stage",
        "previous_applied_verified_command_index",
        "calibration_plan_latched",
        "calibration_plan_complete",
        "calibration_plan_generation",
        "calibration_activation_control_sequence",
        "calibration_activation_intended_index",
        "calibration_activation_failsafe_index",
    }
    required.update(f"measured_q{joint}" for joint in range(1, 8))
    required.update(f"measured_dq{joint}" for joint in range(1, 8))
    controls: Dict[int, dict] = {}
    calibrations: Dict[int, dict] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, required, path)
        for row in reader:
            controls[as_int(row, "control_loop_sequence")] = row
            if not as_bool(row["calibration_plan_latched"]):
                continue
            generation = as_int(row, "calibration_plan_generation")
            if generation <= 0:
                continue
            previous = calibrations.get(generation, {})
            calibrations[generation] = {
                "activation_intended_index": as_int(
                    row, "calibration_activation_intended_index"
                ),
                "activation_control_sequence": as_int(
                    row, "calibration_activation_control_sequence"
                ),
                "activation_failsafe_index": as_int(
                    row, "calibration_activation_failsafe_index"
                ),
                "complete": previous.get("complete", False)
                or as_bool(row["calibration_plan_complete"]),
            }
    return controls, calibrations


def state_error(prediction: dict, actual: dict, prefix: str) -> float:
    squared = 0.0
    for joint in range(1, 8):
        predicted = float(prediction[f"pred_{prefix}{joint}"])
        measured = float(actual[f"measured_{prefix}{joint}"])
        squared += (measured - predicted) ** 2
    return math.sqrt(squared)


def prediction_key(prediction: dict) -> Tuple[str, int]:
    return prediction["stage"], as_int(prediction, "index")


def expected_plan_keys(prediction: dict) -> set:
    intended_count = as_int(prediction, "plan_intended_steps")
    failsafe_count = as_int(prediction, "plan_failsafe_steps")
    keys = {
        ("intended", index) for index in range(intended_count)
    }
    keys.update(("failsafe", index) for index in range(failsafe_count))
    return keys


def ordered_plan_keys(
    prediction: dict, failsafe_start: int = 0
) -> List[Tuple[str, int]]:
    """Return monitored-plan keys in their runtime execution order."""
    intended_count = as_int(prediction, "plan_intended_steps")
    failsafe_count = as_int(prediction, "plan_failsafe_steps")
    keys = [("intended", index) for index in range(intended_count)]
    keys.extend(
        ("failsafe", index)
        for index in range(failsafe_start, failsafe_count)
    )
    return keys


def energy_residual(entry: dict, kind: str) -> float:
    actual = entry["actual"]
    prediction = entry["prediction"]
    return (
        float(actual[f"previous_applied_{kind}"])
        - float(prediction[f"pred_{kind}"])
    )


def add_stage_context(axis, entries: List[dict]) -> None:
    intended_count = sum(
        entry["prediction"]["stage"] == "intended" for entry in entries
    )
    if intended_count == 0 or intended_count == len(entries):
        return
    boundary = intended_count - 0.5
    axis.axvspan(
        boundary,
        len(entries) - 0.5,
        color="#7f7f7f",
        alpha=0.08,
        zorder=0,
    )
    axis.axvline(
        boundary,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="intended -> failsafe",
    )


def write_matched_samples(path: Path, entries: List[dict]) -> None:
    fields = [
        "trajectory_index",
        "stage",
        "stage_index",
        "expected_control_loop_sequence",
        "prediction_horizon_steps",
        "measured_kinetic_energy_joule",
        "predicted_kinetic_energy_joule",
        "kinetic_residual_joule",
        "measured_potential_energy_joule",
        "predicted_potential_energy_joule",
        "potential_residual_joule",
        "joint_position_error_norm_rad",
        "joint_velocity_error_norm_rad_s",
    ]
    for prefix in ("q", "dq"):
        for joint in range(1, 8):
            fields.extend(
                (
                    f"measured_{prefix}{joint}",
                    f"predicted_{prefix}{joint}",
                    f"{prefix}{joint}_error_measured_minus_predicted",
                )
            )

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trajectory_index, entry in enumerate(entries):
            prediction = entry["prediction"]
            actual = entry["actual"]
            row = {
                "trajectory_index": trajectory_index,
                "stage": prediction["stage"],
                "stage_index": as_int(prediction, "index"),
                "expected_control_loop_sequence": as_int(
                    prediction, "expected_control_loop_sequence"
                ),
                "prediction_horizon_steps": as_int(
                    prediction, "prediction_horizon_steps"
                ),
                "measured_kinetic_energy_joule": float(
                    actual["previous_applied_joint_kinetic_energy"]
                ),
                "predicted_kinetic_energy_joule": float(
                    prediction["pred_joint_kinetic_energy"]
                ),
                "kinetic_residual_joule": energy_residual(
                    entry, "joint_kinetic_energy"
                ),
                "measured_potential_energy_joule": float(
                    actual["previous_applied_cartesian_potential_energy"]
                ),
                "predicted_potential_energy_joule": float(
                    prediction["pred_cartesian_potential_energy"]
                ),
                "potential_residual_joule": energy_residual(
                    entry, "cartesian_potential_energy"
                ),
                "joint_position_error_norm_rad": state_error(
                    prediction, actual, "q"
                ),
                "joint_velocity_error_norm_rad_s": state_error(
                    prediction, actual, "dq"
                ),
            }
            for prefix in ("q", "dq"):
                for joint in range(1, 8):
                    measured = float(actual[f"measured_{prefix}{joint}"])
                    predicted = float(prediction[f"pred_{prefix}{joint}"])
                    row[f"measured_{prefix}{joint}"] = measured
                    row[f"predicted_{prefix}{joint}"] = predicted
                    row[
                        f"{prefix}{joint}_error_measured_minus_predicted"
                    ] = measured - predicted
            writer.writerow(row)


def generate_comparison_outputs(
    output_dir: Path,
    plan_label: str,
    entries: List[dict],
    guard_factor: float,
    dpi: int,
) -> List[Path]:
    """Write exact-match CSV and comparison figures for one full plan."""
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = Path(tempfile.gettempdir()) / "cps_xdg_cache"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_cache = Path(tempfile.gettempdir()) / "cps_matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in plan_label
    )
    prefix = output_dir / safe_label
    x = list(range(len(entries)))
    title_suffix = (
        f"{plan_label} ({len(entries)} exactly matched control cycles)"
    )

    measured_k = [
        float(entry["actual"]["previous_applied_joint_kinetic_energy"])
        for entry in entries
    ]
    predicted_k = [
        float(entry["prediction"]["pred_joint_kinetic_energy"])
        for entry in entries
    ]
    residual_k = [
        energy_residual(entry, "joint_kinetic_energy") for entry in entries
    ]
    measured_v = [
        float(
            entry["actual"]["previous_applied_cartesian_potential_energy"]
        )
        for entry in entries
    ]
    predicted_v = [
        float(entry["prediction"]["pred_cartesian_potential_energy"])
        for entry in entries
    ]
    residual_v = [
        energy_residual(entry, "cartesian_potential_energy")
        for entry in entries
    ]
    q_norm = [
        state_error(entry["prediction"], entry["actual"], "q")
        for entry in entries
    ]
    dq_norm = [
        state_error(entry["prediction"], entry["actual"], "dq")
        for entry in entries
    ]
    q_abs = [
        [
            abs(
                float(entry["actual"][f"measured_q{joint}"])
                - float(entry["prediction"][f"pred_q{joint}"])
            )
            for entry in entries
        ]
        for joint in range(1, 8)
    ]
    dq_abs = [
        [
            abs(
                float(entry["actual"][f"measured_dq{joint}"])
                - float(entry["prediction"][f"pred_dq{joint}"])
            )
            for entry in entries
        ]
        for joint in range(1, 8)
    ]

    output_paths: List[Path] = []
    csv_path = Path(f"{prefix}_matched_samples.csv")
    write_matched_samples(csv_path, entries)
    output_paths.append(csv_path)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis in axes.flat:
        add_stage_context(axis, entries)
        axis.grid(True, alpha=0.3)
    axes[0, 0].plot(x, measured_k, label="measured", linewidth=1.8)
    axes[0, 0].plot(x, predicted_k, label="predicted", linewidth=1.5)
    axes[0, 0].set_title("Joint kinetic energy")
    axes[0, 0].set_ylabel("energy [J]")
    axes[0, 0].legend()
    axes[1, 0].plot(x, residual_k, color="#d62728", linewidth=1.5)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    beta_k = guard_factor * max(0.0, max(residual_k))
    if beta_k > 0.0:
        axes[1, 0].axhline(
            beta_k,
            color="#9467bd",
            linestyle=":",
            label=f"guarded beta_K={beta_k:.4g} J",
        )
        axes[1, 0].legend()
    else:
        axes[1, 0].text(
            0.02,
            0.06,
            "observed one-sided beta_K = 0 J",
            transform=axes[1, 0].transAxes,
        )
    axes[1, 0].set_title("K residual: measured - predicted")
    axes[1, 0].set_ylabel("residual [J]")
    axes[1, 0].set_xlabel("monitored trajectory index")

    axes[0, 1].plot(x, measured_v, label="measured", linewidth=1.8)
    axes[0, 1].plot(x, predicted_v, label="predicted", linewidth=1.5)
    axes[0, 1].set_title("Cartesian potential energy")
    axes[0, 1].set_ylabel("energy [J]")
    axes[0, 1].legend()
    axes[1, 1].plot(x, residual_v, color="#d62728", linewidth=1.5)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    beta_v = guard_factor * max(0.0, max(residual_v))
    if beta_v > 0.0:
        axes[1, 1].axhline(
            beta_v,
            color="#9467bd",
            linestyle=":",
            label=f"guarded beta_V={beta_v:.4g} J",
        )
        axes[1, 1].legend()
    axes[1, 1].set_title("V residual: measured - predicted")
    axes[1, 1].set_ylabel("residual [J]")
    axes[1, 1].set_xlabel("monitored trajectory index")
    figure.suptitle(f"Energy comparison: {title_suffix}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    energy_path = Path(f"{prefix}_energy_comparison.png")
    figure.savefig(energy_path, dpi=dpi)
    plt.close(figure)
    output_paths.append(energy_path)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis in axes.flat:
        add_stage_context(axis, entries)
    axes[0, 0].plot(x, q_norm, color="#2ca02c", linewidth=1.5)
    axes[0, 0].set_title("Joint-position error norm")
    axes[0, 0].set_ylabel("||q_meas - q_pred|| [rad]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(x, dq_norm, color="#d62728", linewidth=1.5)
    axes[0, 1].set_title("Joint-velocity error norm")
    axes[0, 1].set_ylabel("||dq_meas - dq_pred|| [rad/s]")
    axes[0, 1].grid(True, alpha=0.3)
    q_image = axes[1, 0].imshow(
        q_abs,
        aspect="auto",
        origin="lower",
        extent=(-0.5, len(entries) - 0.5, 0.5, 7.5),
        cmap="viridis",
    )
    axes[1, 0].set_title("Absolute position error per joint")
    axes[1, 0].set_ylabel("joint")
    axes[1, 0].set_yticks(range(1, 8))
    axes[1, 0].set_xlabel("monitored trajectory index")
    figure.colorbar(q_image, ax=axes[1, 0], label="absolute error [rad]")
    dq_image = axes[1, 1].imshow(
        dq_abs,
        aspect="auto",
        origin="lower",
        extent=(-0.5, len(entries) - 0.5, 0.5, 7.5),
        cmap="magma",
    )
    axes[1, 1].set_title("Absolute velocity error per joint")
    axes[1, 1].set_ylabel("joint")
    axes[1, 1].set_yticks(range(1, 8))
    axes[1, 1].set_xlabel("monitored trajectory index")
    figure.colorbar(
        dq_image, ax=axes[1, 1], label="absolute error [rad/s]"
    )
    figure.suptitle(f"State prediction error: {title_suffix}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    state_path = Path(f"{prefix}_state_error.png")
    figure.savefig(state_path, dpi=dpi)
    plt.close(figure)
    output_paths.append(state_path)

    for prefix_name, measured_prefix, predicted_prefix, unit, description in (
        ("joint_position", "measured_q", "pred_q", "rad", "Joint position"),
        (
            "joint_velocity",
            "measured_dq",
            "pred_dq",
            "rad/s",
            "Joint velocity",
        ),
    ):
        figure, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
        for joint, axis in enumerate(axes.flat[:7], start=1):
            measured = [
                float(entry["actual"][f"{measured_prefix}{joint}"])
                for entry in entries
            ]
            predicted = [
                float(entry["prediction"][f"{predicted_prefix}{joint}"])
                for entry in entries
            ]
            add_stage_context(axis, entries)
            axis.plot(x, measured, label="measured", linewidth=1.6)
            axis.plot(x, predicted, label="predicted", linewidth=1.3)
            axis.set_title(f"Joint {joint}")
            axis.set_ylabel(f"[{unit}]")
            axis.grid(True, alpha=0.3)
            if joint == 1:
                axis.legend()
        axes.flat[6].set_xlabel("monitored trajectory index")
        axes.flat[7].remove()
        figure.suptitle(f"{description} comparison: {title_suffix}")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure_path = Path(f"{prefix}_{prefix_name}_comparison.png")
        figure.savefig(figure_path, dpi=dpi)
        plt.close(figure)
        output_paths.append(figure_path)

    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Match prediction endpoints to the exact verified "
            "plan generation/stage/index applied by the runtime controller, "
            "then estimate direct one-sided beta_K/beta_V bounds."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--scope",
        choices=("auto", "committed", "full"),
        default="auto",
        help=(
            "full requires a completed calibration action (or a monitored "
            "plan fully executed by normal fallback); auto falls back to "
            "committed-prefix samples"
        ),
    )
    parser.add_argument(
        "--guard-factor",
        type=float,
        default=1.2,
        help="multiplicative guard applied to the observed positive maximum",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help=(
            "write exact-match CSV plus energy/state comparison PNGs for "
            "each complete monitored plan"
        ),
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=160,
        help="PNG resolution used with --plot-dir (default: 160)",
    )
    args = parser.parse_args()
    if args.guard_factor < 1.0:
        parser.error("--guard-factor must be at least 1.0")
    if args.plot_dpi <= 0:
        parser.error("--plot-dpi must be positive")

    prediction_required = {
        "async_timing_valid",
        "monitor_input_sequence",
        "monitor_input_control_loop_sequence",
        "source_plan_matches_at_handoff",
        "async_output_usable",
        "candidate_verified",
        "accepted_plan_generation",
        "plan_intended_steps",
        "plan_failsafe_steps",
        "stage",
        "index",
        "guaranteed_committed_sample",
        "pred_joint_state_valid",
        "pred_energy_valid",
        "expected_control_sequence_valid",
        "expected_control_loop_sequence",
        "prediction_horizon_steps",
        "pred_joint_kinetic_energy",
        "pred_cartesian_potential_energy",
    }
    prediction_required.update(f"pred_q{joint}" for joint in range(1, 8))
    prediction_required.update(f"pred_dq{joint}" for joint in range(1, 8))

    selected_entries: List[dict] = []
    total_rows = 0
    schema_filtered = 0
    unmatched = 0
    run_scopes: Counter[str] = Counter()
    complete_generations: List[str] = []
    selected_plans: List[Tuple[str, List[dict]]] = []

    for run_dir in args.run_dirs:
        controls, calibrations = load_controls(run_dir / CONTROL_LOG)
        trusted_entries: List[dict] = []
        accepted_plans: Dict[int, List[dict]] = defaultdict(list)

        prediction_path = run_dir / PREDICTION_LOG
        with prediction_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require_columns(reader.fieldnames, prediction_required, prediction_path)
            for prediction in reader:
                total_rows += 1
                base_trusted = (
                    as_bool(prediction["async_timing_valid"])
                    and as_bool(prediction["source_plan_matches_at_handoff"])
                    and as_bool(prediction["async_output_usable"])
                    and as_bool(prediction["candidate_verified"])
                    and as_bool(prediction["pred_joint_state_valid"])
                    and as_bool(prediction["pred_energy_valid"])
                    and as_bool(prediction["expected_control_sequence_valid"])
                    and prediction["stage"] in STAGE_CODE
                )
                if not base_trusted:
                    schema_filtered += 1
                    continue

                input_sequence = as_int(
                    prediction, "monitor_input_control_loop_sequence"
                )
                horizon = as_int(prediction, "prediction_horizon_steps")
                expected_sequence = as_int(
                    prediction, "expected_control_loop_sequence"
                )
                sequence_identity_valid = (
                    horizon > 0
                    and expected_sequence == input_sequence + horizon
                )
                actual = controls.get(expected_sequence)
                if actual is None:
                    unmatched += 1

                accepted_generation = as_int(
                    prediction, "accepted_plan_generation"
                )
                stage = prediction["stage"]
                index = as_int(prediction, "index")
                energy_valid = actual is not None and as_bool(
                    actual["previous_applied_energy_valid"]
                ) and not as_bool(actual["runtime_energy_scaling_enabled"])
                committed_match = (
                    sequence_identity_valid
                    and energy_valid
                    and as_bool(prediction["guaranteed_committed_sample"])
                )
                executed_plan_match = (
                    sequence_identity_valid
                    and energy_valid
                    and accepted_generation > 0
                    and as_bool(actual["previous_applied_verified_plan_valid"])
                    and as_int(
                        actual, "previous_applied_verified_plan_generation"
                    ) == accepted_generation
                    and as_int(
                        actual, "previous_applied_verified_command_stage"
                    ) == STAGE_CODE[stage]
                    and as_int(
                        actual, "previous_applied_verified_command_index"
                    ) == index
                )
                entry = {
                    "prediction": prediction,
                    "actual": actual,
                    "committed_match": committed_match,
                    "executed_plan_match": executed_plan_match,
                }
                trusted_entries.append(entry)
                if accepted_generation > 0:
                    accepted_plans[accepted_generation].append(entry)

        complete_normal_plans: Dict[int, List[dict]] = {}
        complete_calibration_plans: Dict[int, List[dict]] = {}
        for generation, entries in accepted_plans.items():
            rows_by_key = {
                prediction_key(entry["prediction"]): entry
                for entry in entries
            }
            required_keys = expected_plan_keys(entries[0]["prediction"])
            complete = (
                bool(required_keys)
                and set(rows_by_key) == required_keys
                and all(
                    entry["committed_match"]
                    or entry["executed_plan_match"]
                    for entry in rows_by_key.values()
                )
            )
            if complete:
                chronological_keys = ordered_plan_keys(
                    entries[0]["prediction"]
                )
                complete_normal_plans[generation] = [
                    rows_by_key[key] for key in chronological_keys
                ]

            calibration = calibrations.get(generation)
            if calibration is None or not calibration["complete"]:
                continue
            intended_count = as_int(
                entries[0]["prediction"], "plan_intended_steps"
            )
            failsafe_count = as_int(
                entries[0]["prediction"], "plan_failsafe_steps"
            )
            activation_intended = calibration["activation_intended_index"]
            activation_failsafe = calibration["activation_failsafe_index"]
            # Samples before activation were already sent while the worker was
            # evaluating this plan. They are still exact matches when marked
            # as part of the guaranteed committed prefix, so retain them in
            # the full monitored-trajectory calibration.
            calibration_keys = {
                ("intended", index) for index in range(intended_count)
            }
            calibration_keys.update(
                ("failsafe", index)
                for index in range(activation_failsafe, failsafe_count)
            )

            def calibration_entry_matches(key: Tuple[str, int]) -> bool:
                entry = rows_by_key[key]
                stage, index = key
                if stage == "intended" and index < activation_intended:
                    return entry["committed_match"]
                return entry["executed_plan_match"]

            calibration_complete = (
                activation_intended < intended_count
                and bool(calibration_keys)
                and calibration_keys.issubset(rows_by_key)
                and as_int(
                    rows_by_key[("intended", activation_intended)][
                        "prediction"
                    ],
                    "expected_control_loop_sequence",
                ) == calibration["activation_control_sequence"] + 1
                and all(
                    calibration_entry_matches(key) for key in calibration_keys
                )
            )
            if calibration_complete:
                chronological_keys = ordered_plan_keys(
                    entries[0]["prediction"], activation_failsafe
                )
                complete_calibration_plans[generation] = [
                    rows_by_key[key] for key in chronological_keys
                ]

        complete_plans = (
            complete_calibration_plans
            if complete_calibration_plans
            else complete_normal_plans
        )

        use_full = args.scope == "full" or (
            args.scope == "auto" and bool(complete_plans)
        )
        if use_full and not complete_plans:
            raise RuntimeError(
                f"{run_dir} contains no completed monitored-trajectory "
                "calibration. Send a goal to calibrate_monitored_trajectory "
                "and wait for the action result before stopping the run."
            )

        if use_full:
            scope_name = (
                "calibration_execution"
                if complete_calibration_plans
                else "full_normal_execution"
            )
            run_scopes[scope_name] += 1
            for generation, entries in sorted(complete_plans.items()):
                selected_entries.extend(entries)
                complete_generations.append(f"{run_dir.name}:{generation}")
                selected_plans.append(
                    (f"{run_dir.name}_generation_{generation}", entries)
                )
        else:
            run_scopes["committed_prefix"] += 1
            selected_entries.extend(
                entry for entry in trusted_entries
                if entry["committed_match"]
            )

    if not selected_entries:
        raise RuntimeError(
            "No exactly matched samples were found. Restart the controller "
            "with enable_calibration_logging=true and run the "
            "calibration action."
        )

    kinetic_residuals: List[float] = []
    potential_residuals: List[float] = []
    q_errors: List[float] = []
    dq_errors: List[float] = []
    matched_stages: Counter[str] = Counter()
    matched_horizons: List[int] = []
    for entry in selected_entries:
        prediction = entry["prediction"]
        actual = entry["actual"]
        kinetic_residuals.append(
            float(actual["previous_applied_joint_kinetic_energy"])
            - float(prediction["pred_joint_kinetic_energy"])
        )
        potential_residuals.append(
            float(actual["previous_applied_cartesian_potential_energy"])
            - float(prediction["pred_cartesian_potential_energy"])
        )
        q_errors.append(state_error(prediction, actual, "q"))
        dq_errors.append(state_error(prediction, actual, "dq"))
        matched_stages[prediction["stage"]] += 1
        matched_horizons.append(
            as_int(prediction, "prediction_horizon_steps")
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
    print(f"matched_samples: {len(selected_entries)}")
    print(f"unmatched_prediction_rows: {unmatched}")
    print(f"untrusted_prediction_rows: {schema_filtered}")
    print(f"unselected_prediction_rows: {total_rows - len(selected_entries)}")
    print(
        "run_scopes: "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(run_scopes.items())
        )
    )
    if complete_generations:
        print(
            "complete_matched_plan_generations: "
            + ", ".join(complete_generations)
        )
    print(
        "matched_stages: "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(matched_stages.items())
        )
    )
    print(f"maximum_matched_horizon_steps: {max(matched_horizons)}")
    print_summary("kinetic_residual_joule", kinetic_residuals)
    print_summary("potential_residual_joule", potential_residuals)
    print_summary("joint_position_error_norm_rad", q_errors)
    print_summary("joint_velocity_error_norm_rad_s", dq_errors)
    print("\nPreliminary calibration values (validate on independent runs):")
    print(f"kinetic_energy_error_bound_joule: {beta_k:.9f}")
    print(f"potential_energy_error_bound_joule: {beta_v:.9f}")
    if args.plot_dir is not None:
        if not selected_plans:
            raise RuntimeError(
                "--plot-dir requires a complete matched monitored plan. "
                "Use --scope full (or auto) after completing calibration."
            )
        output_paths: List[Path] = []
        for plan_label, entries in selected_plans:
            output_paths.extend(
                generate_comparison_outputs(
                    args.plot_dir,
                    plan_label,
                    entries,
                    args.guard_factor,
                    args.plot_dpi,
                )
            )
        print("\ncomparison_outputs:")
        for output_path in output_paths:
            print(f"  {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
