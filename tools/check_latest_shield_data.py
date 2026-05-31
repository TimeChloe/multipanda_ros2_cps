#!/usr/bin/env python3
"""Inspect the latest reachable-impedance run and plot prediction vs execution.

Default behavior:
  - find the newest data_log/<timestamp>/ containing both CSV files
  - compare measured collision-center pz with shield predicted pz
  - write plots and a short text summary into the same run directory
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


VALIDATION_CSV = "reachable_cartesian_impedance_validation.csv"
PREDICTION_CSV = "shield_prediction_trajectory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latest shield prediction trajectory against measured data."
    )
    parser.add_argument(
        "--data-root",
        default="data_log",
        help="Directory containing timestamped run folders.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific run directory. Defaults to newest complete run.",
    )
    parser.add_argument(
        "--stage",
        default="intended",
        choices=["intended", "failsafe"],
        help="Prediction stage to compare against measured trajectory.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Prediction sample index to compare. Use 0 for next-step prediction.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.25,
        help="Seconds before/after first failsafe time to emphasize in zoom plot.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.002,
        help="Max time difference for matching validation rows to prediction rows.",
    )
    parser.add_argument(
        "--mode1-only",
        action="store_true",
        help="Only generate the mode=1 nominal prediction vs measurement diagnostic.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print/write summary, do not create PNG plots.",
    )
    return parser.parse_args()


def newest_complete_run(data_root: Path) -> Path:
    candidates = []
    for path in data_root.iterdir():
        if not path.is_dir():
            continue
        if (path / VALIDATION_CSV).is_file() and (path / PREDICTION_CSV).is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No run directory under {data_root} contains both {VALIDATION_CSV} "
            f"and {PREDICTION_CSV}."
        )
    return sorted(candidates)[-1]


def to_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [
            row
            for row in csv.DictReader(f)
            if row and any(value not in (None, "") for value in row.values())
        ]


def first_time(rows: list[dict[str, str]], key: str, value: int = 1) -> float | None:
    for row in rows:
        if to_int(row, key) == value:
            return to_float(row, "wall_time_sec")
    return None


def nearest_validation_row(rows: list[dict[str, str]], t: float) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(to_float(row, "wall_time_sec") - t))


def prediction_absolute_time(row: dict[str, str]) -> float:
    return to_float(row, "wall_time_sec") + to_float(row, "sample_t", 0.0)


def find_nearest_prediction(
    rows: list[dict[str, str]],
    t: float,
    start_index: int,
) -> tuple[dict[str, str] | None, int]:
    if not rows:
        return None, start_index

    idx = min(max(start_index, 0), len(rows) - 1)
    while idx + 1 < len(rows) and prediction_absolute_time(rows[idx + 1]) <= t:
        idx += 1

    candidates = [rows[idx]]
    if idx + 1 < len(rows):
        candidates.append(rows[idx + 1])

    best = min(candidates, key=lambda row: abs(prediction_absolute_time(row) - t))
    return best, idx


def build_comparison(
    validation_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    stage: str,
    index: int,
) -> list[dict[str, float]]:
    selected_predictions = [
        row
        for row in prediction_rows
        if row.get("stage") == stage and to_int(row, "index", -1) == index
    ]

    comparison = []
    for pred in selected_predictions:
        t = to_float(pred, "wall_time_sec")
        measured = nearest_validation_row(validation_rows, t)
        if measured is None:
            continue

        measured_pz = to_float(measured, "collision_center_pz")
        pred_next_pz = to_float(pred, "pred_next_pz")
        target_pz = to_float(pred, "collision_target_pz")
        comparison.append(
            {
                "t": t,
                "measured_collision_pz": measured_pz,
                "pred_next_pz": pred_next_pz,
                "collision_target_pz": target_pz,
                "pred_error_pz": pred_next_pz - measured_pz,
                "target_error_pz": target_pz - measured_pz,
                "contact_possible": to_int(pred, "contact_possible"),
                "is_worst_T": to_int(pred, "is_worst_T"),
                "is_worst_V": to_int(pred, "is_worst_V"),
                "V_potential_ub": to_float(pred, "V_potential_ub"),
                "T_n_ub": to_float(pred, "T_n_ub"),
                "mode": to_int(measured, "mode"),
                "predicted_trigger": to_int(measured, "predicted_trigger"),
            }
        )
    return comparison


def build_executed_reference_comparison(
    validation_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    match_tolerance: float,
) -> list[dict[str, float | str]]:
    """Build one comparison row per control loop.

    mode 0:
      compare measured collision-center pz with the accepted intended prediction
      for stage=intended,index=0.

    mode 1:
      compare measured TCP pz with the actually executed failsafe command pz.
      The command pz is validation.csv des_pz, which is assigned from
      last_verified_plan_.failsafe[next_index] in failsafe mode.
    """

    accepted_intended = sorted(
        [
            row
            for row in prediction_rows
            if row.get("stage") == "intended"
            and to_int(row, "index", -1) == 0
            and to_int(row, "candidate_verified") == 1
        ],
        key=lambda row: to_float(row, "wall_time_sec"),
    )

    comparison = []
    pred_idx = 0
    for measured in validation_rows:
      t = to_float(measured, "wall_time_sec")
      mode = to_int(measured, "mode")

      row: dict[str, float | str] = {
          "t": t,
          "mode": mode,
          "measured_tcp_pz": to_float(measured, "cur_pz"),
          "measured_collision_center_pz": to_float(measured, "collision_center_pz"),
          "command_des_pz": to_float(measured, "des_pz"),
          "predicted_trigger": to_int(measured, "predicted_trigger"),
          "monitored_contact_possible": to_int(measured, "monitored_contact_possible"),
          "reference_pz": math.nan,
          "reference_kind": "",
          "reference_time_error_sec": math.nan,
          "error_mm": math.nan,
      }

      if mode == 0:
          while (
              pred_idx + 1 < len(accepted_intended)
              and to_float(accepted_intended[pred_idx + 1], "wall_time_sec") <= t
          ):
              pred_idx += 1

          candidates = []
          if pred_idx < len(accepted_intended):
              candidates.append(accepted_intended[pred_idx])
          if pred_idx + 1 < len(accepted_intended):
              candidates.append(accepted_intended[pred_idx + 1])

          if candidates:
              pred = min(
                  candidates,
                  key=lambda p: abs(to_float(p, "wall_time_sec") - t),
              )
              dt = to_float(pred, "wall_time_sec") - t
              if abs(dt) <= match_tolerance:
                  reference_pz = to_float(pred, "pred_next_pz")
                  measured_pz = row["measured_collision_center_pz"]
                  row["reference_pz"] = reference_pz
                  row["reference_kind"] = "accepted_intended_pred_next_collision_pz"
                  row["reference_time_error_sec"] = dt
                  row["error_mm"] = 1000.0 * (reference_pz - float(measured_pz))
      else:
          reference_pz = to_float(measured, "des_pz")
          measured_pz = to_float(measured, "cur_pz")
          row["reference_pz"] = reference_pz
          row["reference_kind"] = "executed_cached_failsafe_command_tcp_pz"
          row["reference_time_error_sec"] = 0.0
          row["error_mm"] = 1000.0 * (reference_pz - measured_pz)

      comparison.append(row)

    return comparison


def build_mode1_nominal_prediction_comparison(
    validation_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    match_tolerance: float,
) -> list[dict[str, float | str]]:
    """For mode=1 rows, compare monitored nominal prediction with measurement.

    This intentionally looks at the nominal/intended command still being
    monitored while the robot is already executing failsafe.
    """

    intended_index0 = sorted(
        [
            row
            for row in prediction_rows
            if row.get("stage") == "intended" and to_int(row, "index", -1) == 0
        ],
        key=lambda row: to_float(row, "wall_time_sec"),
    )

    comparison = []
    pred_idx = 0
    for measured in validation_rows:
        if to_int(measured, "mode") != 1:
            continue

        t = to_float(measured, "wall_time_sec")
        while (
            pred_idx + 1 < len(intended_index0)
            and to_float(intended_index0[pred_idx + 1], "wall_time_sec") <= t
        ):
            pred_idx += 1

        candidates = []
        if pred_idx < len(intended_index0):
            candidates.append(intended_index0[pred_idx])
        if pred_idx + 1 < len(intended_index0):
            candidates.append(intended_index0[pred_idx + 1])
        if not candidates:
            continue

        pred = min(candidates, key=lambda p: abs(to_float(p, "wall_time_sec") - t))
        dt = to_float(pred, "wall_time_sec") - t
        if abs(dt) > match_tolerance:
            continue

        measured_collision_pz = to_float(measured, "collision_center_pz")
        pred_nominal_pz = to_float(pred, "pred_next_pz")
        nominal_target_pz = to_float(pred, "collision_target_pz")

        comparison.append(
            {
                "t": t,
                "prediction_wall_time_sec": to_float(pred, "wall_time_sec"),
                "time_error_sec": dt,
                "measured_tcp_pz": to_float(measured, "cur_pz"),
                "measured_collision_center_pz": measured_collision_pz,
                "executed_failsafe_des_pz": to_float(measured, "des_pz"),
                "nominal_pred_next_pz": pred_nominal_pz,
                "nominal_collision_target_pz": nominal_target_pz,
                "nominal_pred_error_mm": 1000.0 * (pred_nominal_pz - measured_collision_pz),
                "nominal_target_error_mm": 1000.0 * (nominal_target_pz - measured_collision_pz),
                "candidate_verified": to_int(pred, "candidate_verified"),
                "predicted_trigger": to_int(pred, "predicted_trigger"),
                "monitored_contact_possible": to_int(pred, "monitored_contact_possible"),
                "contact_possible": to_int(pred, "contact_possible"),
                "is_worst_T": to_int(pred, "is_worst_T"),
                "is_worst_V": to_int(pred, "is_worst_V"),
                "T_n_ub": to_float(pred, "T_n_ub"),
                "V_potential_ub": to_float(pred, "V_potential_ub"),
                "distance_segment": to_float(pred, "distance_segment"),
            }
        )

    return comparison


def build_verification_aligned_comparison(
    validation_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    match_tolerance: float,
) -> list[dict[str, float | str]]:
    """Compare each measured loop with the verified prediction for that loop.

    mode 0:
      use the accepted intended prediction whose absolute predicted time
      wall_time + sample_t corresponds to the measured loop.

    mode 1:
      use the failsafe prediction from the most recent accepted verification
      whose absolute predicted time corresponds to the measured loop.
    """

    accepted_intended = sorted(
        [
            row
            for row in prediction_rows
            if row.get("stage") == "intended"
            and to_int(row, "candidate_verified") == 1
        ],
        key=prediction_absolute_time,
    )
    accepted_failsafe = sorted(
        [
            row
            for row in prediction_rows
            if row.get("stage") == "failsafe"
            and to_int(row, "candidate_verified") == 1
        ],
        key=prediction_absolute_time,
    )

    intended_idx = 0
    failsafe_idx = 0
    comparison = []

    for measured in validation_rows:
        t = to_float(measured, "wall_time_sec")
        mode = to_int(measured, "mode")
        predictions = accepted_intended if mode == 0 else accepted_failsafe
        start_idx = intended_idx if mode == 0 else failsafe_idx
        pred, new_idx = find_nearest_prediction(predictions, t, start_idx)
        if mode == 0:
            intended_idx = new_idx
        else:
            failsafe_idx = new_idx

        pred_time = prediction_absolute_time(pred) if pred is not None else math.nan
        time_error = pred_time - t if pred is not None else math.nan
        has_match = pred is not None and abs(time_error) <= match_tolerance

        measured_pz = to_float(measured, "collision_center_pz")
        pred_pz = to_float(pred, "pred_next_pz") if has_match else math.nan
        target_pz = to_float(pred, "collision_target_pz") if has_match else math.nan

        comparison.append(
            {
                "t": t,
                "mode": mode,
                "prediction_stage": pred.get("stage", "") if has_match else "",
                "prediction_index": to_int(pred, "index", -1) if has_match else -1,
                "verification_wall_time_sec": to_float(pred, "wall_time_sec") if has_match else math.nan,
                "prediction_abs_time_sec": pred_time,
                "time_error_sec": time_error,
                "measured_collision_center_pz": measured_pz,
                "measured_tcp_pz": to_float(measured, "cur_pz"),
                "pred_next_pz": pred_pz,
                "collision_target_pz": target_pz,
                "prediction_error_mm": 1000.0 * (pred_pz - measured_pz) if has_match else math.nan,
                "target_error_mm": 1000.0 * (target_pz - measured_pz) if has_match else math.nan,
                "measured_predicted_trigger": to_int(measured, "predicted_trigger"),
                "measured_monitored_contact_possible": to_int(measured, "monitored_contact_possible"),
                "candidate_verified": to_int(pred, "candidate_verified") if has_match else 0,
                "prediction_predicted_trigger": to_int(pred, "predicted_trigger") if has_match else 0,
                "contact_possible": to_int(pred, "contact_possible") if has_match else 0,
                "is_worst_T": to_int(pred, "is_worst_T") if has_match else 0,
                "is_worst_V": to_int(pred, "is_worst_V") if has_match else 0,
                "T_n_ub": to_float(pred, "T_n_ub") if has_match else math.nan,
                "V_potential_ub": to_float(pred, "V_potential_ub") if has_match else math.nan,
                "distance_segment": to_float(pred, "distance_segment") if has_match else math.nan,
            }
        )

    return comparison


def finite_values(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def summarize(
    run_dir: Path,
    validation_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    comparison: list[dict[str, float]],
    stage: str,
    index: int,
) -> str:
    first_contact = first_time(validation_rows, "monitored_contact_possible")
    first_trigger = first_time(validation_rows, "predicted_trigger")
    first_failsafe = first_time(validation_rows, "mode")
    first_mujoco_contact = first_time(validation_rows, "mujoco_contact_active")

    errors_mm = finite_values([1000.0 * row["pred_error_pz"] for row in comparison])
    abs_errors_mm = [abs(v) for v in errors_mm]

    worst_v_rows = [
        row for row in prediction_rows if to_int(row, "is_worst_V") == 1
    ]
    first_worst_v = (
        to_float(worst_v_rows[0], "wall_time_sec") if worst_v_rows else None
    )

    lines = [
        f"run_dir: {run_dir}",
        f"validation_rows: {len(validation_rows)}",
        f"prediction_rows: {len(prediction_rows)}",
        f"comparison_stage: {stage}",
        f"comparison_index: {index}",
        f"comparison_rows: {len(comparison)}",
        f"first_monitored_contact_possible_sec: {first_contact}",
        f"first_predicted_trigger_sec: {first_trigger}",
        f"first_mode_1_failsafe_sec: {first_failsafe}",
        f"first_mujoco_contact_active_sec: {first_mujoco_contact}",
        f"first_prediction_worst_V_sec: {first_worst_v}",
    ]

    if first_trigger is not None and first_failsafe is not None:
        lines.append(
            "trigger_to_failsafe_delay_ms: "
            f"{1000.0 * (first_failsafe - first_trigger):.3f}"
        )

    if errors_mm:
        lines.extend(
            [
                f"pred_next_pz_error_mean_mm: {mean(errors_mm):.3f}",
                f"pred_next_pz_abs_error_mean_mm: {mean(abs_errors_mm):.3f}",
                f"pred_next_pz_abs_error_max_mm: {max(abs_errors_mm):.3f}",
            ]
        )

    return "\n".join(lines) + "\n"


def write_comparison_csv(run_dir: Path, comparison: list[dict[str, float]]) -> Path:
    out_path = run_dir / "prediction_vs_measured_pz_index0.csv"
    if not comparison:
        out_path.write_text("")
        return out_path

    keys = list(comparison[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    return out_path


def write_executed_reference_csv(
    run_dir: Path,
    comparison: list[dict[str, float | str]],
) -> Path:
    out_path = run_dir / "executed_reference_vs_measured_pz.csv"
    if not comparison:
        out_path.write_text("")
        return out_path

    keys = list(comparison[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    return out_path


def write_mode1_nominal_csv(
    run_dir: Path,
    comparison: list[dict[str, float | str]],
) -> Path:
    out_path = run_dir / "mode1_nominal_prediction_vs_measured_pz.csv"
    if not comparison:
        out_path.write_text("")
        return out_path

    keys = list(comparison[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    return out_path


def write_verification_aligned_csv(
    run_dir: Path,
    comparison: list[dict[str, float | str]],
) -> Path:
    out_path = run_dir / "verification_aligned_prediction_vs_measured_pz.csv"
    if not comparison:
        out_path.write_text("")
        return out_path

    keys = list(comparison[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    return out_path


def make_plots(
    run_dir: Path,
    validation_rows: list[dict[str, str]],
    comparison: list[dict[str, float]],
    window: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_paths = []

    t_meas = [to_float(row, "wall_time_sec") for row in validation_rows]
    measured_pz = [to_float(row, "collision_center_pz") for row in validation_rows]
    mode = [to_int(row, "mode") for row in validation_rows]
    trigger = [to_int(row, "predicted_trigger") for row in validation_rows]

    t_pred = [row["t"] for row in comparison]
    pred_next_pz = [row["pred_next_pz"] for row in comparison]
    target_pz = [row["collision_target_pz"] for row in comparison]
    pred_error_mm = [1000.0 * row["pred_error_pz"] for row in comparison]

    first_failsafe = first_time(validation_rows, "mode")
    first_trigger = first_time(validation_rows, "predicted_trigger")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t_meas, measured_pz, label="measured collision_center_pz", lw=1.4)
    axes[0].plot(t_pred, pred_next_pz, label="predicted next pz", lw=1.1)
    axes[0].plot(t_pred, target_pz, label="prediction target pz", lw=1.0, alpha=0.8)
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t_pred, pred_error_mm, label="pred_next - measured [mm]", color="tab:red")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("error [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t_meas, mode, label="mode", drawstyle="steps-post")
    axes[2].plot(t_meas, trigger, label="predicted_trigger", drawstyle="steps-post")
    axes[2].set_ylabel("state")
    axes[2].set_xlabel("wall_time_sec")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    for ax in axes:
        if first_trigger is not None:
            ax.axvline(first_trigger, color="orange", linestyle="--", lw=1.0)
        if first_failsafe is not None:
            ax.axvline(first_failsafe, color="purple", linestyle="--", lw=1.0)

    fig.suptitle("Shield prediction vs measured collision-center pz")
    fig.tight_layout()
    out_path = run_dir / "prediction_vs_measured_pz.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)

    if first_failsafe is not None:
        xmin = first_failsafe - window
        xmax = first_failsafe + window
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t_meas, measured_pz, label="measured collision_center_pz", lw=1.4)
        ax.plot(t_pred, pred_next_pz, label="predicted next pz", lw=1.1)
        ax.plot(t_pred, target_pz, label="prediction target pz", lw=1.0, alpha=0.8)
        if first_trigger is not None:
            ax.axvline(first_trigger, color="orange", linestyle="--", label="first trigger")
        ax.axvline(first_failsafe, color="purple", linestyle="--", label="first mode=1")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("wall_time_sec")
        ax.set_ylabel("z [m]")
        ax.set_title("Zoom around first failsafe")
        ax.legend(loc="best")
        ax.grid(True)
        fig.tight_layout()
        out_path = run_dir / "prediction_vs_measured_pz_zoom.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def make_executed_reference_plot(
    run_dir: Path,
    executed_comparison: list[dict[str, float | str]],
    validation_rows: list[dict[str, str]],
    window: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in executed_comparison
        if isinstance(row["reference_pz"], float)
        and math.isfinite(row["reference_pz"])
    ]
    if not rows:
        return []

    t = [float(row["t"]) for row in rows]
    reference_pz = [float(row["reference_pz"]) for row in rows]
    measured_pz = [
        float(row["measured_collision_center_pz"])
        if int(row["mode"]) == 0
        else float(row["measured_tcp_pz"])
        for row in rows
    ]
    error_mm = [float(row["error_mm"]) for row in rows]
    mode = [int(row["mode"]) for row in rows]

    first_failsafe = first_time(validation_rows, "mode")
    first_trigger = first_time(validation_rows, "predicted_trigger")

    out_paths = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, measured_pz, label="measured pz used for comparison", lw=1.3)
    axes[0].plot(t, reference_pz, label="executed reference pz", lw=1.1)
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, error_mm, color="tab:red", label="reference - measured [mm]")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("error [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t, mode, drawstyle="steps-post", label="mode")
    axes[2].set_xlabel("wall_time_sec")
    axes[2].set_ylabel("mode")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    for ax in axes:
        if first_trigger is not None:
            ax.axvline(first_trigger, color="orange", linestyle="--", lw=1.0)
        if first_failsafe is not None:
            ax.axvline(first_failsafe, color="purple", linestyle="--", lw=1.0)

    fig.suptitle("Actual executed reference vs measured pz")
    fig.tight_layout()
    out_path = run_dir / "executed_reference_vs_measured_pz.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)

    if first_failsafe is not None:
        xmin = first_failsafe - window
        xmax = first_failsafe + window
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, measured_pz, label="measured pz", lw=1.3)
        ax.plot(t, reference_pz, label="executed reference pz", lw=1.1)
        if first_trigger is not None:
            ax.axvline(first_trigger, color="orange", linestyle="--", label="first trigger")
        ax.axvline(first_failsafe, color="purple", linestyle="--", label="first mode=1")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("wall_time_sec")
        ax.set_ylabel("z [m]")
        ax.set_title("Executed reference zoom around first failsafe")
        ax.legend(loc="best")
        ax.grid(True)
        fig.tight_layout()
        out_path = run_dir / "executed_reference_vs_measured_pz_zoom.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def make_mode1_nominal_plot(
    run_dir: Path,
    comparison: list[dict[str, float | str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in comparison
        if isinstance(row["nominal_pred_next_pz"], float)
        and math.isfinite(row["nominal_pred_next_pz"])
    ]
    if not rows:
        return []

    t = [float(row["t"]) for row in rows]
    measured = [float(row["measured_collision_center_pz"]) for row in rows]
    pred = [float(row["nominal_pred_next_pz"]) for row in rows]
    target = [float(row["nominal_collision_target_pz"]) for row in rows]
    executed_failsafe = [float(row["executed_failsafe_des_pz"]) for row in rows]
    error_mm = [float(row["nominal_pred_error_mm"]) for row in rows]
    trigger = [int(row["predicted_trigger"]) for row in rows]
    candidate_verified = [int(row["candidate_verified"]) for row in rows]

    out_paths = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, measured, label="measured collision_center_pz", lw=1.3)
    axes[0].plot(t, pred, label="monitored nominal pred_next_pz", lw=1.1)
    axes[0].plot(t, target, label="monitored nominal target_pz", lw=1.0, alpha=0.8)
    axes[0].plot(t, executed_failsafe, label="executed failsafe des_pz", lw=1.0, alpha=0.8)
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, error_mm, color="tab:red", label="nominal pred_next - measured [mm]")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("error [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t, trigger, drawstyle="steps-post", label="predicted_trigger")
    axes[2].plot(t, candidate_verified, drawstyle="steps-post", label="candidate_verified")
    axes[2].set_xlabel("wall_time_sec")
    axes[2].set_ylabel("state")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    fig.suptitle("Mode=1: monitored nominal prediction vs measured pose")
    fig.tight_layout()
    out_path = run_dir / "mode1_nominal_prediction_vs_measured_pz.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)
    return out_paths


def make_verification_aligned_plot(
    run_dir: Path,
    comparison: list[dict[str, float | str]],
    validation_rows: list[dict[str, str]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in comparison
        if isinstance(row["pred_next_pz"], float)
        and math.isfinite(row["pred_next_pz"])
    ]
    if not rows:
        return []

    t = [float(row["t"]) for row in rows]
    measured = [float(row["measured_collision_center_pz"]) for row in rows]
    pred = [float(row["pred_next_pz"]) for row in rows]
    target = [float(row["collision_target_pz"]) for row in rows]
    error_mm = [float(row["prediction_error_mm"]) for row in rows]
    mode = [int(row["mode"]) for row in rows]
    stage_numeric = [0 if row["prediction_stage"] == "intended" else 1 for row in rows]

    first_failsafe = first_time(validation_rows, "mode")
    first_trigger = first_time(validation_rows, "predicted_trigger")

    out_paths = []
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(t, measured, label="measured collision_center_pz", lw=1.3)
    axes[0].plot(t, pred, label="verified prediction pred_next_pz", lw=1.1)
    axes[0].plot(t, target, label="verified prediction target_pz", lw=1.0, alpha=0.8)
    axes[0].set_ylabel("z [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, error_mm, color="tab:red", label="pred_next - measured [mm]")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("error [mm]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    axes[2].plot(t, mode, drawstyle="steps-post", label="measured mode")
    axes[2].plot(t, stage_numeric, drawstyle="steps-post", label="prediction stage: 0 intended, 1 failsafe")
    axes[2].set_ylabel("state")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    axes[3].plot(
        t,
        [1000.0 * float(row["time_error_sec"]) for row in rows],
        label="prediction absolute time - measured time [ms]",
    )
    axes[3].axhline(0.0, color="black", lw=0.8)
    axes[3].set_xlabel("wall_time_sec")
    axes[3].set_ylabel("time err [ms]")
    axes[3].legend(loc="best")
    axes[3].grid(True)

    for ax in axes:
        if first_trigger is not None:
            ax.axvline(first_trigger, color="orange", linestyle="--", lw=1.0)
        if first_failsafe is not None:
            ax.axvline(first_failsafe, color="purple", linestyle="--", lw=1.0)

    fig.suptitle("Verification-aligned prediction vs measured pz")
    fig.tight_layout()
    out_path = run_dir / "verification_aligned_prediction_vs_measured_pz.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    out_paths.append(out_path)
    return out_paths


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else newest_complete_run(Path(args.data_root))
    validation_path = run_dir / VALIDATION_CSV
    prediction_path = run_dir / PREDICTION_CSV

    validation_rows = read_csv(validation_path)
    prediction_rows = read_csv(prediction_path)
    comparison = build_comparison(
        validation_rows,
        prediction_rows,
        args.stage,
        args.index,
    )
    executed_comparison = build_executed_reference_comparison(
        validation_rows,
        prediction_rows,
        args.match_tolerance,
    )
    mode1_nominal_comparison = build_mode1_nominal_prediction_comparison(
        validation_rows,
        prediction_rows,
        args.match_tolerance,
    )
    verification_aligned_comparison = build_verification_aligned_comparison(
        validation_rows,
        prediction_rows,
        args.match_tolerance,
    )

    comparison_path = write_comparison_csv(run_dir, comparison)
    executed_comparison_path = write_executed_reference_csv(
        run_dir,
        executed_comparison,
    )
    mode1_nominal_path = write_mode1_nominal_csv(
        run_dir,
        mode1_nominal_comparison,
    )
    verification_aligned_path = write_verification_aligned_csv(
        run_dir,
        verification_aligned_comparison,
    )
    summary = summarize(
        run_dir,
        validation_rows,
        prediction_rows,
        comparison,
        args.stage,
        args.index,
    )
    summary_path = run_dir / "shield_prediction_summary.txt"
    summary_path.write_text(summary)

    print(summary, end="")
    print(f"comparison_csv: {comparison_path}")
    print(f"executed_reference_csv: {executed_comparison_path}")
    print(f"mode1_nominal_prediction_csv: {mode1_nominal_path}")
    print(f"verification_aligned_csv: {verification_aligned_path}")
    print(f"summary_file: {summary_path}")

    if not args.no_plot:
        try:
            plot_paths = []
            if not args.mode1_only:
                plot_paths.extend(
                    make_plots(run_dir, validation_rows, comparison, args.window)
                )
                plot_paths.extend(
                    make_executed_reference_plot(
                        run_dir,
                        executed_comparison,
                        validation_rows,
                        args.window,
                    )
                )
            plot_paths.extend(
                make_mode1_nominal_plot(run_dir, mode1_nominal_comparison)
            )
            plot_paths.extend(
                make_verification_aligned_plot(
                    run_dir,
                    verification_aligned_comparison,
                    validation_rows,
                )
            )
            for path in plot_paths:
                print(f"plot: {path}")
        except ImportError as exc:
            print(f"plot_skipped: {exc}")
            print("Install matplotlib in the container to create PNG plots.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
