#!/usr/bin/env python3
"""Analyze handoff events and computation variability in v15/v20/v21 shield logs.

Usage: python3 tools/analyze_async_monitor_timing.py [data_log/<run>] [--output file.json]
The prediction CSV is checked independently using one row per monitor input.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


VALIDATION = "reachable_cartesian_impedance_validation.csv"
PREDICTION = "shield_prediction_trajectory.csv"
MASKS = {
    1: "source_generation",
    2: "recovery_epoch",
    4: "recovery_state",
    8: "workspace_policy",
    16: "calibration_target",
    32: "activation_window_or_continuity",
}


def number(row, key):
    return float(row[key])


def distribution(values):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return {"count": 0}
    # Nearest-rank quantiles, also defined for a single observation.
    return {
        "count": len(values),
        "mean": mean(values),
        "min": values[0],
        "p50": values[math.ceil(0.50 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "p99": values[math.ceil(0.99 * len(values)) - 1],
        "max": values[-1],
    }


def mask_counts(rows):
    counts = Counter(int(number(r, "async_handoff_rejection_mask")) for r in rows)
    return {
        "exact_masks": dict(sorted(counts.items())),
        "individual_causes": {
            name: sum(n for mask, n in counts.items() if mask & bit)
            for bit, name in MASKS.items()
        },
    }


def analyze_rows(rows):
    if not rows:
        raise ValueError("Validation CSV contains no samples")
    required = {
        "async_output_processed_this_cycle", "async_monitor_input_sequence",
        "async_candidate_verified_at_handoff", "async_plan_accepted",
        "async_handoff_rejection_mask", "control_loop_sequence",
        "async_monitor_input_control_sequence", "async_committed_prefix_steps",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError("Log lacks event diagnostics: " + ", ".join(sorted(missing)))
    events = [r for r in rows if number(r, "async_output_processed_this_cycle") == 1]
    sequences = [int(number(r, "async_monitor_input_sequence")) for r in events]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Duplicate monitor input sequences in processed events")
    verified = [r for r in events if number(r, "async_candidate_verified_at_handoff") == 1]
    rejected = [r for r in verified if number(r, "async_plan_accepted") == 0]
    unverified = [r for r in events if number(r, "async_candidate_verified_at_handoff") == 0]
    result = {
        "control_rows": len(rows),
        "processed_outputs": len(events),
        "verified_outputs": len(verified),
        "accepted_verified_outputs": len(verified) - len(rejected),
        "rejected_verified_outputs": len(rejected),
        "unverified_outputs": len(unverified),
        "all_handoff_masks": mask_counts(events),
        "verified_rejection_masks": mask_counts(rejected),
        "verified_rejection_events": [
            {key: number(r, key) for key in (
                "wall_time_sec", "control_loop_sequence", "async_monitor_input_sequence",
                "async_monitor_input_control_sequence", "async_committed_prefix_steps",
                "async_handoff_rejection_mask", "async_monitor_end_to_end_ms",
            ) if key in r}
            for r in rejected
        ],
        "timing_ms": {},
    }
    for key in (
        "async_monitor_worker_queue_wait_ms", "async_monitor_worker_compute_ms",
        "async_monitor_output_handoff_ms",
        "async_worker_publish_ms", "async_mailbox_wait_ms", "async_acceptance_ms",
        "async_monitor_end_to_end_ms",
        "worker_thread_cpu_ms", "worker_non_cpu_ms",
    ):
        if key in rows[0]:
            result["timing_ms"][key] = distribution(
                number(r, key) for r in events if number(r, key) >= 0
            )
    for key in (
        "control_start_interval_ms", "control_previous_execution_ms",
        "control_elapsed_to_log_ms", "control_goal_ingress_ms", "control_action_status_ms",
    ):
        if key in rows[0]:
            values = [number(r, key) for r in rows]
            result["timing_ms"][key] = {
                **distribution(values), "over_1ms": sum(v > 1 for v in values),
            }
    result["control_cycle_fallback_reasons"] = dict(Counter(
        int(number(r, "fallback_reason")) for r in rows
    ))
    # These are cached controller diagnostics, not an independent worker verdict.
    if "collision_energy_unsafe" in rows[0]:
        result["unverified_events_with_cached_collision_energy_unsafe"] = sum(
            number(r, "collision_energy_unsafe") == 1 for r in unverified
        )
    return result


def read_prediction_events(path):
    unique = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            event = unique.setdefault(int(number(row, "monitor_input_sequence")), row)
            if "prediction_horizon_steps" in row:
                event["_logged_prediction_horizon_steps"] = max(
                    event.get("_logged_prediction_horizon_steps", 0),
                    int(number(row, "prediction_horizon_steps")),
                )
    return list(unique.values())


def accepted(row):
    if "async_plan_accepted" in row:
        return number(row, "async_plan_accepted") == 1
    return number(row, "accepted_plan_generation") > 0


def analyze_legacy(rows, events):
    """v15 prediction rows carry the actual result; control verdicts are cached."""
    verified = [r for r in events if number(r, "candidate_verified") == 1]
    rejected = [r for r in verified if not accepted(r)]
    unverified = [r for r in events if number(r, "candidate_verified") == 0]
    controls = {int(number(r, "control_loop_sequence")): r for r in rows}
    same_cycle_replacements = 0
    for event in rejected:
        control = controls.get(int(number(event, "monitor_input_control_loop_sequence")))
        if control and number(control, "executed_verified_plan_generation") > number(
            event, "monitor_source_plan_generation"
        ):
            same_cycle_replacements += 1
    return {
        "event_source": "unique_prediction_inputs_v15; includes only valid evaluated plans",
        "control_rows": len(rows),
        "processed_outputs_with_logged_plans": len(events),
        "verified_outputs": len(verified),
        "accepted_verified_outputs": len(verified) - len(rejected),
        "rejected_verified_outputs": len(rejected),
        "unverified_outputs": len(unverified),
        "verified_rejection_causes": {
            name: sum(number(r, field) == 0 for r in rejected)
            for name, field in (
                ("source_generation", "source_plan_matches_at_handoff"),
                ("recovery_epoch", "recovery_epoch_matches_at_handoff"),
                ("recovery_state", "recovery_state_matches_at_handoff"),
            )
        },
        "verified_rejections_already_obsolete_at_end_of_input_cycle": same_cycle_replacements,
        "timing_ms": {
            key: distribution(number(r, key) for r in events)
            for key in ("worker_queue_wait_ms", "worker_compute_ms", "output_handoff_ms",
                        "monitor_end_to_end_ms")
        },
        "control_cycle_fallback_reasons": dict(Counter(
            int(number(r, "fallback_reason")) for r in rows
        )),
        "limitations": "v15 does not record separate mailbox, acceptance, or controller wall-clock timings",
    }


def prediction_summary(events):
    unverified = [r for r in events if number(r, "candidate_verified") == 0]
    return {
        "unique_inputs": len(events),
        "verified_and_accepted": sum(
            number(r, "candidate_verified") == 1 and accepted(r)
            for r in events
        ),
        "unverified": len(unverified),
        "unverified_with_contact_possible": sum(
            number(r, "monitored_contact_possible") == 1 for r in unverified
        ),
        "unverified_with_joint_limit_unsafe": sum(
            number(r, "joint_limit_unsafe") == 1 for r in unverified
        ),
        "unverified_worst_case_energy_ub_joule": distribution(
            number(r, "monitor_worst_case_total_control_energy_ub") for r in unverified
        ),
    }


def correlation(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denominator == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def computation_variability(events):
    """Group by actual trace count (v21) or final logged control horizon (v15).

    plan_intended_steps/plan_failsafe_steps count the CSV's sparse view in
    normal mode; they must not be reported as the dense rollout workload.
    """
    if not events:
        return {}
    timings = {}
    for key in ("worker_compute_ms", "planner_ms", "plan_build_ms", "monitor_eval_ms",
                "monitor_total_ms", "worker_thread_cpu_ms", "worker_non_cpu_ms"):
        if key in events[0]:
            timings[key] = distribution(number(r, key) for r in events if number(r, key) >= 0)
    use_trace_count = "worker_rollout_steps" in events[0]
    count_key = "worker_rollout_steps" if use_trace_count else "_logged_prediction_horizon_steps"
    valid = [r for r in events if count_key in r and number(r, count_key) > 0]
    groups = {}
    for event in valid:
        groups.setdefault(int(number(event, count_key)), []).append(number(event, "worker_compute_ms"))
    phase_share = {}
    if "worker_compute_ms" in timings:
        worker_total = sum(number(r, "worker_compute_ms") for r in events)
        if worker_total > 0:
            phase_share = {
                key: sum(number(r, key) for r in events) / worker_total
                for key in ("planner_ms", "plan_build_ms", "monitor_eval_ms") if key in timings
            }
    return {
        "timing_ms": timings,
        "phase_fraction_of_worker_elapsed": phase_share,
        "workload_source": "actual_trace_intervals" if use_trace_count else "final_logged_control_horizon_proxy",
        "workload_steps": distribution(number(r, count_key) for r in valid),
        "workload_compute_pearson_r": correlation(
            [number(r, count_key) for r in valid], [number(r, "worker_compute_ms") for r in valid]
        ),
        "worker_compute_ms_by_workload": {count: distribution(values) for count, values in sorted(groups.items())},
        "interpretation": "Wall time includes preemption and blocking. Same-length outliers alone cannot identify their cause.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run
    if run is None:
        runs = sorted(p for p in Path("data_log").iterdir()
                      if p.is_dir() and (p / VALIDATION).is_file() and (p / PREDICTION).is_file())
        if not runs:
            parser.error("No complete run under data_log")
        run = runs[-1]
    with (run / VALIDATION).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    events = read_prediction_events(run / PREDICTION)
    if rows and "async_output_processed_this_cycle" in rows[0]:
        result = analyze_rows(rows)
    else:
        result = analyze_legacy(rows, events)
    result["run"] = str(run)
    if (run / PREDICTION).is_file():
        result["prediction_cross_check"] = prediction_summary(events)
        result["computation_variability"] = computation_variability(events)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
