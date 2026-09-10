#!/usr/bin/env python3
"""Summarize MuJoCo outer-loop timing CSVs (all durations in milliseconds).

Example: python3 tools/analyze_mujoco_timing.py data_log/mujoco_A.csv --skip-seconds 2
The warmup cutoff is relative to the first running physics sample, in steady time.
"""
import argparse
import csv
import json
from pathlib import Path

from analyze_async_monitor_timing import distribution


def analyze(rows, skip_seconds=2.0):
    physics = [r for r in rows if r["channel"] == "physics" and int(r["running"]) and int(r["steps"]) > 0]
    if not physics:
        raise ValueError("No advancing, unpaused physics samples in CSV")
    cutoff = int(physics[0]["steady_ns"]) + int(skip_seconds * 1e9)
    selected = [r for r in rows if int(r["steady_ns"]) >= cutoff]
    physics = [r for r in physics if int(r["steady_ns"]) >= cutoff]
    if not physics:
        raise ValueError("No physics samples remain after warmup; reduce --skip-seconds")
    channels = {}
    for name in ("physics", "event", "viewer", "render"):
        samples = physics if name == "physics" else [r for r in selected if r["channel"] == name]
        if not samples:
            continue
        metrics = ("elapsed_ms", "cpu_ms", "lock_wait_ms")
        if name == "physics":
            metrics += ("interval_ms", "wake_late_ms", "step_ms", "clock_publish_ms", "callbacks_ms", "render_snapshot_ms")
        channels[name] = {
            "samples": len(samples),
            "timing_ms": {k: distribution(float(r[k]) for r in samples if float(r[k]) >= 0) for k in metrics},
            "dropped_total": max(int(r["dropped_total"]) for r in samples),
        }
    intervals = [float(r["interval_ms"]) for r in physics]
    # Compute progress only within consecutive unpaused records, excluding pauses,
    # reload/reset discontinuities and dropped telemetry intervals.
    previous = None
    sim_elapsed = wall_elapsed = 0.0
    for row in sorted((r for r in selected if r["channel"] == "physics"), key=lambda r: int(r["sequence"])):
        if previous and int(row["running"]) and int(previous["running"]) and int(row["sequence"]) == int(previous["sequence"]) + 1:
            delta_sim = float(row["sim_time"]) - float(previous["sim_time"])
            delta_wall = (int(row["steady_ns"]) - int(previous["steady_ns"])) / 1e9
            if delta_sim > 0 and delta_wall > 0:
                sim_elapsed += delta_sim
                wall_elapsed += delta_wall
        previous = row
    event_rows = [r for r in selected if r["channel"] == "event"]
    event_span = (max(int(r["steady_ns"]) for r in event_rows) - min(int(r["steady_ns"]) for r in event_rows)) / 1e9 if len(event_rows) > 1 else 0
    keys = ("sequence", "steady_ns", "sim_time", "interval_ms", "wake_late_ms", "lock_wait_ms", "lock_misses", "elapsed_ms", "step_ms", "render_snapshot_ms", "render_busy", "steps")
    return {
        "warmup_seconds": skip_seconds,
        "channels": channels,
        "physics_intervals_over_ms": {str(t): sum(v > t for v in intervals) for t in (1, 2, 5, 10)},
        "physics_lock_misses": sum(int(r["lock_misses"]) for r in physics),
        "physics_steps_per_iteration": distribution(int(r["steps"]) for r in physics),
        "render_busy_physics_iterations": sum(int(r["render_busy"]) for r in physics),
        "render_requests": sum(int(r["render_requests"]) for r in physics),
        "simulation_seconds_per_wall_second": sim_elapsed / wall_elapsed if wall_elapsed else None,
        "event_observed_hz": (len(event_rows) - 1) / event_span if event_span else None,
        "largest_physics_intervals": [{k: r[k] for k in keys} for r in sorted(physics, key=lambda r: float(r["interval_ms"]), reverse=True)[:20]],
        "interpretation": "interval_ms is start-to-start; elapsed_ms and phase times belong to the current iteration. Inspect the previous iteration for work that delayed this start. mj_step includes controller callbacks. render_busy counts skipped submission attempts, not dropped camera frames. Channel rows are buffered independently; join by steady_ns, not CSV order.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--skip-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip_seconds < 0:
        parser.error("--skip-seconds must be nonnegative")
    try:
        with args.csv.open(newline="") as stream:
            result = analyze(list(csv.DictReader(stream)), args.skip_seconds)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    result["csv"] = str(args.csv)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
