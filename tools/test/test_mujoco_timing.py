import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_mujoco_timing import analyze


def sample(sequence, steady, sim_time, **values):
    row = dict(channel="physics", sequence=sequence, steady_ns=int(steady * 1e9), sim_time=sim_time,
               interval_ms=1, elapsed_ms=.2, cpu_ms=.19, wake_late_ms=.05, lock_wait_ms=0,
               step_ms=.15, clock_publish_ms=.03, callbacks_ms=.01, render_snapshot_ms=.01,
               lock_misses=0, steps=1, render_busy=0, render_requests=0, running=1, dropped_total=0)
    row.update(values)
    return {k: str(v) for k, v in row.items()}


def test_warmup_is_wall_time_and_paused_rows_are_excluded():
    rows = [sample(1, 10, 0, running=0), sample(2, 20, .001), sample(3, 21, .002), sample(4, 22.001, .003)]
    result = analyze(rows, 2)
    assert result["channels"]["physics"]["samples"] == 1


def test_progress_excludes_pause_and_dropped_intervals():
    rows = [sample(1, 1, .001), sample(2, 1.001, .002), sample(3, 2, .002, running=0),
            sample(4, 3, .002), sample(5, 3.001, .003), sample(7, 4, .004)]
    assert analyze(rows, 0)["simulation_seconds_per_wall_second"] == pytest.approx(1)


def test_reports_stalls_and_busy_submissions_separately():
    result = analyze([sample(1, 1, .001), sample(2, 1.024, .002, interval_ms=24, lock_misses=4, render_busy=1)], 0)
    assert result["physics_intervals_over_ms"]["5"] == 1
    assert result["physics_lock_misses"] == 4
    assert result["render_busy_physics_iterations"] == 1
    assert result["largest_physics_intervals"][0]["sequence"] == "2"


def test_empty_running_log_has_actionable_error():
    with pytest.raises(ValueError, match="unpaused"):
        analyze([sample(1, 0, 0, running=0)])
