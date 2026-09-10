"""Prevent cached control verdicts or trajectory samples inflating event counts."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analyze_async_monitor_timing.py"
SPEC = importlib.util.spec_from_file_location("async_timing", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_legacy_uses_worker_verdict_and_detects_same_cycle_invalidation():
    # The control verdict is cached from the accepted result, while this
    # different, also-verified worker result was rejected during handoff.
    control = {"control_loop_sequence": "12.000000000", "candidate_verified": "0",
               "executed_verified_plan_generation": "8", "fallback_reason": "0"}
    event = {"candidate_verified": "1", "accepted_plan_generation": "0",
             "monitor_input_control_loop_sequence": "12", "monitor_source_plan_generation": "7",
             "source_plan_matches_at_handoff": "0", "recovery_epoch_matches_at_handoff": "1",
             "recovery_state_matches_at_handoff": "1", "worker_queue_wait_ms": "0.1",
             "worker_compute_ms": "1.2", "output_handoff_ms": "2.3", "monitor_end_to_end_ms": "3.6"}
    result = analysis.analyze_legacy([control], [event])
    assert result["verified_outputs"] == 1
    assert result["unverified_outputs"] == 0
    assert result["verified_rejection_causes"]["source_generation"] == 1
    assert result["verified_rejections_already_obsolete_at_end_of_input_cycle"] == 1


def test_prediction_rows_are_deduplicated_by_input(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("monitor_input_sequence,stage,index\n1,intended,0\n1,intended,1\n"
                    "1,failsafe,0\n2,intended,0\n")
    assert len(analysis.read_prediction_events(path)) == 2


def test_new_schema_counts_only_processed_events_and_all_mask_bits():
    event = {"async_output_processed_this_cycle": "1.000000000",
             "async_monitor_input_sequence": "3", "async_candidate_verified_at_handoff": "1",
             "async_plan_accepted": "0", "async_handoff_rejection_mask": "33",
             "control_loop_sequence": "20", "async_monitor_input_control_sequence": "9",
             "async_committed_prefix_steps": "8", "fallback_reason": "0",
             "async_monitor_output_handoff_ms": "7.5"}
    cached = dict(event, async_output_processed_this_cycle="0", control_loop_sequence="21")
    result = analysis.analyze_rows([event, cached, cached])
    assert result["processed_outputs"] == 1
    assert result["rejected_verified_outputs"] == 1
    assert result["timing_ms"]["async_monitor_output_handoff_ms"]["count"] == 1
    assert result["timing_ms"]["async_monitor_output_handoff_ms"]["mean"] == 7.5
    causes = result["verified_rejection_masks"]["individual_causes"]
    assert causes["source_generation"] == causes["activation_window_or_continuity"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.analyze_rows([event, event])


def test_distribution_ignores_nonfinite_samples():
    assert analysis.distribution([float("nan"), 2.0, float("inf")])["p99"] == 2.0
    assert analysis.distribution([]) == {"count": 0}


def test_workload_uses_final_horizon_not_sparse_csv_row_count(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("monitor_input_sequence,prediction_horizon_steps,plan_intended_steps,"
                    "plan_failsafe_steps,worker_compute_ms\n"
                    "1,5,6,1,0.7\n1,29,6,1,0.7\n2,5,6,21,2.4\n2,130,6,21,2.4\n")
    result = analysis.computation_variability(analysis.read_prediction_events(path))
    assert set(result["worker_compute_ms_by_workload"]) == {29, 130}
    assert result["workload_compute_pearson_r"] == pytest.approx(1)


def test_cpu_timing_invalid_sentinel_is_not_zero_work():
    rows = [{"worker_rollout_steps": "30", "worker_compute_ms": "1.2",
             "worker_thread_cpu_ms": "-1", "worker_non_cpu_ms": "-1"},
            {"worker_rollout_steps": "30", "worker_compute_ms": "2.4",
             "worker_thread_cpu_ms": "1.1", "worker_non_cpu_ms": "1.3"}]
    result = analysis.computation_variability(rows)
    assert result["workload_compute_pearson_r"] is None
    assert result["timing_ms"]["worker_thread_cpu_ms"]["count"] == 1
    assert result["timing_ms"]["worker_thread_cpu_ms"]["mean"] == pytest.approx(1.1)
