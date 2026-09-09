"""Regression checks for TCP log migration and historical prediction references."""

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "check_latest_shield_data.py"
SPEC = importlib.util.spec_from_file_location("shield_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def load_rows(tmp_path, name, rows):
    path = tmp_path / name
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return analysis.read_csv(path)


@pytest.mark.parametrize("legacy_aliases", [False, True])
def test_all_comparisons_read_tcp_logs_without_duplicate_output(tmp_path, legacy_aliases):
    measured = [
        {"wall_time_sec": t, "cur_pz": 0.20, "des_pz": 0.23,
         "mode": 0, "execution_stage": stage}
        for t, stage in [(1.0, 0), (1.001, 2)]
    ]
    predictions = [
        {"wall_time_sec": t, "sample_t": 0.0, "stage": stage,
         "index": 0, "candidate_verified": 1, "pred_next_pz": 0.21,
         "tcp_target_pz": 0.23}
        for t in [1.0, 1.001] for stage in ["intended", "failsafe"]
    ]
    if legacy_aliases:
        for row in measured:
            row["collision_center_pz"] = row["cur_pz"]
        for row in predictions:
            row["collision_target_pz"] = row.pop("tcp_target_pz")
    measured = load_rows(tmp_path, "validation.csv", measured)
    predictions = load_rows(tmp_path, "prediction.csv", predictions)

    basic = analysis.build_comparison(measured, predictions, "intended", 0)
    executed = analysis.build_executed_reference_comparison(measured, predictions, 0.002)
    fallback = analysis.build_mode1_nominal_prediction_comparison(measured, predictions, 0.002)
    aligned = analysis.build_verification_aligned_comparison(measured, predictions, 0.002)
    assert [len(rows) for rows in [basic, executed, fallback, aligned]] == [2, 2, 1, 2]
    for rows in [basic, executed, fallback, aligned]:
        for row in rows:
            assert row["measured_pz"] == pytest.approx(0.20)
            assert not any("collision" in key for key in row)
            assert "measured_tcp_pz" not in row
    assert basic[0]["pred_error_pz"] == pytest.approx(0.01)
    assert basic[0]["target_error_pz"] == pytest.approx(0.03)
    assert executed[0]["error_mm"] == pytest.approx(10.0)
    assert executed[1]["error_mm"] == pytest.approx(30.0)
    assert fallback[0]["nominal_pred_error_mm"] == pytest.approx(10.0)
    assert aligned[1]["prediction_error_mm"] == pytest.approx(10.0)


def test_historical_two_point_logs_keep_the_original_reference(tmp_path):
    measured = load_rows(tmp_path, "validation.csv", [
        {"wall_time_sec": t, "cur_pz": 0.24, "collision_center_pz": 0.20,
         "des_pz": 0.25, "execution_stage": stage}
        for t, stage in [(1.0, 0), (1.001, 2)]
    ])
    predictions = load_rows(tmp_path, "prediction.csv", [
        {"wall_time_sec": 1.0, "stage": "intended", "index": 0,
         "candidate_verified": 1, "pred_next_pz": 0.21, "collision_target_pz": 0.22}
    ])
    rows = analysis.build_executed_reference_comparison(measured, predictions, 0.002)
    assert rows[0]["measured_pz"] == pytest.approx(0.20)
    assert rows[1]["measured_pz"] == pytest.approx(0.24)
    assert rows[0]["error_mm"] == pytest.approx(10.0)
    assert rows[1]["error_mm"] == pytest.approx(10.0)


def test_missing_tcp_measurement_is_not_silently_zero(tmp_path):
    measured = load_rows(tmp_path, "validation.csv", [
        {"wall_time_sec": 1.0, "cur_pz": "", "execution_stage": 0}
    ])
    predictions = load_rows(tmp_path, "prediction.csv", [
        {"wall_time_sec": 1.0, "stage": "intended", "index": 0,
         "pred_next_pz": 0.21, "tcp_target_pz": 0.23}
    ])
    import math

    row = analysis.build_comparison(measured, predictions, "intended", 0)[0]
    assert math.isnan(row["measured_pz"])
    assert math.isnan(row["pred_error_pz"])
