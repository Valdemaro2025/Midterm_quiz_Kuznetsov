"""Focused Stage 6 signal, cutoff, and cache-contract checks."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("stage6_study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_coverage_weighting_uses_original_duration_and_inclusive_threshold():
    covered = np.array([19, 20, 35, 50, 100])
    duration = np.full(5, 100)
    ends = np.full(5, 1000)
    np.testing.assert_array_equal(study.airing_weights(covered, duration, ends, 1000, .20, "count"), [0, 1, 1, 1, 1])
    np.testing.assert_allclose(study.airing_weights(covered, duration, ends, 1000, .20, "fraction"), [0, .20, .35, .50, 1])
    np.testing.assert_allclose(study.airing_weights(covered, duration, ends, 1000, .50, "fraction"), [0, 0, 0, .50, 1])


@pytest.mark.parametrize("half_life", [14, 28, 56])
def test_recency_uses_airing_end_at_cutoff_half_life_and_two_half_lives(half_life):
    cutoff = study.FINAL_END
    day = 86400
    ends = np.array([cutoff, cutoff-half_life*day, cutoff-2*half_life*day])
    np.testing.assert_allclose(study.airing_weights([50]*3, [100]*3, ends, cutoff, .5, "count", half_life), [1, .5, .25])
    np.testing.assert_allclose(study.airing_weights([50]*3, [100]*3, ends, cutoff, .2, "fraction", half_life), [.5, .25, .125])


def test_recency_rejects_future_airing_end():
    with pytest.raises(ValueError, match="Future airing end"):
        study.airing_weights([80], [100], [1001], 1000, .2, "fraction", 28)


def test_union_cache_never_reads_bridged_gap_seconds(tmp_path):
    archive = tmp_path / "coverage.npz"
    np.savez(archive, user_idx=[0], airing_idx=[0], covered_sec=[19],
             airing_duration=[100], show_id=[10], bridged_covered_sec=[100])
    path = tmp_path / "airings.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_epoch", "end_epoch", "duration", "tv_show_id"])
        writer.writerow([900, 1000, 100, 10])
    table, _ = study.load_table(archive, path)
    assert "bridged_covered_sec" not in table
    assert study.aggregate(table, 1000, {"threshold": .2, "weighting": "fraction"}) == {}


def test_aggregation_sums_per_airing_coverage_not_seconds_or_bridged_gaps():
    # Same user/program: 50/100 and 50/200 contribute .5 + .25, not 100/300.
    table = {"user_idx": np.array([0, 0]), "show_id": np.array([10, 10]),
             "covered_sec": np.array([50, 50]), "airing_duration": np.array([100, 200]),
             "start": np.array([900, 800]), "end": np.array([1000, 1000])}
    result = study.aggregate(table, 1000, {"threshold": .2, "weighting": "fraction"})
    assert result == {0: {10: .75}}


def test_cutoff_mask_excludes_crossings_and_future_behavior_cannot_change_training():
    table = {"user_idx": np.array([0, 0, 0, 0]), "show_id": np.array([10, 20, 30, 40]),
             "covered_sec": np.array([50, 100, 100, 100]), "airing_duration": np.array([100]*4),
             "start": np.array([900, 950, 1000, 1100]), "end": np.array([1000, 1050, 1100, 1200])}
    train = study.slice_table(table, 0, 1000)
    val = study.slice_table(table, 1000, 1200)
    assert list(train["show_id"]) == [10]
    assert list(val["show_id"]) == [30, 40]
    params = {"threshold": .2, "weighting": "fraction", "half_life_days": 28}
    before = study.aggregate(train, 1000, params)
    table["covered_sec"][1:] = 0
    after = study.aggregate(study.slice_table(table, 0, 1000), 1000, params)
    assert before == after == {0: {10: .5}}


def test_original_duration_contract_rejects_mismatched_cache(tmp_path):
    archive = tmp_path / "bad.npz"
    np.savez(archive, user_idx=[0], airing_idx=[0], covered_sec=[50], airing_duration=[50], show_id=[10])
    path = tmp_path / "airings.csv"
    path.write_text("start_epoch,end_epoch,duration,tv_show_id\n900,1000,100,10\n")
    with pytest.raises(AssertionError):
        study.load_table(archive, path)
