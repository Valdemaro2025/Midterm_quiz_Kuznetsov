import csv
import json
from pathlib import Path

import numpy as np

from stage2.config import load_config
from stage2.interactions import build_interactions
from stage2.labels import build_labels_and_candidates
from stage2.schedule import build_all_schedules
from stage2.views import split_views


SCHEDULE_HEADER = [
    "channel_id",
    "channel_title",
    "start_time",
    "duration",
    "tv_show_title",
    "tv_show_id",
    "tv_show_category",
    "tv_show_genre_1",
    "tv_show_genre_2",
    "tv_show_genre_3",
    "year_of_production",
    "director",
    "actors",
]


def _schedule_row(start, duration, show_id, title, channel="C1"):
    return [
        channel,
        f"Channel {channel}",
        start,
        duration,
        title,
        show_id,
        "Cat",
        "",
        "",
        "",
        "",
        "",
        "",
    ]


def _write_schedule(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCHEDULE_HEADER)
        writer.writerows(rows)


def _write_views(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "vsetv_id", "start_time", "stop_time"])
        writer.writerows(rows)


def _setup_project(tmp_path):
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    data.mkdir()
    config_dir.mkdir()
    _write_schedule(
        data / "train.csv",
        [
            _schedule_row("01.01.2020 00:00:00", 3600, "100", "A"),
            _schedule_row("01.01.2020 01:00:00", 3600, "200", "B"),
            _schedule_row("01.01.2020 02:00:00", 3600, "300", "C"),
            _schedule_row("01.01.2020 03:00:00", 3600, "0", "D"),
            _schedule_row("01.01.2020 04:00:00", 3600, "400", "E"),
            _schedule_row("01.01.2020 23:00:00", 1800, "500", "F"),
            _schedule_row("01.01.2020 23:50:00", 1200, "600", "G-crossing"),
        ],
    )
    _write_schedule(
        data / "val.csv",
        [_schedule_row("02.01.2020 00:00:00", 3600, "700", "H", channel="C2")],
    )
    _write_schedule(
        data / "test.csv",
        [_schedule_row("03.01.2020 00:00:00", 3600, "800", "I")],
    )
    _write_views(
        data / "views.csv",
        [
            # U1 spans A and B, has fragmented C, and a gap in E.
            ["U1", "C1", "2020-01-01 00:30:00", "2020-01-01 02:00:00"],
            ["U1", "C1", "2020-01-01 02:00:00", "2020-01-01 02:10:00"],
            ["U1", "C1", "2020-01-01 02:10:30", "2020-01-01 02:20:30"],
            ["U1", "C1", "2020-01-01 04:00:00", "2020-01-01 04:10:00"],
            ["U1", "C1", "2020-01-01 04:20:00", "2020-01-01 04:30:00"],
            # U2 overlapping observations of A must union, not sum.
            ["U2", "C1", "2020-01-01 00:00:00", "2020-01-01 00:30:00"],
            ["U2", "C1", "2020-01-01 00:15:00", "2020-01-01 00:45:00"],
            # U3 only watches unlabelled show 0.
            ["U3", "C1", "2020-01-01 03:00:00", "2020-01-01 03:30:00"],
            # U4 invalid row, then a view crossing the train cutoff.
            ["U4", "C1", "2020-01-01 05:00:00", "2020-01-01 05:00:00"],
            ["U4", "C1", "2020-01-01 23:50:00", "2020-01-02 00:10:00"],
            ["U4", "C2", "2020-01-02 00:00:00", "2020-01-02 00:10:00"],
            # U1 fully watches validation show H; U5 only has low coverage.
            ["U1", "C2", "2020-01-02 00:00:00", "2020-01-02 01:00:00"],
            ["U5", "C2", "2020-01-02 00:10:00", "2020-01-02 00:20:00"],
        ],
    )
    with (data / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "items"])
        for user in ("U1", "U2", "U3", "U4", "U5"):
            writer.writerow([user, ""])
    config = {
        "data_dir": "data",
        "output_dir": "outputs/stage2",
        "view_file": "views.csv",
        "submission_file": "submission.csv",
        "schedule_files": {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
        "windows": {
            "train": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"],
            "val": ["2020-01-02T00:00:00", "2020-01-03T00:00:00"],
            "test": ["2020-01-03T00:00:00", "2020-01-04T00:00:00"],
        },
        "primary_coverage": 0.8,
        "sensitivity_coverage": 0.5,
        "merge_gap_primary_seconds": 0,
        "merge_gap_sensitivity_seconds": 60,
        "top_k": 5,
        "tie_break": "show_id_asc",
        "primary_evaluation_cohort": "submission_users",
        "allow_repeats_initial": True,
        "popularity_fallback": "global_train_qualifying_user_count",
        "schedule_ownership": "row_start_in_nominal_window; spillover_rows_dropped",
        "interaction_airing_rule": "airing_start_in_window_and_airing_end_at_or_before_window_end",
        "test_candidate_rule": "labelled_airing_start_in_test_window_nonpositive_duration_excluded",
    }
    (config_dir / "stage2.json").write_text(json.dumps(config), encoding="utf-8")
    return config_dir / "stage2.json"


def _rows_by_user_show(npz_path, airings):
    data = np.load(npz_path)
    show_by_airing = {idx: int(a.show_id) for idx, a in enumerate(airings.rows)}
    rows = {}
    for user_idx, airing_idx, covered, duration, bridged in zip(
        data["user_idx"], data["airing_idx"], data["covered_sec"],
        data["airing_duration"], data["bridged_covered_sec"],
    ):
        rows[(int(user_idx), show_by_airing[int(airing_idx)])] = (
            int(covered), int(duration), int(bridged)
        )
    return rows


def test_synthetic_pipeline_counts_and_frozen_labels(tmp_path):
    config = load_config(_setup_project(tmp_path))
    config.ensure_dirs()
    schedules = build_all_schedules(config)
    split = split_views(config)
    train_audit = build_interactions(
        config,
        "train",
        schedules["airings"]["train"],
        split["windows"]["train"]["channel_ids"],
        "train",
    )
    val_audit = build_interactions(
        config,
        "val",
        schedules["airings"]["val"],
        split["windows"]["val"]["channel_ids"],
        "val",
    )
    labels = build_labels_and_candidates(
        config,
        schedules["airings"]["val"],
        schedules["candidate_airings"]["val"],
        schedules["candidate_airings"]["test"],
        schedules["airings"]["train"],
    )

    user_index = split["windows"]["train"]["user_index"]
    val_user_index = split["windows"]["val"]["user_index"]
    assert split["windows"]["train"]["channel_ids"] == ["C1"]
    assert set(split["windows"]["val"]["channel_ids"]) == {"C1", "C2"}
    assert "U5" not in user_index
    assert "U5" in val_user_index
    train_rows = _rows_by_user_show(
        config.out("interactions", "train_user_airing.npz"),
        schedules["airings"]["train"],
    )
    u1, u2, u4 = (user_index[x] for x in ("U1", "U2", "U4"))
    u5_val = val_user_index["U5"]
    u1_val = val_user_index["U1"]
    u4_val = val_user_index["U4"]

    # One view spanning consecutive airings receives coverage in both.
    assert train_rows[(u1, 100)] == (1800, 3600, 1800)
    assert train_rows[(u1, 200)] == (3600, 3600, 3600)
    # Fragmented sub-intervals: actual coverage excludes the 30s gap; bridged sensitivity adds it.
    assert train_rows[(u1, 300)] == (1200, 3600, 1230)
    # A 10-minute unobserved gap is not counted, even by the 60s bridged sensitivity.
    assert train_rows[(u1, 400)] == (1200, 3600, 1200)
    # Overlapping view observations are unioned, not added.
    assert train_rows[(u2, 100)] == (2700, 3600, 2700)
    # Show 0 has no interaction; invalid duration is skipped; crossing view is clipped.
    assert not any(show == 0 for (_, show) in train_rows)
    assert u4 not in {user for (user, _) in train_rows}
    assert "U5" not in user_index

    val_rows = _rows_by_user_show(
        config.out("interactions", "val_user_airing.npz"),
        schedules["airings"]["val"],
    )
    assert val_rows[(u5_val, 700)] == (600, 3600, 600)
    assert val_rows[(u1_val, 700)] == (3600, 3600, 3600)
    assert val_rows[(u4_val, 700)] == (600, 3600, 600)

    assert split["audit"]["clipped_at_train_end"] == 1
    assert split["audit"]["rows_split_across_windows"] == 1
    assert split["audit"]["segments_written"]["train"] == 9
    assert split["audit"]["segments_written"]["val"] == 4
    assert split["audit"]["nonpositive_view_duration"] == 1
    assert train_audit["user_airing_rows"] == 5
    assert train_audit["coverage_primary_qualifying_airings"] == 1
    assert train_audit["coverage_sensitivity_qualifying_airings"] == 3
    assert val_audit["user_airing_rows"] == 3
    assert val_audit["coverage_primary_qualifying_airings"] == 1

    manifest = labels["label_manifest"]
    assert manifest["cohort"]["submission_users"] == 5
    assert manifest["cohort"]["cohort_users_matched_to_train_views"] == 4
    assert manifest["cohort"]["cohort_users_matched_to_val_views"] == 3
    assert manifest["cohort"]["users_with_primary_top5_positive"] == 1
    assert manifest["cohort"]["users_with_primary_zero"] == 4
    assert manifest["cohort"]["users_with_any_positive"] == 3

    import json as jsonlib

    frozen = jsonlib.loads(config.out("labels", "val_labels_frozen.json").read_text())
    assert frozen["primary_top5"]["U1"] == [700]
    assert frozen["primary_top5"]["U2"] == []
    assert frozen["primary_all_qualifying"]["U1"] == [700]
    assert frozen["sensitivity_any_positive_all"]["U1"] == [700]
    assert frozen["sensitivity_any_positive_all"]["U4"] == [700]
    assert frozen["sensitivity_any_positive_all"]["U5"] == [700]
    assert frozen["sensitivity_50_top5"]["U1"] == [700]
    assert frozen["sensitivity_bridged_80_top5"]["U1"] == [700]

    assert labels["val_candidates"] == [700]
    assert labels["test_candidates"] == [800]
    assert labels["test_summary"]["test_candidates_with_no_training_positive"] == 1

    # Schedule boundary audit records the crossing train row and its conflict with val.
    boundary = schedules["audit"]["boundaries"]["adjacent_boundaries"][0]
    assert boundary["left_file"] == "train"
    assert boundary["left_rows_within_24h"] >= 1
