import csv
import json

from stage2.config import load_config
from stage2.schedule import ScheduleRow
from stage5.history import build_history_schedule, dedupe_history_rows
from stage5.submission import validate_submission, write_submission


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


def _schedule_row(start, duration, show_id, title="Title", channel="C1"):
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


def _schedule_row_object(source, row, title):
    return ScheduleRow(
        source_file=source,
        source_row=row,
        channel_id="C1",
        channel_title="Channel",
        start=100,
        end=200,
        duration=100,
        show_id="10",
        title=title,
        category="Cat",
        genre_1="",
        genre_2="",
        genre_3="",
        year="",
        director="",
        actors="",
    )


def test_full_history_restores_split_crossing_and_excludes_final_crossing(tmp_path):
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    data.mkdir()
    config_dir.mkdir()
    _write_schedule(
        data / "history_11_20.csv",
        [
            _schedule_row("2020-03-09T00:00:00", 3600, "101"),
            _schedule_row("2020-05-17T23:30:00", 3600, "102"),
        ],
    )
    _write_schedule(
        data / "history_21_30.csv",
        [
            _schedule_row("2020-05-18T00:00:00", 3600, "103"),
            _schedule_row("2020-07-26T23:30:00", 3600, "104"),
            _schedule_row("2020-07-27T00:00:00", 3600, "105"),
            _schedule_row("2020-07-01T00:00:00", 3600, "0"),
        ],
    )
    _write_schedule(data / "test.csv", [_schedule_row("2020-07-27T00:00:00", 3600, "900")])
    config = {
        "data_dir": "data",
        "output_dir": "out",
        "view_file": "views.csv",
        "submission_file": "submission.csv",
        "schedule_files": {
            "history_11_20": "history_11_20.csv",
            "history_21_30": "history_21_30.csv",
            "test": "test.csv",
        },
        "windows": {
            "train": ["2020-03-09T00:00:00", "2020-07-27T00:00:00"],
            "test": ["2020-07-27T00:00:00", "2020-10-19T00:00:00"],
        },
        "schedule_windows": {
            "history_11_20": ["2020-03-09T00:00:00", "2020-05-18T00:00:00"],
            "history_21_30": ["2020-05-18T00:00:00", "2020-07-27T00:00:00"],
            "test": ["2020-07-27T00:00:00", "2020-10-19T00:00:00"],
        },
        "primary_coverage": 0.8,
        "sensitivity_coverage": 0.5,
        "merge_gap_primary_seconds": 0,
        "merge_gap_sensitivity_seconds": 60,
        "top_k": 5,
        "tie_break": "training_popularity_then_id",
        "primary_evaluation_cohort": "submission_users",
        "allow_repeats_initial": True,
        "popularity_fallback": "global_full_history_qualifying_user_count",
        "schedule_ownership": "row_start_in_nominal_window",
        "interaction_airing_rule": "airing_start_in_full_history_window_and_airing_end_at_or_before_window_end",
        "test_candidate_rule": "labelled_positive_duration_airing_start_in_test_window_including_crossing_end",
    }
    config_path = config_dir / "stage5.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(config_path)
    loaded.ensure_dirs()
    history = build_history_schedule(loaded)

    assert history["airings"].audit["selected_airings"] == 3
    assert history["restored_may18_crossing_airings"] == 1
    assert history["audit"]["excluded_full_history_crossing_end"] == 1
    assert history["audit"]["airings"]["excluded"]["zero_show_id"] == 1
    assert [row.show_id for row in history["airings"].rows] == ["101", "102", "103"]


def test_history_dedupe_uses_source_priority_and_reports_conflicts():
    priority = _schedule_row_object("history_11_20", 2, "priority")
    duplicate = _schedule_row_object("history_21_30", 2, "later")
    exact = _schedule_row_object("history_21_30", 3, "priority")
    rows, audit = dedupe_history_rows([(1, duplicate), (1, exact), (0, priority)])

    assert len(rows) == 1
    assert rows[0].source_file == "history_11_20"
    assert audit["exact_duplicates_dropped"] == 1
    assert audit["identity_duplicates_dropped"] == 1
    assert audit["conflicting_identity_duplicates"][0]["kept_source"] == "history_11_20"
    assert audit["conflicting_identity_duplicates"][0]["dropped_source"] == "history_21_30"


def test_submission_round_trip_preserves_long_user_ids_and_format(tmp_path):
    users = [f"90000000000000{i:04d}" for i in range(1260)]
    candidates = list(range(1, 21))
    predictions = {user: [1, 2, 3, 4, 5] for user in users}
    path = tmp_path / "submission.csv"
    write_submission(path, users, predictions)

    audit = validate_submission(path, users, candidates)
    assert audit["all_passed"], audit
    assert audit["row_count"] == 1260
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["user_id", "tv_show_id"]
    assert rows[1] == [users[0], "1 2 3 4 5"]
    assert rows[-1][0] == users[-1]
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_submission_validation_rejects_malformed_rows(tmp_path):
    users = [f"80000000000000{i:04d}" for i in range(1260)]
    candidates = list(range(1, 21))
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        writer.writerow([users[0], "1 2 3 4"])
        writer.writerow([users[1], "1 1 2 3 4"])
        writer.writerow([users[2], "1 2 3 4 999"])
        writer.writerow([users[3], "1 2 3 4 5 6"])
        for user in users[4:]:
            writer.writerow([user, "1 2 3 4 5"])

    audit = validate_submission(path, users, candidates)
    failed = {item["check"] for item in audit["checks"] if not item["ok"]}
    assert audit["all_passed"] is False
    assert "five_tokens_each" in failed
    assert "distinct_ids_each" in failed
    assert "candidate_pool_membership" in failed
