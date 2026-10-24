import csv
from pathlib import Path

from stage2.config import Window, parse_epoch
from stage2.schedule import build_airings, load_schedule_file


HEADER = [
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


def write_schedule(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def row(start, duration, show_id, title="T"):
    return ["C1", "Channel", start, duration, title, show_id, "Cat", "", "", "", "", "", ""]


def test_ownership_spillover_crossing_and_invalid_rows(tmp_path):
    path = tmp_path / "schedule.csv"
    write_schedule(
        path,
        [
            row("31.12.2019 23:00:00", 3600, "900"),  # spillover before
            row("01.01.2020 00:00:00", 3600, "100"),
            row("01.01.2020 23:00:00", 7200, "500"),  # crosses window end
            row("01.01.2020 12:00:00", 0, "600"),  # invalid duration
            row("01.01.2020 13:00:00", 3600, "0"),  # unlabelled
            row("02.01.2020 00:00:00", 3600, "700"),  # spillover after
        ],
    )
    window = Window("train", parse_epoch("2020-01-01T00:00:00"), parse_epoch("2020-01-02T00:00:00"))
    loaded = load_schedule_file(path, "train", window)
    assert loaded.audit["total_rows"] == 6
    assert loaded.audit["spillover_before"] == 1
    assert loaded.audit["spillover_after"] == 1
    assert loaded.audit["crossing_window_end"] == 1
    assert loaded.audit["invalid_duration"] == 1
    assert loaded.audit["zero_show_id"] == 1
    assert loaded.audit["owned_rows"] == 3

    airings = build_airings(
        loaded,
        include_crossing_end=False,
        airing_csv=tmp_path / "airings.csv",
    )
    assert [a.show_id for a in airings.rows] == ["100"]
    assert airings.audit["excluded"]["crossing_window_end"] == 1
    assert airings.audit["excluded"]["zero_show_id"] == 1

    airings_with_crossing = build_airings(
        loaded,
        include_crossing_end=True,
        airing_csv=tmp_path / "airings_crossing.csv",
    )
    assert [a.show_id for a in airings_with_crossing.rows] == ["100", "500"]


def test_overlapping_airings_query_handles_earlier_long_airing(tmp_path):
    path = tmp_path / "overlap_schedule.csv"
    write_schedule(
        path,
        [
            row("01.01.2020 00:00:00", 3600, "1"),
            row("01.01.2020 00:10:00", 600, "2"),
        ],
    )
    window = Window("train", parse_epoch("2020-01-01T00:00:00"), parse_epoch("2020-01-02T00:00:00"))
    loaded = load_schedule_file(path, "train", window)
    airings = build_airings(
        loaded,
        include_crossing_end=False,
        airing_csv=tmp_path / "overlap_airings.csv",
    )
    index = airings.by_channel["C1"]
    from stage2.interactions import overlapping_airings

    hits = dict(overlapping_airings(index, parse_epoch("2020-01-01T00:50:00"), parse_epoch("2020-01-01T01:00:00")))
    assert hits == {0: 600}
