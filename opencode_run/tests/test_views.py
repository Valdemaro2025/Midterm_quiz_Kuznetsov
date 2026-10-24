from stage2.config import Window, parse_epoch
from stage2.views import split_view_into_windows


def _ts(value: str) -> int:
    return parse_epoch(value)


def test_view_is_split_at_cutoff_without_extending_time():
    windows = {
        "train": Window("train", _ts("2020-01-01T00:00:00"), _ts("2020-01-01T01:00:00")),
        "val": Window("val", _ts("2020-01-01T01:00:00"), _ts("2020-01-01T02:00:00")),
    }
    parts = split_view_into_windows(
        _ts("2020-01-01T00:50:00"), _ts("2020-01-01T01:10:00"), windows
    )
    assert parts == {
        "train": (_ts("2020-01-01T00:50:00"), _ts("2020-01-01T01:00:00")),
        "val": (_ts("2020-01-01T01:00:00"), _ts("2020-01-01T01:10:00")),
    }
    assert sum(stop - start for start, stop in parts.values()) == 1200


def test_view_on_boundary_belongs_to_later_window_only():
    windows = {
        "train": Window("train", _ts("2020-01-01T00:00:00"), _ts("2020-01-01T01:00:00")),
        "val": Window("val", _ts("2020-01-01T01:00:00"), _ts("2020-01-01T02:00:00")),
    }
    parts = split_view_into_windows(
        _ts("2020-01-01T01:00:00"), _ts("2020-01-01T01:10:00"), windows
    )
    assert parts == {
        "val": (_ts("2020-01-01T01:00:00"), _ts("2020-01-01T01:10:00"))
    }


def test_view_spanning_both_windows_is_not_extended():
    windows = {
        "train": Window("train", _ts("2020-01-01T00:00:00"), _ts("2020-01-01T01:00:00")),
        "val": Window("val", _ts("2020-01-01T01:00:00"), _ts("2020-01-01T02:00:00")),
    }
    parts = split_view_into_windows(
        _ts("2019-12-31T23:00:00"), _ts("2020-01-01T03:00:00"), windows
    )
    assert parts["train"] == (
        _ts("2020-01-01T00:00:00"),
        _ts("2020-01-01T01:00:00"),
    )
    assert parts["val"] == (
        _ts("2020-01-01T01:00:00"),
        _ts("2020-01-01T02:00:00"),
    )
    assert sum(stop - start for start, stop in parts.values()) == 7200


def test_view_entirely_outside_windows_returns_nothing():
    windows = {
        "train": Window("train", _ts("2020-01-01T00:00:00"), _ts("2020-01-01T01:00:00")),
    }
    assert split_view_into_windows(
        _ts("2020-01-01T02:00:00"), _ts("2020-01-01T03:00:00"), windows
    ) == {}
