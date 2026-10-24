from stage2.intervals import (
    covered_seconds,
    covered_seconds_with_gap,
    merge_intervals,
    overlap_seconds,
    qualify,
)


def test_half_open_overlap_adjacent_is_zero():
    assert overlap_seconds(0, 10, 10, 20) == 0
    assert overlap_seconds(0, 11, 10, 20) == 1


def test_merge_overlapping_and_adjacent():
    assert merge_intervals([(0, 10), (5, 15), (15, 20)]) == [(0, 20)]
    assert merge_intervals([(0, 10), (11, 20)]) == [(0, 10), (11, 20)]


def test_covered_seconds_does_not_double_count():
    assert covered_seconds([(0, 20), (10, 30)], 0, 60) == 30
    assert covered_seconds([(0, 10), (10, 20)], 0, 60) == 20


def test_positive_gap_is_not_counted_in_actual_coverage():
    actual, bridged, gap = covered_seconds_with_gap(
        [(0, 10), (20, 30)], 0, 60, max_gap=60
    )
    assert actual == 20
    assert bridged == 30
    assert gap == 10


def test_qualify_is_integer_safe():
    assert qualify(800, 1000, 0.8)
    assert not qualify(799, 1000, 0.8)
    assert not qualify(100, 0, 0.8)
