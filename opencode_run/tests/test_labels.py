from stage2.labels import _all_qualifying, _top_k


def make_counts():
    return {
        10: {"primary": 1, "sensitivity": 2, "positive": 2},
        11: {"primary": 3, "sensitivity": 3, "positive": 3},
        12: {"primary": 1, "sensitivity": 1, "positive": 1},
        13: {"primary": 0, "sensitivity": 4, "positive": 4},
    }


def test_top_k_orders_by_count_then_ascending_show_id():
    assert _top_k(make_counts(), "primary", 3) == [11, 10, 12]
    assert _top_k(make_counts(), "primary", 10) == [11, 10, 12]


def test_top_k_ties_are_deterministic():
    assert _top_k(make_counts(), "primary", 2) == [11, 10]


def test_top_k_excludes_zero_counts_and_respects_fewer_than_k():
    assert _top_k(make_counts(), "primary", 5) == [11, 10, 12]
    assert _top_k(make_counts(), "sensitivity", 1) == [13]


def test_all_qualifying_is_sorted_ascending_and_excludes_zero():
    assert _all_qualifying(make_counts(), "primary") == [10, 11, 12]
    assert _all_qualifying(make_counts(), "sensitivity") == [10, 11, 12, 13]
