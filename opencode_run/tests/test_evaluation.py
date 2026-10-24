import math

from stage2.evaluation import ap_at_k, label_size_cohorts, map_at_k


def test_no_positives_is_zero():
    assert ap_at_k([1, 2, 3, 4, 5], [], k=5) == 0.0


def test_fewer_than_k_relevant_uses_min_m_k_denominator():
    assert ap_at_k([1, 2, 3, 4, 5], [1, 2], k=5) == 1.0


def test_standard_ap_at_5():
    ap = ap_at_k([1, 9, 2, 8, 3], [1, 2, 3], k=5)
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert math.isclose(ap, expected, rel_tol=1e-12)


def test_duplicate_prediction_counts_once_after_first_occurrence():
    # If duplicates were counted as repeated hits, this would exceed 1.0.
    assert ap_at_k([1, 1, 2, 3, 4], [1], k=5) == 1.0
    assert math.isclose(ap_at_k([1, 1, 2, 3, 4], [2], k=5), 1 / 3)


def test_missing_prediction_positions_are_misses():
    assert ap_at_k([1], [1, 2, 3, 4, 5], k=5) == 1 / 5


def test_map_cohort_breakdown():
    preds = {
        "A": [1, 9, 9, 9, 9],
        "B": [1, 2, 3, 4, 5],
        "C": [1, 2, 3, 4, 5],
    }
    labels = {"A": [1], "B": [1, 2, 3, 4, 5], "C": []}
    result = map_at_k(preds, labels, users=["A", "B", "C"], k=5)
    assert result["cohort_size"] == 3
    assert result["users_with_no_positive"] == 1
    assert result["map_at_k"] == 2 / 3
    assert result["map_with_positive"] == 1.0
    assert result["cohort_with_positive_size"] == 2
    assert result["map_with_at_least_k_positive"] == 1.0
    assert result["cohort_with_at_least_k_positive_size"] == 1


def test_label_size_cohorts():
    assert label_size_cohorts({"a": [], "b": [1], "c": [1, 2, 3, 4], "d": [1, 2, 3, 4, 5]}) == {
        "total": 4,
        "zero": 1,
        "one_to_four": 2,
        "at_least_five": 1,
        "positive": 3,
    }
