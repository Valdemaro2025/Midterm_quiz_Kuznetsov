from stage4.hybrid import fused_predictions
from stage4.ranking import top_k


def test_popularity_tie_break_is_applied_after_exact_score():
    scores = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.1}
    candidates = [1, 2, 3, 4, 5]
    stats = {1: (10, 1), 2: (10, 5), 3: (20, 1), 4: (100, 100), 5: (0, 0)}
    top, info = top_k(scores, candidates, [4, 3, 2, 1, 5], stats, k=5)
    assert top == [3, 2, 1, 4, 5]
    assert info["scored_candidates"] == 4


def test_id_tie_break_preserves_stage3_behavior():
    scores = {3: 1.0, 1: 1.0, 2: 1.0}
    stats = {1: (99, 99), 2: (0, 0), 3: (0, 0)}
    top, _ = top_k(scores, [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], stats, k=5, tie_break="id")
    assert top == [1, 2, 3, 5, 4]


def test_hybrid_max_normalizes_each_method_before_fusion():
    cohort = ["u"]
    repeat = {"u": {1: 10.0, 2: 5.0}}
    association = {"u": {1: 0.1, 3: 0.2}}
    top, details = fused_predictions(
        cohort,
        repeat,
        association,
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        {1: (1, 1), 2: (1, 1), 3: (1, 1), 4: (0, 0), 5: (0, 0)},
        repeat_weight=0.7,
        association_weight=0.3,
    )
    # repeat norm: 1 -> 1.0, 2 -> 0.5; association norm: 1 -> 0.5, 3 -> 1.0
    # fused: 1 -> 0.85, 2 -> 0.35, 3 -> 0.30
    assert top["u"][:3] == [1, 2, 3]
    assert details["users_with_both_sources"] == 1
