import numpy as np
from scipy.sparse import csr_matrix

from stage3.association import association_predictions, build_association_index
from stage3.content import build_content_features
from stage3.scoring import top_k_from_scores


def test_top_k_is_unique_deterministic_and_pool_constrained():
    scores = {1: 2.0, 2: 2.0, 3: 1.0, 99: 100.0}
    candidates = [1, 2, 3, 4, 5, 6]
    popularity = [6, 5, 4, 3, 2, 1]
    top, info = top_k_from_scores(scores, candidates, popularity, k=5)
    assert top == [1, 2, 3, 6, 5]
    assert len(set(top)) == 5
    assert info["scored_candidates"] == 3
    assert info["fallback_filled"] == 2
    top_again, _ = top_k_from_scores(scores, candidates, popularity, k=5)
    assert top == top_again


def test_top_k_fills_exactly_five_when_scores_are_empty():
    top, info = top_k_from_scores({}, [3, 1, 2], [2, 1, 3, 4], k=5)
    assert top == [2, 1, 3]
    assert info["fallback_filled"] == 3


def test_content_features_are_normalized_and_use_training_metadata():
    metadata = {
        1: {"category": "Фильмы", "genres": "драма|комедия", "year_of_production": "1999"},
        2: {"category": "Сериалы", "genres": "драма", "year_of_production": "2004"},
        3: {"category": "Фильмы", "genres": "комедия", "year_of_production": "1980-1990"},
    }
    matrix, show_ids, features = build_content_features(
        metadata, ["category", "genres", "decade"], use_idf=True
    )
    assert show_ids == [1, 2, 3]
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(features)
    assert "cat:фильмы" in features
    assert "decade:1980" in features
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)


def test_association_predictions_use_only_training_interactions():
    data = {
        "cohort": ["u0", "u1", "u2"],
        "val_candidates": [10, 20, 30, 40, 50],
        "popularity_val": [10, 20, 30, 40, 50],
        "user_items": {
            0: {10: 3, 20: 2},
            1: {10: 2, 30: 1},
            2: {20: 2, 30: 2},
        },
        "user_to_train_idx": {"u0": 0, "u1": 1, "u2": 2},
    }
    index = build_association_index(data)
    predictions, detail = association_predictions(
        data,
        index,
        min_count=1,
        min_confidence=0.1,
        min_lift=0.5,
        max_rules_per_antecedent=5,
        score_scheme="confidence",
        aggregate="max",
        max_antecedents_per_user=5,
    )
    assert set(predictions) == {"u0", "u1", "u2"}
    assert all(len(items) == 5 for items in predictions.values())
    assert predictions["u0"][0] in {10, 20, 30, 40, 50}
    assert detail["training_items"] == 3
    assert detail["total_rules"] > 0
