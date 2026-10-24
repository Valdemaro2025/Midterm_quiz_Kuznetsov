"""Frozen Stage 5 strategy parameters carried from Stage 4 selection."""

from __future__ import annotations

REPEAT_PARAMS = {
    "name": "repeat_50",
    "count_variant": "sens50",
    "threshold": ">=50%",
    "ranking": "training_qualifying_airings_sensitivity",
    "tie_break": "training_popularity_then_id",
}

AR_PARAMS = {
    "name": "ar_c5_conf_max_80",
    "interaction_counts": "qualifying_airings_primary_>=80%",
    "min_count": 5,
    "min_confidence": 0.005,
    "min_lift": 1.0,
    "max_rules_per_antecedent": 50,
    "score_scheme": "confidence",
    "aggregate": "max",
    "max_antecedents_per_user": 50,
    "tie_break": "training_popularity_then_id",
}

HYBRID_PARAMS = {
    "name": "hybrid_70_30",
    "repeat_source": "repeat_50",
    "association_source": "ar_c5_conf_max_80",
    "repeat_weight": 0.7,
    "association_weight": 0.3,
    "normalization": "per-user max normalization over eligible candidate scores",
    "fusion": "weighted sum of normalized scores; no score rounding",
    "tie_break": "exact_score_desc, training_popularity_desc, program_id_asc",
    "zero_score_handling": "scores <= 0 are not ranked",
    "fallback": "full-history training popularity restricted to test candidates",
}

POPULARITY_PARAMS = {
    "name": "popularity_fallback",
    "count_variant": "primary80",
    "ranking": "distinct_full_history_training_users_primary_desc, qualifying_airings_primary_desc, program_id_asc",
}

SELECTED_STRATEGY = "hybrid_70_30"
PRODUCTION_ALTERNATIVE = "repeat_50"
