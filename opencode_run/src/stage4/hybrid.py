"""Explicit per-user max-normalized score fusion for Stage 4 hybrids."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .ranking import top_k


def _max_normalize(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    maximum = max(float(value) for value in scores.values())
    if maximum <= 0:
        return {}
    return {int(show_id): float(value) / maximum for show_id, value in scores.items()}


def fused_predictions(
    cohort: Sequence[str],
    repeat_scores: Mapping[str, Mapping[int, float]],
    association_scores: Mapping[str, Mapping[int, float]],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
    repeat_weight: float,
    association_weight: float,
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if abs(repeat_weight + association_weight - 1.0) > 1e-12:
        raise ValueError("Fusion weights must sum to 1")
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_total = 0
    users_with_repeat_only = 0
    users_with_association_only = 0
    users_with_both = 0
    for uid in cohort:
        repeat = _max_normalize(repeat_scores.get(uid, {}))
        association = _max_normalize(association_scores.get(uid, {}))
        if repeat and association:
            users_with_both += 1
        elif repeat:
            users_with_repeat_only += 1
        elif association:
            users_with_association_only += 1

        fused: dict[int, float] = {}
        for show_id, score in repeat.items():
            fused[show_id] = fused.get(show_id, 0.0) + repeat_weight * score
        for show_id, score in association.items():
            fused[show_id] = fused.get(show_id, 0.0) + association_weight * score
        top, info = top_k(
            fused, candidate_ids, popularity_ids, popularity_stats, k=k
        )
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])
    return predictions, {
        "repeat_weight": repeat_weight,
        "association_weight": association_weight,
        "normalization": "per-user max normalization of each method's candidate scores",
        "fusion": "weighted sum of normalized scores; no score rounding",
        "tie_break": "exact_score_desc, training_popularity_desc, program_id_asc",
        "users_with_both_sources": users_with_both,
        "users_with_repeat_only": users_with_repeat_only,
        "users_with_association_only": users_with_association_only,
        "users_needing_fallback": fallback_users,
        "mean_scored_candidates_per_user": scored_total / len(cohort),
    }
