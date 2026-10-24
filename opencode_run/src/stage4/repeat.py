"""Repeat-frequency methods using explicit training coverage thresholds."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .ranking import top_k


def repeat_scores(
    cohort: Sequence[str],
    user_items: Mapping[int, Mapping[int, int]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    allowed = set(int(x) for x in candidate_ids)
    scores: dict[str, dict[int, float]] = {}
    users_with_history = 0
    for uid in cohort:
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        user_scores = {
            int(show_id): float(count)
            for show_id, count in items.items()
            if int(show_id) in allowed and count > 0
        }
        if user_scores:
            users_with_history += 1
        scores[uid] = user_scores
    return scores, {"users_with_scored_candidates": users_with_history}


def repeat_predictions(
    cohort: Sequence[str],
    user_items: Mapping[int, Mapping[int, int]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    scores, detail = repeat_scores(
        cohort, user_items, user_to_train_idx, candidate_ids
    )
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_total = 0
    for uid in cohort:
        top, info = top_k(
            scores[uid], candidate_ids, popularity_ids, popularity_stats, k=k
        )
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])
    detail.update(
        {
            "users_needing_fallback": fallback_users,
            "mean_scored_candidates_per_user": scored_total / len(cohort),
        }
    )
    return predictions, detail
