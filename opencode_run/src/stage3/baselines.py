"""Global popularity and personal repeat-frequency baselines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .scoring import top_k_from_scores


def popularity_predictions(
    cohort: Sequence[str],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    top = [int(x) for x in popularity_ids[:k]]
    return ({uid: list(top) for uid in cohort}, {"fallback_filled": 0})


def repeat_predictions(
    cohort: Sequence[str],
    user_items: Mapping[int, Mapping[int, int]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Rank a user's training programmes by observed positive airings, then fallback."""
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    for uid in cohort:
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        scores = {show_id: float(count) for show_id, count in items.items()}
        top, info = top_k_from_scores(scores, candidate_ids, popularity_ids, k=k)
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
    return predictions, {
        "users_needing_fallback": fallback_users,
    }
