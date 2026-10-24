"""Frozen Stage 5 prediction path, shared by final refit and reproduction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from stage3.association import build_association_index
from stage3.baselines import popularity_predictions
from stage4.association import association_scores
from stage4.hybrid import fused_predictions
from stage4.ranking import top_k
from stage4.repeat import repeat_predictions, repeat_scores

from .frozen import AR_PARAMS, HYBRID_PARAMS, REPEAT_PARAMS


def fit_predict_hybrid(
    cohort: Sequence[str],
    count_variants: Mapping[str, Mapping[int, Mapping[int, int]]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
) -> dict[str, Any]:
    repeat_map, repeat_detail = repeat_scores(
        cohort,
        count_variants["sens50"],
        user_to_train_idx,
        candidate_ids,
    )
    ar_data = {
        "cohort": list(cohort),
        "user_to_train_idx": dict(user_to_train_idx),
        "user_items": count_variants["primary80"],
        "val_candidates": list(candidate_ids),
    }
    association_index = build_association_index(ar_data)
    ar_map, ar_detail = association_scores(
        ar_data,
        association_index,
        **{
            key: AR_PARAMS[key]
            for key in (
                "min_count",
                "min_confidence",
                "min_lift",
                "max_rules_per_antecedent",
                "score_scheme",
                "aggregate",
                "max_antecedents_per_user",
            )
        },
    )
    predictions, fusion_detail = fused_predictions(
        cohort,
        repeat_map,
        ar_map,
        candidate_ids,
        popularity_ids,
        popularity_stats,
        repeat_weight=HYBRID_PARAMS["repeat_weight"],
        association_weight=HYBRID_PARAMS["association_weight"],
    )
    return {
        "predictions": predictions,
        "repeat_scores": repeat_map,
        "association_scores": ar_map,
        "association_index": association_index,
        "details": {
            "repeat": repeat_detail,
            "association": ar_detail,
            "fusion": fusion_detail,
        },
    }


def repeat_alternative(
    cohort: Sequence[str],
    count_variants: Mapping[str, Mapping[int, Mapping[int, int]]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    return repeat_predictions(
        cohort,
        count_variants["sens50"],
        user_to_train_idx,
        candidate_ids,
        popularity_ids,
        popularity_stats,
        k=5,
    )


def association_alternative(
    cohort: Sequence[str],
    count_variants: Mapping[str, Mapping[int, Mapping[int, int]]],
    user_to_train_idx: Mapping[str, int],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    ar_data = {
        "cohort": list(cohort),
        "user_to_train_idx": dict(user_to_train_idx),
        "user_items": count_variants["primary80"],
        "val_candidates": list(candidate_ids),
    }
    association_index = build_association_index(ar_data)
    ar_map, detail = association_scores(
        ar_data,
        association_index,
        **{
            key: AR_PARAMS[key]
            for key in (
                "min_count",
                "min_confidence",
                "min_lift",
                "max_rules_per_antecedent",
                "score_scheme",
                "aggregate",
                "max_antecedents_per_user",
            )
        },
    )
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_total = 0
    for uid in cohort:
        top, info = top_k(
            ar_map[uid],
            candidate_ids,
            popularity_ids,
            popularity_stats,
            k=5,
        )
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])
    detail.update(
        {
            "users_needing_fallback": fallback_users,
            "mean_scored_candidates_per_user": scored_total / len(cohort),
            "association_index_users": int(association_index["n_users"]),
            "association_index_items": len(association_index["item_ids"]),
            "association_cooccurrence_nnz": int(association_index["cooccurrence"].nnz),
        }
    )
    return predictions, detail


def popularity_alternative(
    cohort: Sequence[str],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    predictions, detail = popularity_predictions(cohort, candidate_ids, popularity_ids, k=5)
    return predictions, {
        **detail,
        "rule": "full-history training popularity restricted to test candidates",
    }
