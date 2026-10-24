"""Association-rule scoring that exposes raw scores for Stage 4 fusion."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from stage3.association import _build_rules, build_association_index

from .ranking import top_k


def association_scores(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
    min_count: int,
    min_confidence: float,
    min_lift: float,
    max_rules_per_antecedent: int,
    score_scheme: str,
    aggregate: str,
    max_antecedents_per_user: int,
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    rules = _build_rules(
        data,
        index,
        min_count,
        min_confidence,
        min_lift,
        max_rules_per_antecedent,
        score_scheme,
    )
    user_items = data["user_items"]
    item_col = index["item_col"]
    cohort = list(data["cohort"])
    user_to_train_idx = data["user_to_train_idx"]
    scores: dict[str, dict[int, float]] = {}
    users_with_rules = 0
    for uid in cohort:
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        antecedents = sorted(
            ((count, item_col[show_id]) for show_id, count in items.items() if show_id in item_col),
            key=lambda pair: (-pair[0], pair[1]),
        )[:max_antecedents_per_user]
        user_scores: dict[int, float] = {}
        for _, item_idx in antecedents:
            rule = rules.get(item_idx)
            if rule is None:
                continue
            candidate_indices, rule_scores = rule
            for candidate_index, rule_score in zip(candidate_indices, rule_scores):
                previous = user_scores.get(int(candidate_index))
                if aggregate == "sum":
                    user_scores[int(candidate_index)] = (previous or 0.0) + float(rule_score)
                elif previous is None or rule_score > previous:
                    user_scores[int(candidate_index)] = float(rule_score)
        if user_scores:
            users_with_rules += 1
        scores[uid] = user_scores
    return scores, {
        "antecedents_with_rules": len(rules),
        "total_rules": int(sum(len(rule[0]) for rule in rules.values())),
        "users_with_rules": users_with_rules,
        "training_users": int(index["n_users"]),
        "training_items": len(index["item_ids"]),
        "aggregate": aggregate,
        "score": score_scheme,
        "max_antecedents_per_user": max_antecedents_per_user,
    }


def association_predictions(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
    k: int = 5,
    tie_break: str = "id",
    **params: Any,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    scores, detail = association_scores(data, index, **params)
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_total = 0
    for uid in data["cohort"]:
        top, info = top_k(
            scores[uid],
            candidate_ids,
            popularity_ids,
            popularity_stats,
            k=k,
            tie_break=tie_break,
        )
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])
    detail.update(
        {
            "tie_break": "exact_score_desc, program_id_asc" if tie_break == "id" else tie_break,
            "users_needing_fallback": fallback_users,
            "mean_scored_candidates_per_user": scored_total / len(data["cohort"]),
        }
    )
    return predictions, detail
