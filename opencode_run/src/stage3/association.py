"""Item-to-item association rules fitted on training interactions only."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .scoring import top_k_from_scores


def build_association_index(data: Mapping[str, Any]) -> dict[str, Any]:
    user_items = data["user_items"]
    train_indices = sorted(user_items)
    train_row = {train_idx: idx for idx, train_idx in enumerate(train_indices)}
    item_ids = sorted({show_id for items in user_items.values() for show_id in items})
    item_col = {show_id: idx for idx, show_id in enumerate(item_ids)}

    rows: list[int] = []
    cols: list[int] = []
    for train_idx in train_indices:
        for show_id in user_items[train_idx]:
            rows.append(train_row[train_idx])
            cols.append(item_col[show_id])
    interactions = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_indices), len(item_ids)),
    )
    interactions.data[:] = 1.0
    cooccurrence = (interactions.T @ interactions).tocsr()
    item_totals = np.asarray(interactions.sum(axis=0)).ravel().astype(np.float64)
    return {
        "user_items": user_items,
        "train_indices": train_indices,
        "train_row": train_row,
        "item_ids": item_ids,
        "item_col": item_col,
        "interactions": interactions,
        "cooccurrence": cooccurrence,
        "item_totals": item_totals,
        "n_users": len(train_indices),
    }


def _rule_score(cooc: np.ndarray, total_i: np.ndarray, total_j: np.ndarray, n_users: int, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    confidence = cooc / total_i
    lift = confidence / (total_j / n_users)
    if scheme == "confidence":
        score = confidence
    elif scheme == "lift":
        score = lift
    elif scheme == "conf_x_loglift":
        score = confidence * np.log1p(lift)
    else:
        raise ValueError(f"Unknown association score: {scheme}")
    return score, lift


def _build_rules(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
    min_count: int,
    min_confidence: float,
    min_lift: float,
    max_rules: int,
    score_scheme: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    candidate_ids = data["val_candidates"]
    candidate_set = set(candidate_ids)
    cand_idx_by_item = np.full(len(index["item_ids"]), -1, dtype=np.int64)
    for idx, show_id in enumerate(index["item_ids"]):
        if show_id in candidate_set:
            cand_idx_by_item[idx] = int(show_id)

    cooccurrence: csr_matrix = index["cooccurrence"]
    item_totals = index["item_totals"]
    n_users = int(index["n_users"])
    rules: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for item_idx in range(len(index["item_ids"])):
        row = cooccurrence.getrow(item_idx)
        js = row.indices
        cooc = row.data.astype(np.float64)
        if js.size == 0:
            continue
        mask = (
            (cooc >= min_count)
            & (cand_idx_by_item[js] >= 0)
            & (js != item_idx)
        )
        if not mask.any():
            continue
        js = js[mask]
        cooc = cooc[mask]
        total_i = item_totals[item_idx]
        if total_i <= 0:
            continue
        scores, lifts = _rule_score(
            cooc, total_i, item_totals[js], n_users, score_scheme
        )
        confidence = cooc / total_i
        keep = (confidence >= min_confidence) & (lifts >= min_lift)
        if not keep.any():
            continue
        js = js[keep]
        scores = scores[keep]
        if js.size > max_rules:
            order = np.lexsort((js, -scores))[:max_rules]
            js = js[order]
            scores = scores[order]
        else:
            order = np.lexsort((js, -scores))
            js = js[order]
            scores = scores[order]
        rules[item_idx] = (
            np.asarray([cand_idx_by_item[j] for j in js], dtype=np.int64),
            scores.astype(np.float64),
        )
    return rules


def association_predictions(
    data: Mapping[str, Any],
    index: Mapping[str, Any],
    min_count: int,
    min_confidence: float,
    min_lift: float,
    max_rules_per_antecedent: int,
    score_scheme: str,
    aggregate: str,
    max_antecedents_per_user: int,
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    if aggregate not in {"sum", "max"}:
        raise ValueError(f"Unknown aggregation: {aggregate}")
    rules = _build_rules(
        data,
        index,
        min_count,
        min_confidence,
        min_lift,
        max_rules_per_antecedent,
        score_scheme,
    )
    cohort = list(data["cohort"])
    candidate_ids = list(data["val_candidates"])
    popularity = list(data["popularity_val"])
    user_to_train_idx = data["user_to_train_idx"]
    user_items = data["user_items"]
    item_col = index["item_col"]

    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    users_with_rules = 0
    scored_total = 0
    for uid in cohort:
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        antecedents = sorted(
            ((count, item_col[show_id]) for show_id, count in items.items() if show_id in item_col),
            key=lambda pair: (-pair[0], pair[1]),
        )[:max_antecedents_per_user]
        scores: dict[int, float] = {}
        for _, item_idx in antecedents:
            rule = rules.get(item_idx)
            if rule is None:
                continue
            candidate_indices, rule_scores = rule
            for cand_idx, rule_score in zip(candidate_indices, rule_scores):
                previous = scores.get(int(cand_idx))
                if aggregate == "sum":
                    scores[int(cand_idx)] = (previous or 0.0) + float(rule_score)
                elif previous is None or rule_score > previous:
                    scores[int(cand_idx)] = float(rule_score)
        if scores:
            users_with_rules += 1
        top, info = top_k_from_scores(scores, candidate_ids, popularity, k=k)
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])

    return predictions, {
        "method": "association_rules",
        "min_count": min_count,
        "min_confidence": min_confidence,
        "min_lift": min_lift,
        "max_rules_per_antecedent": max_rules_per_antecedent,
        "score": score_scheme,
        "aggregate": aggregate,
        "max_antecedents_per_user": max_antecedents_per_user,
        "training_users": int(index["n_users"]),
        "training_items": len(index["item_ids"]),
        "antecedents_with_rules": len(rules),
        "total_rules": int(sum(len(rule[0]) for rule in rules.values())),
        "users_with_rules": users_with_rules,
        "users_needing_fallback": fallback_users,
        "mean_scored_candidates_per_user": (
            scored_total / len(cohort) if cohort else 0.0
        ),
    }
