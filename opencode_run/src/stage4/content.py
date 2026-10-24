"""Content-based scores with Stage 4 tie-breaking."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from stage3.content import _row_normalize, build_content_features

from .ranking import top_k_from_vector


def _weight(value: int, scheme: str) -> float:
    if scheme == "binary":
        return 1.0
    if scheme == "count":
        return float(value)
    if scheme == "logcount":
        return math.log1p(value)
    raise ValueError(f"Unknown profile weight: {scheme}")


def content_scores(
    data: Mapping[str, Any],
    fields: Sequence[str],
    profile_weight: str,
    actor_limit: int = 300,
    use_idf: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    cohort = list(data["cohort"])
    candidates = list(data["val_candidates"])
    metadata = data["metadata"]
    user_items = data["count_variants"]["positive"]
    user_to_train_idx = data["user_to_train_idx"]

    matrix, meta_show_ids, feature_index = build_content_features(
        metadata, fields, actor_limit=actor_limit, use_idf=use_idf
    )
    meta_row = {show_id: idx for idx, show_id in enumerate(meta_show_ids)}
    cand_cols = [idx for idx, show_id in enumerate(candidates) if show_id in meta_row]
    cand_rows = [meta_row[candidates[idx]] for idx in cand_cols]
    candidate_matrix = (
        matrix[cand_rows] if cand_rows else csr_matrix((0, matrix.shape[1]))
    )

    profile_rows: list[int] = []
    profile_cols: list[int] = []
    profile_values: list[float] = []
    users_with_metadata = 0
    for user_row, uid in enumerate(cohort):
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        used = 0
        for show_id, count in items.items():
            row = meta_row.get(show_id)
            if row is None:
                continue
            profile_rows.append(user_row)
            profile_cols.append(row)
            profile_values.append(_weight(int(count), profile_weight))
            used += 1
        if used:
            users_with_metadata += 1

    profile_weights = csr_matrix(
        (np.asarray(profile_values, dtype=np.float32), (profile_rows, profile_cols)),
        shape=(len(cohort), matrix.shape[0]),
    )
    profiles = _row_normalize(profile_weights @ matrix)
    scores = np.zeros((len(cohort), len(candidates)), dtype=np.float32)
    if cand_rows and len(cohort):
        scores[:, np.asarray(cand_cols)] = (profiles @ candidate_matrix.T).toarray()

    return scores, {
        "features": list(fields),
        "profile_weight": profile_weight,
        "actor_limit": actor_limit,
        "use_idf": use_idf,
        "feature_count": len(feature_index),
        "metadata_show_count": len(meta_show_ids),
        "candidates_with_training_metadata": len(cand_rows),
        "users_with_training_metadata": users_with_metadata,
    }


def content_predictions(
    data: Mapping[str, Any],
    fields: Sequence[str],
    profile_weight: str,
    actor_limit: int = 300,
    use_idf: bool = True,
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    scores, detail = content_scores(
        data, fields, profile_weight, actor_limit=actor_limit, use_idf=use_idf
    )
    cohort = list(data["cohort"])
    candidates = list(data["val_candidates"])
    popularity = list(data["popularity_val"])
    stats = data["popularity"]
    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_total = 0
    for row, uid in enumerate(cohort):
        top, info = top_k_from_vector(
            scores[row], candidates, popularity, stats, k=k, tie_break="popularity"
        )
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_total += int(info["scored_candidates"])
    detail.update(
        {
            "tie_break": "exact_score_desc, training_popularity_desc, program_id_asc",
            "popularity_source": "training distinct users with >=80% qualifying airings",
            "popularity_is_score_component": False,
            "users_needing_fallback": fallback_users,
            "mean_scored_candidates_per_user": scored_total / len(cohort),
        }
    )
    return predictions, detail
