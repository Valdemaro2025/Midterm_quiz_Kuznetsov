"""Training-only content-based filtering for Stage 3."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .scoring import top_k_from_vector

_YEAR = re.compile(r"(\d{4})")


def _split_tokens(value: str) -> list[str]:
    return [token.strip().lower() for token in value.replace("|", ",").split(",") if token.strip()]


def _actor_vocabulary(
    metadata: Mapping[int, Mapping[str, str]], limit: int
) -> set[str]:
    counts: Counter[str] = Counter()
    for row in metadata.values():
        counts.update(set(_split_tokens(row.get("actors", ""))))
    return {name for name, _ in counts.most_common(limit)}


def _row_tokens(
    row: Mapping[str, str],
    fields: Sequence[str],
    actors: set[str],
) -> set[str]:
    tokens: set[str] = set()
    if "category" in fields and row.get("category"):
        tokens.add(f"cat:{row['category'].strip().lower()}")
    if "genres" in fields:
        tokens.update(f"genre:{x}" for x in _split_tokens(row.get("genres", "")))
    if "decade" in fields and row.get("year_of_production"):
        match = _YEAR.search(row["year_of_production"])
        if match:
            tokens.add(f"decade:{int(match.group(1)) // 10 * 10}")
    if "director" in fields:
        tokens.update(f"dir:{x}" for x in _split_tokens(row.get("director", "")))
    if "actors" in fields:
        tokens.update(
            f"actor:{x}"
            for x in _split_tokens(row.get("actors", ""))
            if x in actors
        )
    return tokens


def _row_normalize(matrix: csr_matrix) -> csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inv = np.zeros_like(norms, dtype=np.float32)
    nonzero = norms > 0
    inv[nonzero] = 1.0 / norms[nonzero]
    return csr_matrix(matrix.multiply(inv[:, None]))


def build_content_features(
    metadata: Mapping[int, Mapping[str, str]],
    fields: Sequence[str],
    actor_limit: int = 300,
    use_idf: bool = True,
) -> tuple[csr_matrix, list[int], dict[str, int]]:
    show_ids = sorted(metadata)
    actors = _actor_vocabulary(metadata, actor_limit) if "actors" in fields else set()
    feature_index: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for row_idx, show_id in enumerate(show_ids):
        for feature in sorted(_row_tokens(metadata[show_id], fields, actors)):
            if feature not in feature_index:
                feature_index[feature] = len(feature_index)
            rows.append(row_idx)
            cols.append(feature_index[feature])

    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(show_ids), len(feature_index)),
    )
    if use_idf and matrix.shape[0] > 0:
        doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
        idf = np.log((1.0 + matrix.shape[0]) / (1.0 + doc_freq)) + 1.0
        matrix = csr_matrix(matrix.multiply(idf.astype(np.float32)))
    matrix = _row_normalize(matrix)
    return matrix, show_ids, feature_index


def _weight_for(count: int, scheme: str) -> float:
    if scheme == "binary":
        return 1.0
    if scheme == "count":
        return float(count)
    if scheme == "logcount":
        return math.log1p(count)
    raise ValueError(f"Unknown profile weight: {scheme}")


def content_predictions(
    data: Mapping[str, Any],
    fields: Sequence[str],
    profile_weight: str,
    actor_limit: int = 300,
    use_idf: bool = True,
    k: int = 5,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    cohort = list(data["cohort"])
    candidates = list(data["val_candidates"])
    popularity = list(data["popularity_val"])
    metadata = data["metadata"]
    user_items = data["user_items"]
    user_to_train_idx = data["user_to_train_idx"]

    matrix, meta_show_ids, feature_index = build_content_features(
        metadata, fields, actor_limit=actor_limit, use_idf=use_idf
    )
    meta_row = {show_id: idx for idx, show_id in enumerate(meta_show_ids)}

    cand_meta_rows = [meta_row[show_id] for show_id in candidates if show_id in meta_row]
    cand_meta_cols = [idx for idx, show_id in enumerate(candidates) if show_id in meta_row]
    candidate_matrix = matrix[cand_meta_rows] if cand_meta_rows else csr_matrix((0, matrix.shape[1]))

    profile_rows: list[int] = []
    profile_cols: list[int] = []
    profile_data: list[float] = []
    users_with_metadata = 0
    for user_row, uid in enumerate(cohort):
        train_idx = user_to_train_idx.get(uid)
        items = user_items.get(train_idx, {}) if train_idx is not None else {}
        used = 0
        for show_id, count in items.items():
            row_idx = meta_row.get(show_id)
            if row_idx is None:
                continue
            profile_rows.append(user_row)
            profile_cols.append(row_idx)
            profile_data.append(_weight_for(int(count), profile_weight))
            used += 1
        if used:
            users_with_metadata += 1

    profile_weights = csr_matrix(
        (np.asarray(profile_data, dtype=np.float32), (profile_rows, profile_cols)),
        shape=(len(cohort), matrix.shape[0]),
    )
    profiles = _row_normalize(profile_weights @ matrix)

    scores = np.zeros((len(cohort), len(candidates)), dtype=np.float32)
    if cand_meta_rows and len(cohort):
        block = (profiles @ candidate_matrix.T).toarray()
        scores[:, np.asarray(cand_meta_cols)] = block

    predictions: dict[str, list[int]] = {}
    fallback_users = 0
    scored_candidate_total = 0
    for user_row, uid in enumerate(cohort):
        top, info = top_k_from_vector(scores[user_row], candidates, popularity, k=k)
        predictions[uid] = top
        fallback_users += int(info["used_fallback"])
        scored_candidate_total += int(info["scored_candidates"])

    return predictions, {
        "method": "cbf",
        "features": list(fields),
        "profile_weight": profile_weight,
        "actor_limit": actor_limit,
        "use_idf": use_idf,
        "feature_count": len(feature_index),
        "metadata_show_count": len(meta_show_ids),
        "candidates_with_training_metadata": len(cand_meta_rows),
        "users_with_training_metadata": users_with_metadata,
        "users_needing_fallback": fallback_users,
        "mean_scored_candidates_per_user": (
            scored_candidate_total / len(cohort) if cohort else 0.0
        ),
    }
