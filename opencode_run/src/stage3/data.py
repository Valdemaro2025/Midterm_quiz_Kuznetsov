"""Shared data loading for Stage 3 recommenders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from stage2.config import Stage2Config
from stage2.io_utils import read_json


def _read_candidates(config: Stage2Config, name: str) -> list[int]:
    payload = read_json(config.out("candidates", f"candidates_{name}.json"))
    return [int(x) for x in payload["ids"]]


def _read_popularity(config: Stage2Config, name: str) -> list[int]:
    path = config.out("candidates", f"popularity_{name}.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(
        key=lambda row: (
            -int(row["distinct_training_users_primary"]),
            -int(row["qualifying_training_airings_primary"]),
            int(row["tv_show_id"]),
        )
    )
    return [int(row["tv_show_id"]) for row in rows]


def _read_train_interactions(config: Stage2Config) -> tuple[dict[int, dict[int, int]], dict[int, str]]:
    path = config.out("interactions", "train_user_program_counts.csv")
    user_items: dict[int, dict[int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_idx = int(row["user_idx"])
            show_id = int(row["tv_show_id"])
            positive = int(row["positive_airings"])
            if positive <= 0:
                continue
            user_items.setdefault(user_idx, {})[show_id] = positive

    id_maps = read_json(config.out("views", "id_maps.json"))
    user_ids: list[str] = id_maps["train"]["user_ids"]
    return user_items, {idx: uid for idx, uid in enumerate(user_ids)}


def _read_metadata(config: Stage2Config) -> dict[int, dict[str, str]]:
    path = config.out("metadata", "metadata_train.csv")
    metadata: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata[int(row["tv_show_id"])] = row
    return metadata


def load_stage3_data(config: Stage2Config) -> dict[str, Any]:
    """Load only training-derived state plus frozen validation artifacts."""
    labels_path = config.out("labels", "val_labels_frozen.json")
    manifest_path = config.out("labels", "val_labels_manifest.json")
    manifest = read_json(manifest_path)
    if not manifest.get("frozen"):
        raise RuntimeError("Validation labels are not frozen")
    if manifest.get("protocol_version") != config.raw.get("protocol_version"):
        raise RuntimeError("Label protocol does not match config protocol")

    with config.submission_file.open(newline="", encoding="utf-8") as handle:
        cohort = [row["user_id"].strip() for row in csv.DictReader(handle)]

    labels = read_json(labels_path)
    user_items, train_idx_to_user = _read_train_interactions(config)
    user_to_train_idx = {uid: idx for idx, uid in train_idx_to_user.items()}
    metadata = _read_metadata(config)
    val_candidates = _read_candidates(config, "val")
    test_candidates = _read_candidates(config, "test")
    popularity_val = _read_popularity(config, "val")
    popularity_test = _read_popularity(config, "test")

    if set(val_candidates) != set(popularity_val):
        raise RuntimeError("Popularity table does not cover validation candidates")
    if set(test_candidates) != set(popularity_test):
        raise RuntimeError("Popularity table does not cover test candidates")

    return {
        "labels": labels,
        "labels_manifest": manifest,
        "cohort": cohort,
        "user_items": user_items,
        "train_idx_to_user": train_idx_to_user,
        "user_to_train_idx": user_to_train_idx,
        "metadata": metadata,
        "val_candidates": val_candidates,
        "test_candidates": test_candidates,
        "popularity_val": popularity_val,
        "popularity_test": popularity_test,
    }
