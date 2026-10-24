"""Stage 4 data access: reuse frozen Stage 3 artifacts, add count variants."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any, Mapping

from stage2.config import Stage2Config
from stage2.io_utils import read_json
from stage3.data import load_stage3_data


def _read_count_variants(config: Stage2Config) -> dict[str, dict[int, dict[int, int]]]:
    path = config.out("interactions", "train_user_program_counts.csv")
    variants: dict[str, dict[int, dict[int, int]]] = {
        "positive": defaultdict(dict),
        "sens50": defaultdict(dict),
        "primary80": defaultdict(dict),
    }
    fields = {
        "positive": "positive_airings",
        "sens50": "qualifying_airings_sensitivity",
        "primary80": "qualifying_airings_primary",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_idx = int(row["user_idx"])
            show_id = int(row["tv_show_id"])
            for variant, field in fields.items():
                value = int(row[field])
                if value > 0:
                    variants[variant][user_idx][show_id] = value
    return {name: {user: dict(items) for user, items in users.items()} for name, users in variants.items()}


def _read_popularity(config: Stage2Config) -> dict[int, tuple[int, int]]:
    path = config.out("candidates", "popularity_val.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["tv_show_id"]): (
            int(row["distinct_training_users_primary"]),
            int(row["qualifying_training_airings_primary"]),
        )
        for row in rows
    }


def _read_titles(config: Stage2Config) -> dict[int, dict[str, str]]:
    path = config.out("metadata", "metadata_train.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["tv_show_id"]): row
            for row in csv.DictReader(handle)
        }


def load_stage4_data(stage3_config: Stage2Config) -> dict[str, Any]:
    base = load_stage3_data(stage3_config)
    variants = _read_count_variants(stage3_config)
    popularity = _read_popularity(stage3_config)
    titles = _read_titles(stage3_config)

    labels = base["labels"]["primary_top5"]
    cohort = list(base["cohort"])
    user_to_train_idx = base["user_to_train_idx"]
    history: dict[str, dict[str, Any]] = {}
    for uid in cohort:
        idx = user_to_train_idx.get(uid)
        rows = {
            name: users.get(idx, {}) if idx is not None else {}
            for name, users in variants.items()
        }
        history[uid] = {
            "train_idx": idx,
            "distinct_positive": len(rows["positive"]),
            "distinct_sens50": len(rows["sens50"]),
            "distinct_primary80": len(rows["primary80"]),
            "total_positive": sum(rows["positive"].values()),
            "total_sens50": sum(rows["sens50"].values()),
            "total_primary80": sum(rows["primary80"].values()),
            "validation_positive_count": len(set(labels.get(uid, []))),
        }

    return {
        **base,
        "stage3_config": stage3_config,
        "count_variants": variants,
        "popularity": popularity,
        "titles": titles,
        "history": history,
    }
