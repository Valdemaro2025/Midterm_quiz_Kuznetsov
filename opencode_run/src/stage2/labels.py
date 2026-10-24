"""Frozen validation labels, candidate pools, cohort sizes, and fallbacks.

The primary label is a local approximation, not verified Kaggle behaviour:
an airing qualifies at >=80% coverage; each user's programmes are ranked by
the number of qualifying airings; the five highest-frequency programmes are
the primary relevance set (fewer if fewer qualify); ties break by ascending
programme id.  Sensitivity files keep all qualifying programmes and the
any-positive-coverage interpretation separate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Stage2Config
from .io_utils import read_json, sha256_file, write_csv, write_json
from .schedule import Airings


def _load_counts(path: Path) -> dict[int, dict[int, dict[str, int]]]:
    by_user: dict[int, dict[int, dict[str, int]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_idx = int(row["user_idx"])
            show_id = int(row["tv_show_id"])
            by_user[user_idx][show_id] = {
                "primary": int(row["qualifying_airings_primary"]),
                "sensitivity": int(row["qualifying_airings_sensitivity"]),
                "positive": int(row["positive_airings"]),
                "bridged_primary": int(row["bridged_qualifying_airings_primary"]),
                "bridged_positive": int(row["bridged_positive_airings"]),
                "covered": int(row["covered_seconds_total"]),
            }
    return by_user


def _top_k(counts: dict[int, dict[str, int]], field: str, k: int) -> list[int]:
    ranked = [
        (show_id, values[field])
        for show_id, values in counts.items()
        if values.get(field, 0) > 0
    ]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [show_id for show_id, _ in ranked[:k]]


def _all_qualifying(counts: dict[int, dict[str, int]], field: str) -> list[int]:
    return sorted(show_id for show_id, values in counts.items() if values.get(field, 0) > 0)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_labels_and_candidates(
    config: Stage2Config,
    val_airings: Airings,
    val_candidate_airings: Airings,
    test_candidate_airings: Airings,
    train_airings: Airings,
) -> dict[str, Any]:
    id_payload = read_json(config.out("views", "id_maps.json"))
    train_user_ids: list[str] = id_payload["train"]["user_ids"]
    val_user_ids: list[str] = id_payload["val"]["user_ids"]
    val_user_index = {uid: idx for idx, uid in enumerate(val_user_ids)}
    train_user_index = {uid: idx for idx, uid in enumerate(train_user_ids)}

    counts = _load_counts(config.out("interactions", "val_user_program_counts.csv"))
    train_counts = _load_counts(config.out("interactions", "train_user_program_counts.csv"))

    with config.submission_file.open(newline="", encoding="utf-8") as handle:
        cohort = [row["user_id"].strip() for row in csv.DictReader(handle)]
    cohort_set = set(cohort)
    cohort_indices_val = [
        val_user_index[uid] for uid in cohort if uid in val_user_index
    ]
    cohort_indices_train = [
        train_user_index[uid] for uid in cohort if uid in train_user_index
    ]
    missing_from_val = sorted(cohort_set - set(val_user_index))
    missing_from_train = sorted(cohort_set - set(train_user_index))

    primary_top5: dict[str, list[int]] = {}
    primary_all: dict[str, list[int]] = {}
    any_all: dict[str, list[int]] = {}
    sensitivity50_top5: dict[str, list[int]] = {}
    bridged_top5: dict[str, list[int]] = {}
    label_rows = []
    sizes: list[int] = []
    any_sizes: list[int] = []
    for uid in cohort:
        val_idx = val_user_index.get(uid)
        user_counts = counts.get(val_idx, {}) if val_idx is not None else {}
        top5 = _top_k(user_counts, "primary", config.top_k)
        all_primary = _all_qualifying(user_counts, "primary")
        all_any = _all_qualifying(user_counts, "positive")
        sens50 = _top_k(user_counts, "sensitivity", config.top_k)
        bridged = _top_k(user_counts, "bridged_primary", config.top_k)
        primary_top5[uid] = top5
        primary_all[uid] = all_primary
        any_all[uid] = all_any
        sensitivity50_top5[uid] = sens50
        bridged_top5[uid] = bridged
        sizes.append(len(top5))
        any_sizes.append(len(all_any))
        for show_id, values in sorted(user_counts.items()):
            if values["primary"] or values["positive"]:
                label_rows.append(
                    {
                        "user_id": uid,
                        "tv_show_id": show_id,
                        "qualifying_airings_primary_80": values["primary"],
                        "qualifying_airings_sensitivity_50": values["sensitivity"],
                        "positive_airings": values["positive"],
                        "bridged_qualifying_airings_primary": values["bridged_primary"],
                        "covered_seconds_total": values["covered"],
                        "in_primary_top5": int(show_id in top5),
                    }
                )

    label_payload = {
        "primary_top5": primary_top5,
        "primary_all_qualifying": primary_all,
        "sensitivity_any_positive_all": any_all,
        "sensitivity_50_top5": sensitivity50_top5,
        "sensitivity_bridged_80_top5": bridged_top5,
    }
    label_path = config.out("labels", "val_labels_frozen.json")
    write_json(label_path, label_payload)
    write_csv(
        config.out("labels", "val_label_counts.csv"),
        [
            "user_id",
            "tv_show_id",
            "qualifying_airings_primary_80",
            "qualifying_airings_sensitivity_50",
            "positive_airings",
            "bridged_qualifying_airings_primary",
            "covered_seconds_total",
            "in_primary_top5",
        ],
        label_rows,
    )

    label_manifest = {
        "label_file": str(label_path),
        "label_sha256": _canonical_hash(label_payload),
        "protocol_version": config.raw.get("protocol_version", "stage2_v1"),
        "config_file": str(config.config_path),
        "config_sha256": sha256_file(config.config_path),
        "view_split_audit": str(config.out("audit", "view_split_audit.json")),
        "interaction_audit_val": str(config.out("audit", "interaction_audit_val.json")),
        "val_candidate_airings": str(config.out("airings", "candidate_airings_val.csv")),
        "test_candidate_airings": str(config.out("airings", "candidate_airings_test.csv")),
        "train_interaction_airings": str(config.out("airings", "airings_train.csv")),
        "val_interaction_airings": str(config.out("airings", "airings_val.csv")),
        "user_scope": {
            "train_ids": "training view segments only",
            "val_ids": "validation view segments only",
            "source": str(config.out("views", "id_maps.json")),
        },
        "definitions": {
            "primary": (
                "Local approximation: airing coverage >= 0.8; rank programmes by "
                "count of qualifying airings; take top 5 or fewer; tie-break "
                "ascending programme id. Not verified Kaggle behaviour."
            ),
            "sensitivity_all_primary": "All programmes with >=1 airing at >=80% coverage.",
            "sensitivity_any_positive": (
                "All programmes with >=1 airing with any positive observed coverage."
            ),
            "sensitivity_50_top5": (
                "Top 5 by count of airings at >=50% coverage (local threshold sensitivity)."
            ),
            "sensitivity_bridged_80_top5": (
                "Top 5 by count of airings at >=80% coverage when observed sub-intervals "
                "separated by <=60 s are bridged; bridged gap seconds are explicitly a "
                "sensitivity quantity, not primary watched time."
            ),
        },
        "cohort": {
            "submission_users": len(cohort),
            "cohort_users_matched_to_train_views": len(cohort_indices_train),
            "cohort_users_matched_to_val_views": len(cohort_indices_val),
            "missing_from_train_views": len(missing_from_train),
            "missing_from_val_views": len(missing_from_val),
            "missing_from_train_examples": missing_from_train[:10],
            "missing_from_val_examples": missing_from_val[:10],
            "users_with_primary_top5_positive": sum(1 for n in sizes if n > 0),
            "users_with_primary_ge5": sum(1 for n in sizes if n >= 5),
            "users_with_primary_1_to_4": sum(1 for n in sizes if 0 < n < 5),
            "users_with_primary_zero": sum(1 for n in sizes if n == 0),
            "primary_top5_size_distribution": dict(sorted(Counter(sizes).items())),
            "users_with_any_positive": sum(1 for n in any_sizes if n > 0),
            "any_positive_program_count_distribution": dict(
                sorted(Counter(any_sizes).items())
            ),
        },
        "frozen": True,
        "note": (
            "This manifest freezes validation labels. Model/threshold comparisons "
            "must use this same file; do not regenerate labels per model."
        ),
    }
    write_json(config.out("labels", "val_labels_manifest.json"), label_manifest)

    # Candidate pools: start-in-window labelled IDs, including crossing-end airings.
    val_candidates = sorted({int(row.show_id) for row in val_candidate_airings.rows})
    test_candidates = sorted({int(row.show_id) for row in test_candidate_airings.rows})
    train_ids = sorted({int(row.show_id) for row in train_airings.rows})
    val_set, test_set, train_set = set(val_candidates), set(test_candidates), set(train_ids)
    write_json(
        config.out("candidates", "candidates_val.json"),
        {
            "candidate_scope": "distinct labelled show_ids with a positive-duration airing starting in the validation window, including crossing-end airings",
            "count": len(val_candidates),
            "ids": val_candidates,
            "overlap_with_train_ids": len(val_set & train_set),
            "new_vs_train": len(val_set - train_set),
        },
    )
    write_json(
        config.out("candidates", "candidates_test.json"),
        {
            "candidate_scope": "distinct labelled show_ids with a positive-duration airing starting in the test window, including crossing-end airings",
            "count": len(test_candidates),
            "ids": test_candidates,
            "overlap_with_train_ids": len(test_set & train_set),
            "new_vs_train": len(test_set - train_set),
        },
    )

    # Popularity fallback from training only, restricted to each target pool.
    def popularity_for(candidate_ids: list[int]) -> list[dict[str, Any]]:
        user_sets: dict[int, set[int]] = defaultdict(set)
        airing_counts: Counter[int] = Counter()
        for idx, programs in train_counts.items():
            for show_id, values in programs.items():
                if values["primary"] > 0:
                    user_sets[show_id].add(idx)
                    airing_counts[show_id] += values["primary"]
        rows = []
        for show_id in candidate_ids:
            rows.append(
                {
                    "tv_show_id": show_id,
                    "distinct_training_users_primary": len(user_sets.get(show_id, set())),
                    "qualifying_training_airings_primary": airing_counts.get(show_id, 0),
                }
            )
        rows.sort(
            key=lambda r: (
                -r["distinct_training_users_primary"],
                -r["qualifying_training_airings_primary"],
                r["tv_show_id"],
            )
        )
        return rows

    pop_val = popularity_for(val_candidates)
    pop_test = popularity_for(test_candidates)
    pop_fields = [
        "tv_show_id",
        "distinct_training_users_primary",
        "qualifying_training_airings_primary",
    ]
    write_csv(config.out("candidates", "popularity_val.csv"), pop_fields, pop_val)
    write_csv(config.out("candidates", "popularity_test.csv"), pop_fields, pop_test)
    write_json(
        config.out("candidates", "popularity_fallback_test_top5.json"),
        {
            "rule": "top 5 test candidate ids by distinct training users with >=1 qualifying (>=80%) airing; ties by qualifying airing count then ascending id",
            "ids": [r["tv_show_id"] for r in pop_test[:5]],
        },
    )
    write_json(
        config.out("candidates", "popularity_fallback_val_top5.json"),
        {
            "rule": "top 5 validation candidate ids by distinct training users with >=1 qualifying (>=80%) airing; ties by qualifying airing count then ascending id",
            "ids": [r["tv_show_id"] for r in pop_val[:5]],
        },
    )

    # Test-side descriptive outcomes only; no hidden labels exist.
    train_seen_ids = set()
    for programs in train_counts.values():
        train_seen_ids.update(programs)
    test_candidate_set = set(test_candidates)
    test_summary = {
        "test_candidate_count": len(test_candidates),
        "test_candidates_with_any_training_positive": len(test_candidate_set & train_seen_ids),
        "test_candidates_with_no_training_positive": len(test_candidate_set - train_seen_ids),
        "cohort_users": len(cohort),
        "cohort_users_matched_to_train_views": len(cohort_indices_train),
        "cohort_users_with_any_training_positive": sum(
            1
            for idx in cohort_indices_train
            if any(v["positive"] > 0 for v in train_counts.get(idx, {}).values())
        ),
        "note": "No test labels are available locally; these are descriptive candidate/cohort checks, not Kaggle outcomes.",
    }
    write_json(config.out("candidates", "test_candidate_summary.json"), test_summary)

    return {
        "label_manifest": label_manifest,
        "val_candidates": val_candidates,
        "test_candidates": test_candidates,
        "test_summary": test_summary,
    }
