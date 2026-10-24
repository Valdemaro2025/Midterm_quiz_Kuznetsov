#!/usr/bin/env python3
"""Correct Stage 3 interpretation and diagnose category-only CBF ties."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402
from stage3.association import build_association_index  # noqa: E402
from stage3.data import load_stage3_data  # noqa: E402
from stage4.content import content_scores  # noqa: E402
from stage4.data import load_stage4_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)
    stage4_config.ensure_dirs()
    data = load_stage4_data(stage3_config)

    stage3_cbf = read_json(stage3_config.out("methods", "cbf_category_log", "top5.json"))
    list_counts = Counter(tuple(items) for items in stage3_cbf.values())
    meta = data["titles"]
    candidates = list(data["val_candidates"])
    stats = data["popularity"]

    scores, detail = content_scores(
        data, ["category"], "logcount", actor_limit=0, use_idf=True
    )
    categories = [
        meta.get(show_id, {}).get("category", "<no training metadata>")
        for show_id in candidates
    ]
    category_index = {name: idx for idx, name in enumerate(sorted(set(categories)))}
    category_of_col = [category_index[name] for name in categories]

    tied_at_max: list[int] = []
    exactly_lowest_five = 0
    users_with_gt5_ties = 0
    top_category_counts = Counter()
    list_rows = []
    for row in range(scores.shape[0]):
        user_scores = scores[row]
        max_score = float(user_scores.max())
        tied_cols = np.flatnonzero(user_scores == max_score)
        tied_at_max.append(int(tied_cols.size))
        if tied_cols.size > 5:
            users_with_gt5_ties += 1
        top_category_counts[categories[tied_cols[0]]] += 1
        top5 = list(stage3_cbf[data["cohort"][row]])
        allowed_top = [candidates[col] for col in tied_cols]
        if set(top5) == set(allowed_top[:5]):
            exactly_lowest_five += 1
        list_rows.append(
            {
                "user_id": data["cohort"][row],
                "max_score": max_score,
                "exact_tied_candidates": int(tied_cols.size),
                "top_category": categories[tied_cols[0]],
                "top5": top5,
            }
        )

    list_categories = {}
    for items, count in list_counts.items():
        list_categories[",".join(map(str, items))] = {
            "count": count,
            "categories": [meta.get(show_id, {}).get("category", "<none>") for show_id in items],
        }

    tie_arr = np.asarray(tied_at_max)
    diagnosis = {
        "stage3_category_cbf": {
            "distinct_top5_lists": len(list_counts),
            "list_counts": [
                {"top5": list(items), "users": count}
                for items, count in list_counts.most_common()
            ],
            "list_categories": list_categories,
            "item_vectors": {
                "candidate_count": len(candidates),
                "distinct_candidate_categories": len(category_index),
                "note": (
                    "With only category features, every candidate in a category has "
                    "an identical sparse vector up to IDF weighting; therefore all "
                    "candidates in that category receive exactly the same score."
                ),
                "category_candidate_counts": dict(
                    Counter(categories).most_common()
                ),
            },
            "exact_ties": {
                "users": len(tied_at_max),
                "min_tied_at_max": int(tie_arr.min()),
                "median_tied_at_max": float(np.median(tie_arr)),
                "max_tied_at_max": int(tie_arr.max()),
                "users_with_more_than_5_tied": users_with_gt5_ties,
                "users_whose_top5_are_exactly_lowest_5_tied_ids": exactly_lowest_five,
                "top_category_counts": dict(top_category_counts.most_common()),
            },
            "conclusion": (
                "The six-list collapse is a tie-handling consequence, not evidence "
                "that category metadata alone is sufficient. Ascending program ID "
                "selects the numerically lowest five candidates within the tied "
                "category/categories."
            ),
            "user_rows": list_rows,
            "score_detail": detail,
        },
        "stage3_interpretation_corrections": {
            "repeat_cbf_association_training_counts": (
                "Stage 3 repeat, CBF profiles, and association-rule interactions read "
                "positive_airings: any positive observed coverage, not the >=80% "
                "threshold."
            ),
            "popularity_training_counts": (
                "Popularity used qualifying_airings_primary: distinct training users "
                "with at least one >=80% qualifying airing; ties by qualifying airing "
                "count, then program ID."
            ),
            "validation_labels": (
                "Unchanged Stage 3 frozen primary labels: fully observed airings at "
                ">=80% coverage; top 5 by qualifying-airing count."
            ),
            "different_training_thresholds_allowed": True,
        },
    }

    started = time.perf_counter()
    shared_index = build_association_index(data)
    shared_seconds = time.perf_counter() - started
    stage3_log = read_json(stage3_config.out("methods", "search_log.json"))
    ar_entries = [
        {
            "name": entry["name"],
            "configuration_scoring_seconds": entry["runtime_seconds"],
        }
        for entry in stage3_log["entries"]
        if entry["kind"] == "association_rules"
    ]
    diagnosis["association_runtime_accounting"] = {
        "stage3_reported_runtime_excludes_shared_index": True,
        "shared_cooccurrence_build_seconds_measured_now": round(shared_seconds, 3),
        "shared_index_shape": {
            "training_users": int(shared_index["interactions"].shape[0]),
            "training_items": int(shared_index["interactions"].shape[1]),
            "cooccurrence_nonzero": int(shared_index["cooccurrence"].nnz),
        },
        "stage3_configuration_scoring_seconds": ar_entries,
        "note": (
            "Stage 3 per-configuration runtimes cover rule scoring/ranking only; "
            "the shared co-occurrence construction is a one-time cost per training "
            "interaction definition."
        ),
    }
    write_json(stage4_config.out("diagnosis", "category_cbf_ties.json"), diagnosis)
    print(json.dumps(diagnosis["stage3_category_cbf"]["exact_ties"], indent=2))
    print(
        "distinct lists",
        diagnosis["stage3_category_cbf"]["distinct_top5_lists"],
        "shared index seconds",
        round(shared_seconds, 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
