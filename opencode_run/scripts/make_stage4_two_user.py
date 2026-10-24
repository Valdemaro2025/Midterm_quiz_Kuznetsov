#!/usr/bin/env python3
"""Two-user analysis for Stage 4 using a declared training-history rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402
from stage4.data import load_stage4_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)
    data = load_stage4_data(stage3_config)
    cohort = list(data["cohort"])
    history = data["history"]
    labels = data["labels"]["primary_top5"]
    all_labels = data["labels"]["primary_all_qualifying"]

    global_coverage: dict[int, dict[str, int]] = {}
    for variant in ("positive", "sens50", "primary80"):
        for items in data["count_variants"][variant].values():
            for show_id, count in items.items():
                entry = global_coverage.setdefault(
                    show_id,
                    {
                        "training_users_any_positive": 0,
                        "training_users_at_least_50": 0,
                        "training_users_at_least_80": 0,
                        "positive_airings": 0,
                        "sens50_airings": 0,
                        "primary80_airings": 0,
                    },
                )
                if variant == "positive":
                    entry["training_users_any_positive"] += 1
                    entry["positive_airings"] += int(count)
                elif variant == "sens50":
                    entry["training_users_at_least_50"] += 1
                    entry["sens50_airings"] += int(count)
                else:
                    entry["training_users_at_least_80"] += 1
                    entry["primary80_airings"] += int(count)

    # Declared before looking at recommendation outcomes.
    heavy = max(
        cohort,
        key=lambda uid: (
            history[uid]["distinct_primary80"],
            history[uid]["total_primary80"],
            -int(uid),
        ),
    )
    light = min(
        (uid for uid in cohort if history[uid]["distinct_primary80"] > 0),
        key=lambda uid: (
            history[uid]["distinct_primary80"],
            history[uid]["total_primary80"],
            int(uid),
        ),
    )

    method_specs = [
        ("popularity", "baseline", stage3_config.out("methods", "popularity", "top5.json")),
        ("repeat_50", "repeat >=50%", stage4_config.out("methods", "repeat_50", "top5.json")),
        ("cbf_genre_cast_log_pop", "CBF", stage4_config.out("methods", "cbf_genre_cast_log_pop", "top5.json")),
        ("ar_c5_conf_max_80", "association rules >=80%", stage4_config.out("methods", "ar_c5_conf_max_80", "top5.json")),
        ("hybrid_70_30", "hybrid 70/30", stage4_config.out("methods", "hybrid_70_30", "top5.json")),
    ]
    failure_spec = (
        "cbf_category_log_pop",
        "category-only CBF",
        stage4_config.out("methods", "cbf_category_log_pop", "top5.json"),
    )

    def row_for(user: str, show_id: int, rank: int) -> dict[str, object]:
        train_idx = history[user]["train_idx"]
        title = data["titles"].get(show_id, {})
        counts = {
            name: data["count_variants"][name].get(train_idx, {}).get(show_id, 0)
            for name in ("positive", "sens50", "primary80")
        }
        primary_hit_rank = None
        if show_id in set(labels.get(user, [])):
            primary_hit_rank = list(labels[user]).index(show_id) + 1
        seen_any = counts["positive"] > 0
        global_counts = global_coverage.get(
            show_id,
            {
                "training_users_any_positive": 0,
                "training_users_at_least_50": 0,
                "training_users_at_least_80": 0,
                "positive_airings": 0,
                "sens50_airings": 0,
                "primary80_airings": 0,
            },
        )
        return {
            "rank": rank,
            "tv_show_id": show_id,
            "title": title.get("canonical_title", "<no training metadata>"),
            "category": title.get("category", "<no training metadata>"),
            "in_training_history": seen_any,
            "personal_seen_any_positive_coverage": seen_any,
            "personal_seen_at_least_50_percent_coverage": counts["sens50"] > 0,
            "personal_seen_at_least_80_percent_coverage": counts["primary80"] > 0,
            "training_counts": counts,
            "global_item_coverage": global_counts,
            "validation_primary_hit": primary_hit_rank is not None,
            "validation_primary_label_rank": primary_hit_rank,
            "validation_all_qualifying_hit": show_id in set(all_labels.get(user, [])),
        }

    methods = []
    for name, kind, path in method_specs:
        top5 = read_json(path)
        aps = read_json(config_for_aps(stage3_config, stage4_config, name))
        methods.append(
            {
                "name": name,
                "kind": kind,
                "top5_by_user": {
                    user: [row_for(user, show_id, rank) for rank, show_id in enumerate(top5[user], start=1)]
                    for user in (heavy, light)
                },
                "ap5_by_user": {user: aps[user] for user in (heavy, light)},
            }
        )

    failure_top5 = read_json(failure_spec[2])
    failure = {
        "method": failure_spec[0],
        "kind": failure_spec[1],
        "user": heavy,
        "top5": [
            row_for(heavy, show_id, rank)
            for rank, show_id in enumerate(failure_top5[heavy], start=1)
        ],
        "validation_hits": [
            show_id for show_id in failure_top5[heavy] if show_id in set(labels.get(heavy, []))
        ],
        "training_history_flag_definition": (
            "in_training_history and personal_seen_* flags are keyed by the selected "
            "user's train_idx and describe that user's personal training history. "
            "in_training_history means that user's positive_airings > 0 for the item. "
            "global_item_coverage is separately aggregated across all training users."
        ),
        "cause": (
            "With category-only features, every candidate in a category has an "
            "identical score after IDF weighting. All 1,260 users have at least "
            "19 candidates tied at their maximum score (median 3,211), so the "
            "Top-5 is determined entirely by the tie-break. This user's maximum "
            "category is Фильмы, and popularity-then-ID returns five films, none "
            "of which is a validation hit."
        ),
    }

    worked_user = heavy
    worked_method = "repeat_50"
    worked_top5 = read_json(stage4_config.out("methods", "repeat_50", "top5.json"))[worked_user]
    hit_positions = [
        rank for rank, show_id in enumerate(worked_top5, start=1)
        if show_id in set(labels.get(worked_user, []))
    ]
    worked = {
        "user_id": worked_user,
        "method": worked_method,
        "labels": labels.get(worked_user, []),
        "predictions": worked_top5,
        "relevant_count_m": len(set(labels.get(worked_user, []))),
        "hit_rank_positions": hit_positions,
        "calculation": (
            f"AP@5 = (1/{hit_positions[0]}) / min(5, {len(set(labels.get(worked_user, [])))}) "
            f"= {1.0 / hit_positions[0] / min(5, len(set(labels.get(worked_user, [])))):.12f}"
        ),
        "ap_at_5": 1.0 / hit_positions[0] / min(5, len(set(labels.get(worked_user, [])))),
        "note": "Arithmetic shown for independent checking; not claimed to be user-verified.",
    }

    payload = {
        "selection_rule": {
            "heavy": "maximum distinct training programs with >=80% qualifying airings; ties by total >=80% qualifying airings then ascending user_id",
            "light": "minimum distinct training programs with >=80% qualifying airings among users with at least one such program; ties by total >=80% qualifying airings then ascending user_id",
            "outcome_blind": True,
            "eligible_cohort_size": len(cohort),
        },
        "heavy_user": {"user_id": heavy, **history[heavy]},
        "light_user": {"user_id": light, **history[light]},
        "methods": methods,
        "failure_case": failure,
        "worked_ap_at_5": worked,
        "training_history_flag_definition": (
            "in_training_history and personal_seen_* fields use counts keyed by the "
            "selected user's train_idx; they describe that user's personal training "
            "history. in_training_history means the user has positive_airings > 0 for "
            "the item. global_item_coverage is a separate item-level calculation across "
            "all training users."
        ),
        "global_item_coverage_definition": (
            "global_item_coverage counts distinct training users and their airings for "
            "the item across all training users, split by any-positive, >=50%, and "
            ">=80% coverage. It does not describe the selected user's history."
        ),
        "limitations": [
            "The light user has no primary validation positives, so AP@5 is 0 by definition for every method; this is not evidence of which method is better for light users.",
            "Titles/categories shown are from training-period metadata only.",
            "in_training_history is personal to the selected user; global_item_coverage is reported separately.",
        ],
    }
    write_json(stage4_config.out("two_user", "analysis.json"), payload)
    write_json(stage4_config.out("two_user", "worked_ap5.json"), worked)
    write_json(stage4_config.out("two_user", "failure_case.json"), failure)
    print(json.dumps({
        "heavy": payload["heavy_user"],
        "light": payload["light_user"],
        "worked_ap": worked,
        "failure_hits": failure["validation_hits"],
    }, indent=2, ensure_ascii=False))
    return 0


def config_for_aps(stage3_config, stage4_config, name):
    if name == "popularity":
        return stage3_config.out("methods", "popularity", "per_user_ap5.json")
    return stage4_config.out("methods", name, "per_user_ap5.json")


if __name__ == "__main__":
    raise SystemExit(main())
