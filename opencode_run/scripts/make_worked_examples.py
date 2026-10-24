#!/usr/bin/env python3
"""Build reproducible worked examples for the Stage 2 coverage/evaluation audit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import format_epoch, load_config, parse_epoch  # noqa: E402
from stage2.evaluation import ap_at_k, map_at_k  # noqa: E402
from stage2.io_utils import read_json, write_csv, write_json  # noqa: E402


SPANNING_USER = "3398461054087191302"
SPANNING_VIEW = ("2020-03-11 07:22:08", "2020-03-11 08:45:52")
SPANNING_CHANNEL = "3"
SPANNING_AIRING_IDS = [45345, 45346, 45347]

FRAGMENTED_USER = "3398461054087191302"
FRAGMENTED_CHANNEL = "1020"
FRAGMENTED_AIRING_ID = 1070
FRAGMENTED_VIEWS = [
    ("2020-05-08 15:46:28", "2020-05-08 15:49:33"),
    ("2020-05-08 15:49:38", "2020-05-08 15:51:07"),
]


def _load_airings(path: Path) -> dict[int, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["airing_idx"]): row for row in csv.DictReader(handle)}


def _interaction_rows(npz_path: Path) -> dict[tuple[int, int], tuple[int, int, int]]:
    data = np.load(npz_path)
    return {
        (int(u), int(a)): (int(c), int(d), int(b))
        for u, a, c, d, b in zip(
            data["user_idx"],
            data["airing_idx"],
            data["covered_sec"],
            data["airing_duration"],
            data["bridged_covered_sec"],
        )
    }


def _user_index(config) -> dict[str, int]:
    return {
        uid: idx
        for idx, uid in enumerate(read_json(config.out("views", "user_ids.json"))["user_ids"])
    }


def _ap_details(predictions, relevant):
    relevant_set = {int(x) for x in relevant}
    seen = set()
    hits = []
    for rank, item in enumerate(predictions[:5], start=1):
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        if item in relevant_set:
            hits.append({"rank": rank, "show_id": item, "precision_at_rank": len(hits) / rank + 1 / rank})
    return {
        "predictions": [int(x) for x in predictions],
        "relevant": sorted(relevant_set),
        "relevant_count": len(relevant_set),
        "hit_positions": [h["rank"] for h in hits],
        "ap_at_5": ap_at_k(predictions, relevant_set, k=5),
    }


def main() -> int:
    config = load_config(ROOT / "config" / "stage2.json")
    user_index = _user_index(config)
    airings = _load_airings(config.out("airings", "airings_train.csv"))
    interactions = _interaction_rows(config.out("interactions", "train_user_airing.npz"))

    cut = config.windows["train"].end
    raw_views: dict[str, list[dict[str, str]]] = {"spanning": [], "fragmented": []}
    first_crossing = None
    with config.view_file.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            uid = row["user_id"]
            if uid == SPANNING_USER and row["vsetv_id"] == SPANNING_CHANNEL:
                if (row["start_time"], row["stop_time"]) == SPANNING_VIEW:
                    raw_views["spanning"].append(row)
            if uid == FRAGMENTED_USER and row["vsetv_id"] == FRAGMENTED_CHANNEL:
                if (row["start_time"], row["stop_time"]) in FRAGMENTED_VIEWS:
                    raw_views["fragmented"].append(row)
            if first_crossing is None:
                start = parse_epoch(row["start_time"])
                stop = parse_epoch(row["stop_time"])
                if start < cut < stop:
                    first_crossing = {
                        "user_id": uid,
                        "channel_id": row["vsetv_id"],
                        "raw_start": row["start_time"],
                        "raw_stop": row["stop_time"],
                        "cutoff": format_epoch(cut),
                    }

    spanning_idx = user_index[SPANNING_USER]
    spanning_rows = []
    total_covered = 0
    for airing_idx in SPANNING_AIRING_IDS:
        covered, duration, _ = interactions[(spanning_idx, airing_idx)]
        total_covered += covered
        airing = airings[airing_idx]
        spanning_rows.append(
            {
                "airing_idx": airing_idx,
                "channel_id": airing["channel_id"],
                "start_time": airing["start_time"],
                "duration_seconds": duration,
                "tv_show_id": int(airing["tv_show_id"]),
                "tv_show_title": airing["tv_show_title"],
                "covered_seconds": covered,
                "coverage_fraction": covered / duration,
            }
        )
    spanning_example = {
        "user_id": SPANNING_USER,
        "one_raw_view": {"start": SPANNING_VIEW[0], "stop": SPANNING_VIEW[1]},
        "raw_views_matching_window": raw_views["spanning"],
        "airings_covered_by_the_view": spanning_rows,
        "sum_covered_seconds": total_covered,
        "view_duration_seconds": parse_epoch(SPANNING_VIEW[1]) - parse_epoch(SPANNING_VIEW[0]),
        "explanation": (
            "A single 5,024-second view is matched to every overlapping airing on "
            "the same channel, not only to the largest-overlap airing."
        ),
    }

    fragmented_idx = user_index[FRAGMENTED_USER]
    actual, duration, bridged = interactions[(fragmented_idx, FRAGMENTED_AIRING_ID)]
    fragmented_airing = airings[FRAGMENTED_AIRING_ID]
    fragmented_example = {
        "user_id": FRAGMENTED_USER,
        "channel_id": FRAGMENTED_CHANNEL,
        "raw_views": raw_views["fragmented"],
        "airing": {
            "airing_idx": FRAGMENTED_AIRING_ID,
            "tv_show_id": int(fragmented_airing["tv_show_id"]),
            "tv_show_title": fragmented_airing["tv_show_title"],
            "start_time": fragmented_airing["start_time"],
            "duration_seconds": duration,
        },
        "primary_covered_seconds": actual,
        "bridged_sensitivity_covered_seconds": bridged,
        "bridged_gap_seconds": bridged - actual,
        "primary_coverage_fraction": actual / duration,
        "bridged_coverage_fraction": bridged / duration,
        "explanation": (
            "The two observations have a 5-second unobserved gap. Primary coverage "
            "does not count the gap; the bridged sensitivity merges it."
        ),
    }
    assert actual == 274 and bridged == 279
    assert total_covered == 5024

    boundary_example = None
    if first_crossing is not None:
        boundary_idx = user_index[first_crossing["user_id"]]
        start = parse_epoch(first_crossing["raw_start"])
        cached = None
        train = np.load(config.out("views", "views_train.npz"))
        for i in np.flatnonzero(train["user_idx"] == boundary_idx):
            if int(train["start"][i]) == start:
                cached = {
                    "cached_start": format_epoch(int(train["start"][i])),
                    "cached_stop": format_epoch(int(train["stop"][i])),
                    "cached_duration_seconds": int(train["stop"][i] - train["start"][i]),
                }
                break
        boundary_example = {
            **first_crossing,
            "raw_duration_seconds": parse_epoch(first_crossing["raw_stop"])
            - parse_epoch(first_crossing["raw_start"]),
            "cached_train_view": cached,
            "explanation": (
                "The view starts in train but is clipped at the train cutoff; the "
                "post-cutoff part is not written to the train cache."
            ),
        }
        assert cached is not None
        assert cached["cached_stop"] == first_crossing["cutoff"]

    label_payload = read_json(config.out("labels", "val_labels_frozen.json"))
    labels = label_payload["primary_top5"]
    candidates = set(read_json(config.out("candidates", "candidates_val.json"))["ids"])
    integrity = {
        "users": 0,
        "primary_top5_not_subset_of_all_primary": 0,
        "primary_not_subset_of_any_positive": 0,
        "label_ids_missing_from_val_candidates": 0,
        "top5_size_gt_5": 0,
    }
    for uid, top5 in labels.items():
        integrity["users"] += 1
        all_primary = set(label_payload["primary_all_qualifying"][uid])
        all_any = set(label_payload["sensitivity_any_positive_all"][uid])
        if not set(top5) <= all_primary:
            integrity["primary_top5_not_subset_of_all_primary"] += 1
        if not all_primary <= all_any:
            integrity["primary_not_subset_of_any_positive"] += 1
        if not all_any <= candidates:
            integrity["label_ids_missing_from_val_candidates"] += 1
        if len(top5) > 5:
            integrity["top5_size_gt_5"] += 1

    predictions = read_json(
        config.out("candidates", "popularity_fallback_val_top5.json")
    )["ids"]
    map_result = map_at_k(
        {uid: predictions for uid in labels},
        labels,
        users=labels.keys(),
        k=5,
    )
    aps = map_result.pop("aps")
    write_csv(
        config.out("worked", "ap5_popularity_fallback_per_user.csv"),
        ["user_id", "label_size", "ap_at_5"],
        (
            {"user_id": uid, "label_size": len(labels[uid]), "ap_at_5": ap}
            for uid, ap in sorted(aps.items())
        ),
    )

    example_ids = []
    for predicate in (
        lambda uid: len(labels[uid]) >= 5,
        lambda uid: 0 < len(labels[uid]) < 5,
        lambda uid: len(labels[uid]) == 0,
    ):
        example_ids.append(next(uid for uid in labels if predicate(uid)))
    ap_examples = [
        {
            "user_id": uid,
            "label_size": len(labels[uid]),
            **_ap_details(predictions, labels[uid]),
        }
        for uid in example_ids
    ]
    ap_payload = {
        "not_a_model_result": True,
        "purpose": (
            "Smoke-test the AP@5 implementation and the frozen-label format using "
            "the training-only popularity fallback, not a trained recommender."
        ),
        "prediction_rule": read_json(
            config.out("candidates", "popularity_fallback_val_top5.json")
        )["rule"],
        "predictions": predictions,
        "cohort_size": map_result["cohort_size"],
        "summary": map_result,
        "examples": ap_examples,
        "label_integrity": integrity,
    }
    write_json(config.out("worked", "ap5_popularity_fallback.json"), ap_payload)
    write_json(
        config.out("worked", "coverage_examples.json"),
        {
            "source": {
                "view_file": str(config.view_file),
                "airings": str(config.out("airings", "airings_train.csv")),
                "interactions": str(config.out("interactions", "train_user_airing.npz")),
            },
            "spanning_multiple_airings": spanning_example,
            "fragmented_coverage": fragmented_example,
            "cutoff_clipping": boundary_example,
        },
    )
    print(json.dumps({"coverage": str(config.out("worked", "coverage_examples.json")),
                      "ap5": str(config.out("worked", "ap5_popularity_fallback.json")),
                      "map5": map_result,
                      "label_integrity": integrity}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
