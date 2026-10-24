#!/usr/bin/env python3
"""Validate the produced Stage 2 artifacts without rebuilding the pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.evaluation import ap_at_k, map_at_k  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "config" / "stage2.json")
    summary = read_json(config.out("audit", "stage2_summary.json"))
    split = read_json(config.out("audit", "view_split_audit.json"))
    raw_integrity = read_json(config.out("audit", "raw_integrity.json"))
    labels_manifest = read_json(config.out("labels", "val_labels_manifest.json"))
    labels = read_json(config.out("labels", "val_labels_frozen.json"))
    val_candidates = read_json(config.out("candidates", "candidates_val.json"))
    test_candidates = read_json(config.out("candidates", "candidates_test.json"))
    scope = read_json(config.out("audit", "candidate_scope.json"))
    worked = read_json(config.out("worked", "coverage_examples.json"))
    ap5 = read_json(config.out("worked", "ap5_popularity_fallback.json"))
    ap5_csv = config.out("worked", "ap5_popularity_fallback_per_user.csv")

    checks = []

    def check(name: str, condition: bool, detail=None):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check("raw_csv_checksums_unchanged", raw_integrity["unchanged"] is True)
    check(
        "all_view_rows_accounted_for",
        split["written"]["train"] + split["written"]["val"] == split["total_rows"],
        {
            "train": split["written"]["train"],
            "val": split["written"]["val"],
            "total": split["total_rows"],
        },
    )
    check(
        "view_totals_match_summary",
        summary["train_interaction"]["views_loaded"] + summary["val_interaction"]["views_loaded"]
        == split["total_rows"],
    )
    check(
        "candidate_counts_match_airing_tables",
        val_candidates["count"] == scope["val"]["airing_table_distinct_ids"]
        and test_candidates["count"] == scope["test"]["airing_table_distinct_ids"],
        {
            "val": val_candidates["count"],
            "test": test_candidates["count"],
        },
    )
    for name in ("train", "val", "test"):
        item = scope[name]["expected_vs_airing_table"]
        check(
            f"airing_table_scope_consistent_{name}",
            not item["missing_from_airing_table"] and not item["extra_in_airing_table"],
            item,
        )

    candidate_set = set(val_candidates["ids"])
    subset_all = subset_any = missing_candidate = top5_too_large = 0
    for uid, top5 in labels["primary_top5"].items():
        all_primary = set(labels["primary_all_qualifying"][uid])
        all_any = set(labels["sensitivity_any_positive_all"][uid])
        subset_all += not set(top5) <= all_primary
        subset_any += not all_primary <= all_any
        missing_candidate += not all_any <= candidate_set
        top5_too_large += len(top5) > config.top_k
    check(
        "label_hierarchy_valid",
        subset_all == subset_any == missing_candidate == top5_too_large == 0,
        {
            "primary_top5_not_subset_all": subset_all,
            "all_primary_not_subset_any": subset_any,
            "labels_missing_from_candidates": missing_candidate,
            "top5_too_large": top5_too_large,
        },
    )
    check(
        "cohort_counts_match_manifest",
        summary["cohort"] == labels_manifest["cohort"],
    )

    spanning = worked["spanning_multiple_airings"]
    check(
        "worked_spanning_example_sums_to_view_duration",
        spanning["sum_covered_seconds"] == spanning["view_duration_seconds"] == 5024,
    )
    fragmented = worked["fragmented_coverage"]
    check(
        "worked_fragmented_example_matches_interaction_audit",
        fragmented["primary_covered_seconds"] == 274
        and fragmented["bridged_sensitivity_covered_seconds"] == 279
        and fragmented["bridged_gap_seconds"] == 5,
    )
    boundary = worked["cutoff_clipping"]
    check(
        "worked_cutoff_example_is_clipped",
        boundary is not None
        and boundary["cached_train_view"]["cached_stop"] == config.windows["train"].end_iso
        and boundary["cached_train_view"]["cached_duration_seconds"] < boundary["raw_duration_seconds"],
    )

    recomputed = map_at_k(
        {uid: ap5["predictions"] for uid in labels["primary_top5"]},
        labels["primary_top5"],
        users=labels["primary_top5"].keys(),
        k=5,
    )
    recomputed.pop("aps")
    check(
        "ap5_smoke_test_recomputes",
        recomputed == ap5["summary"],
        {"summary": recomputed},
    )
    check(
        "ap5_examples_match_ap_function",
        all(
            abs(example["ap_at_5"] - ap_at_k(ap5["predictions"], example["relevant"], 5)) < 1e-12
            for example in ap5["examples"]
        ),
    )
    check("ap5_per_user_file_written", ap5_csv.exists() and ap5_csv.stat().st_size > 0)

    payload = {
        "stage": 2,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "passed_count": sum(1 for item in checks if item["passed"]),
        "total_checks": len(checks),
        "note": (
            "These checks validate artifact consistency and the AP@5 implementation. They do not "
            "establish an authoritative Kaggle watched-label definition."
        ),
    }
    write_json(config.out("audit", "stage2_validation.json"), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
