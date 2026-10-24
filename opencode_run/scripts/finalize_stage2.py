#!/usr/bin/env python3
"""Attach auxiliary Stage 2 artifacts to the summary.

The core pipeline writes ``audit/stage2_summary.json``.  The candidate-scope
audit, worked examples, source-verification record, and validation record are
produced by separate scripts, so they are linked into the summary here instead
of forcing a full preprocessing rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "config" / "stage2.json")
    summary_path = config.out("audit", "stage2_summary.json")
    summary = read_json(summary_path)

    scope = read_json(config.out("audit", "candidate_scope.json"))
    validation = read_json(config.out("audit", "stage2_validation.json"))
    ap5 = read_json(config.out("worked", "ap5_popularity_fallback.json"))

    summary["candidate_scope_audit"] = str(config.out("audit", "candidate_scope.json"))
    summary["candidate_scope_summary"] = {
        name: {
            "raw_distinct_labelled_ids": scope[name]["raw_distinct_labelled_ids"],
            "in_window_distinct_labelled_ids": scope[name]["in_window_distinct_labelled_ids"],
            "contained_in_window_distinct_ids": scope[name]["contained_in_window_distinct_ids"],
            "airing_table_distinct_ids": scope[name]["airing_table_distinct_ids"],
            "crossing_window_end_only_ids": scope[name]["crossing_window_end_only_ids"],
            "raw_ids_only_in_spillover": scope[name]["raw_ids_only_in_spillover"],
        }
        for name in ("train", "val", "test")
    }
    summary["test_descriptive"]["raw_file_distinct_labelled_ids"] = scope["test"][
        "raw_distinct_labelled_ids"
    ]
    summary["test_descriptive"]["excluded_only_in_spillover_ids"] = len(
        scope["test"]["raw_ids_only_in_spillover"]["before_window"]
    ) + len(scope["test"]["raw_ids_only_in_spillover"]["after_window"])

    summary["validation_audit"] = str(config.out("audit", "stage2_validation.json"))
    summary["validation_passed"] = validation["passed"]
    summary["validation_checks"] = f"{validation['passed_count']}/{validation['total_checks']}"
    summary["worked_coverage_examples"] = str(config.out("worked", "coverage_examples.json"))
    summary["worked_ap5_smoke_test"] = str(config.out("worked", "ap5_popularity_fallback.json"))
    summary["worked_ap5_per_user"] = str(
        config.out("worked", "ap5_popularity_fallback_per_user.csv")
    )
    summary["source_verification"] = str(config.out("audit", "source_verification.json"))
    summary["authoritative_watched_definition"] = (
        "not found; the local 80%-coverage/top-5 label is an approximation"
    )
    summary["ap5_smoke_test_summary"] = {
        "note": "Training-only popularity fallback, not a trained recommender.",
        "map_at_5": ap5["summary"]["map_at_k"],
        "not_a_model_result": True,
    }

    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "validation_passed": summary["validation_passed"],
                "validation_checks": summary["validation_checks"],
                "candidate_counts": summary["candidate_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
