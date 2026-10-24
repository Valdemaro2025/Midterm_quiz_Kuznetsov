#!/usr/bin/env python3
"""Run Stage 2 preprocessing and evaluation-protocol construction.

This script does not train recommendation models, publish anything, submit to
Kaggle, or write the final report.  It builds and freezes the data artifacts and
audits that later model experiments must reuse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.interactions import build_interactions  # noqa: E402
from stage2.io_utils import environment_manifest, raw_manifest, write_json  # noqa: E402
from stage2.labels import build_labels_and_candidates  # noqa: E402
from stage2.schedule import build_all_schedules  # noqa: E402
from stage2.views import split_views  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "stage2.json"),
        help="Path to the Stage 2 JSON config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    total_started = time.perf_counter()

    raw_paths = {
        "view_file": config.view_file,
        "submission_file": config.submission_file,
        **{f"schedule_{name}": path for name, path in config.schedule_files.items()},
    }
    before = raw_manifest(raw_paths)
    write_json(config.out("audit", "raw_manifest_before.json"), before)
    write_json(config.out("audit", "environment.json"), environment_manifest())

    print("Stage 2: schedule ownership and airing tables", flush=True)
    t0 = time.perf_counter()
    schedule_result = build_all_schedules(config)
    schedule_runtime = time.perf_counter() - t0

    print("Stage 2: chronological view split", flush=True)
    t0 = time.perf_counter()
    view_result = split_views(config)
    view_runtime = time.perf_counter() - t0

    print("Stage 2: training user-airing coverage", flush=True)
    t0 = time.perf_counter()
    train_audit = build_interactions(
        config,
        "train",
        schedule_result["airings"]["train"],
        view_result["channel_ids"],
        "train",
    )
    train_runtime = time.perf_counter() - t0

    print("Stage 2: validation user-airing coverage and labels", flush=True)
    t0 = time.perf_counter()
    val_audit = build_interactions(
        config,
        "val",
        schedule_result["airings"]["val"],
        view_result["channel_ids"],
        "val",
    )
    labels_result = build_labels_and_candidates(
        config,
        schedule_result["airings"]["val"],
        schedule_result["airings"]["test"],
        schedule_result["airings"]["train"],
    )
    val_runtime = time.perf_counter() - t0

    after = raw_manifest(raw_paths)
    integrity_ok = before == after
    write_json(
        config.out("audit", "raw_manifest_after.json"),
        after,
    )
    write_json(
        config.out("audit", "raw_integrity.json"),
        {
            "unchanged": integrity_ok,
            "before": before,
            "after": after,
        },
    )
    if not integrity_ok:
        raise RuntimeError("Raw CSV files changed during Stage 2")

    summary = {
        "stage": 2,
        "scope": (
            "Preprocessing and evaluation-protocol construction only. No "
            "recommendation models were trained, no Kaggle submission was made, "
            "and no final report was written."
        ),
        "config": str(Path(args.config).resolve()),
        "environment": str(config.out("audit", "environment.json")),
        "raw_integrity_unchanged": integrity_ok,
        "runtime_seconds": {
            "schedule": round(schedule_runtime, 3),
            "view_split": round(view_runtime, 3),
            "train_interactions": round(train_runtime, 3),
            "val_interactions_and_labels": round(val_runtime, 3),
            "total": round(time.perf_counter() - total_started, 3),
        },
        "schedule_audit": str(config.out("audit", "schedule_audit.json")),
        "view_split_audit": str(config.out("audit", "view_split_audit.json")),
        "interaction_audit_train": str(config.out("audit", "interaction_audit_train.json")),
        "interaction_audit_val": str(config.out("audit", "interaction_audit_val.json")),
        "labels_manifest": str(config.out("labels", "val_labels_manifest.json")),
        "candidates_val": str(config.out("candidates", "candidates_val.json")),
        "candidates_test": str(config.out("candidates", "candidates_test.json")),
        "train_interaction": {
            key: train_audit[key]
            for key in (
                "views_loaded",
                "users_with_views",
                "user_airing_rows",
                "user_program_rows",
                "positive_coverage_airings",
                "coverage_primary_qualifying_airings",
                "coverage_sensitivity_qualifying_airings",
                "views_on_channel_absent_from_schedule",
                "seconds_on_channel_absent_from_schedule",
                "views_on_channel_with_schedule_but_no_labelled_airings",
                "seconds_on_channel_with_schedule_but_no_labelled_airings",
                "merged_sessions_without_labelled_airing_overlap",
                "seconds_without_labelled_airing_overlap",
                "runtime_seconds",
            )
        },
        "val_interaction": {
            key: val_audit[key]
            for key in (
                "views_loaded",
                "users_with_views",
                "user_airing_rows",
                "user_program_rows",
                "positive_coverage_airings",
                "coverage_primary_qualifying_airings",
                "coverage_sensitivity_qualifying_airings",
                "views_on_channel_absent_from_schedule",
                "seconds_on_channel_absent_from_schedule",
                "views_on_channel_with_schedule_but_no_labelled_airings",
                "seconds_on_channel_with_schedule_but_no_labelled_airings",
                "merged_sessions_without_labelled_airing_overlap",
                "seconds_without_labelled_airing_overlap",
                "runtime_seconds",
            )
        },
        "cohort": labels_result["label_manifest"]["cohort"],
        "candidate_counts": {
            "val": len(labels_result["val_candidates"]),
            "test": len(labels_result["test_candidates"]),
        },
        "test_descriptive": labels_result["test_summary"],
        "observed_vs_assumption": {
            "observed": [
                "Raw CSV checksums unchanged.",
                "Schedule rows owned by the nominal half-open window containing their start time.",
                "View intervals split by start and clipped at the window end.",
                "User-airing coverage computed from unions of observed seconds.",
                "Only labelled schedule airings (non-zero tv_show_id) are matchable; views on unlabelled rows or outside labelled schedule coverage create no user-airing row.",
            ],
            "assumption": [
                "The local 80%-coverage/top-5 label is an approximation, not verified Kaggle behaviour.",
                "The submission template users are the primary evaluation cohort.",
                "The test window is [2020-07-27, 2020-10-19) and test schedule rows are candidate/display metadata only.",
            ],
        },
        "outputs_root": str(config.output_dir),
    }
    write_json(config.out("audit", "stage2_summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
