#!/usr/bin/env python3
"""Build the corrected Stage 3 protocol (v2) without training recommenders."""

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
    parser.add_argument("--config", default=str(ROOT / "config" / "stage3.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    for extra in ("methods", "checkpoints", "report"):
        config.out(extra).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw_paths = {
        "view_file": config.view_file,
        "submission_file": config.submission_file,
        **{f"schedule_{name}": path for name, path in config.schedule_files.items()},
    }
    before = raw_manifest(raw_paths)
    write_json(config.out("audit", "raw_manifest_before.json"), before)
    write_json(config.out("audit", "environment.json"), environment_manifest())

    print("Stage 3 protocol: schedules, interaction airings, and candidate airings", flush=True)
    t0 = time.perf_counter()
    schedules = build_all_schedules(config)
    schedule_runtime = time.perf_counter() - t0

    print("Stage 3 protocol: split views at every cutoff", flush=True)
    t0 = time.perf_counter()
    views = split_views(config)
    view_runtime = time.perf_counter() - t0

    print("Stage 3 protocol: training interactions", flush=True)
    t0 = time.perf_counter()
    train_audit = build_interactions(
        config,
        "train",
        schedules["airings"]["train"],
        views["windows"]["train"]["channel_ids"],
        "train",
    )
    train_runtime = time.perf_counter() - t0

    print("Stage 3 protocol: validation interactions and frozen labels", flush=True)
    t0 = time.perf_counter()
    val_audit = build_interactions(
        config,
        "val",
        schedules["airings"]["val"],
        views["windows"]["val"]["channel_ids"],
        "val",
    )
    labels = build_labels_and_candidates(
        config,
        schedules["airings"]["val"],
        schedules["candidate_airings"]["val"],
        schedules["candidate_airings"]["test"],
        schedules["airings"]["train"],
    )
    val_runtime = time.perf_counter() - t0

    val_interaction_ids = {int(row.show_id) for row in schedules["airings"]["val"].rows}
    val_candidate_ids = set(labels["val_candidates"])
    test_contained_ids = {
        int(row.show_id)
        for row in schedules["loaded"]["test"].owned_rows
        if row.show_id and row.show_id != "0" and row.duration > 0 and row.end <= config.windows["test"].end
    }
    test_candidate_ids = set(labels["test_candidates"])
    candidate_eligibility = {
        "protocol_version": config.raw["protocol_version"],
        "rule": config.raw["candidate_rule"],
        "val": {
            "candidate_count": len(val_candidate_ids),
            "interaction_label_id_count": len(val_interaction_ids),
            "candidate_ids_only_in_crossing_end_airings": sorted(
                val_candidate_ids - val_interaction_ids
            ),
            "interaction_ids_not_candidates": sorted(val_interaction_ids - val_candidate_ids),
        },
        "test": {
            "candidate_count": len(test_candidate_ids),
            "candidate_ids_only_in_crossing_end_airings": sorted(
                test_candidate_ids - test_contained_ids
            ),
            "note": "No local test labels; candidate availability is reported descriptively.",
        },
        "boundary_limitation": (
            "Validation candidates include airings starting in the window even if they "
            "end after it. Validation labels use only fully observed airings. Therefore "
            "crossing-end-only candidate IDs are recommendable but cannot become label "
            "positives under the conservative rule."
        ),
    }
    write_json(config.out("audit", "candidate_eligibility.json"), candidate_eligibility)

    after = raw_manifest(raw_paths)
    integrity_ok = before == after
    write_json(config.out("audit", "raw_manifest_after.json"), after)
    write_json(
        config.out("audit", "raw_integrity.json"),
        {"unchanged": integrity_ok, "before": before, "after": after},
    )
    if not integrity_ok:
        raise RuntimeError("Raw CSV files changed during Stage 3 protocol build")

    summary = {
        "stage": 3,
        "protocol_version": config.raw["protocol_version"],
        "scope": (
            "Corrected preprocessing protocol only. Stage 2 outputs remain the earlier "
            "protocol. No recommender was trained in this step."
        ),
        "config": str(config.config_path),
        "raw_integrity_unchanged": integrity_ok,
        "runtime_seconds": {
            "schedule": round(schedule_runtime, 3),
            "view_split": round(view_runtime, 3),
            "train_interactions": round(train_runtime, 3),
            "val_interactions_and_labels": round(val_runtime, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "view_split_audit": str(config.out("audit", "view_split_audit.json")),
        "schedule_audit": str(config.out("audit", "schedule_audit.json")),
        "interaction_audit_train": str(config.out("audit", "interaction_audit_train.json")),
        "interaction_audit_val": str(config.out("audit", "interaction_audit_val.json")),
        "labels_manifest": str(config.out("labels", "val_labels_manifest.json")),
        "candidate_eligibility": str(config.out("audit", "candidate_eligibility.json")),
        "candidates_val": labels["val_candidates"],
        "candidates_test_count": len(labels["test_candidates"]),
        "candidate_eligibility_summary": candidate_eligibility,
        "id_scopes": {
            "train_users": len(views["windows"]["train"]["user_ids"]),
            "train_channels": len(views["windows"]["train"]["channel_ids"]),
            "val_users": len(views["windows"]["val"]["user_ids"]),
            "val_channels": len(views["windows"]["val"]["channel_ids"]),
            "note": "Training and validation ID maps are independent; train axes use training segments only.",
        },
        "interaction_counts": {
            "train": {
                key: train_audit[key]
                for key in (
                    "views_loaded",
                    "users_with_views",
                    "user_airing_rows",
                    "user_program_rows",
                    "coverage_primary_qualifying_airings",
                    "coverage_sensitivity_qualifying_airings",
                )
            },
            "val": {
                key: val_audit[key]
                for key in (
                    "views_loaded",
                    "users_with_views",
                    "user_airing_rows",
                    "user_program_rows",
                    "coverage_primary_qualifying_airings",
                    "coverage_sensitivity_qualifying_airings",
                )
            },
        },
        "cohort": labels["label_manifest"]["cohort"],
        "statistics_scope": {
            "model_selection": "training window only",
            "final_refit": "not performed in Stage 3",
            "cold_start_note": (
                "Cohort users without a training positive are early-training cold-start "
                "users for model selection; final-refit load counts are not computed."
            ),
        },
    }
    write_json(config.out("audit", "stage3_protocol_summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
