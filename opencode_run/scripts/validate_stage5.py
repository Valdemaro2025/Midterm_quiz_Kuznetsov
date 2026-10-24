#!/usr/bin/env python3
"""Independent Stage 5 audit: provenance, format, boundaries, and reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.io_utils import (  # noqa: E402
    environment_manifest,
    raw_manifest,
    read_json,
    sha256_file,
    write_json,
)
from stage4.data import load_stage4_data  # noqa: E402
from stage5.frozen import (  # noqa: E402
    AR_PARAMS,
    HYBRID_PARAMS,
    POPULARITY_PARAMS,
    REPEAT_PARAMS,
    SELECTED_STRATEGY,
)
from stage5.history import (  # noqa: E402
    build_popularity,
    read_count_variants,
    read_template_users,
)
from stage5.predict import fit_predict_hybrid  # noqa: E402
from stage5.submission import validate_submission  # noqa: E402


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "stage5.json"))
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    template_users = read_template_users(config)
    template_hash = sha256_file(config.submission_file)
    before = read_json(config.out("audit", "raw_manifest_before.json"))
    after = read_json(config.out("audit", "raw_manifest_after.json"))
    current_raw = raw_manifest(
        {
            "view_file": config.view_file,
            "submission_template": config.submission_file,
            **{f"schedule_{name}": path for name, path in config.schedule_files.items()},
        }
    )
    check("raw_manifests_before_after_equal", before == after)
    check("raw_manifest_still_current", current_raw == after)
    check("template_unchanged", template_hash == after["submission_template"]["sha256"])
    check("cohort_exact_1260_unique", len(template_users) == 1260 and len(set(template_users)) == 1260)

    schedule = read_json(config.out("audit", "history_schedule_audit.json"))
    stage3_schedule = read_json(stage3_config.out("audit", "schedule_audit.json"))
    expected_restored = stage3_schedule["airings"]["train"]["excluded"]["crossing_window_end"]
    expected_airings = (
        stage3_schedule["airings"]["train"]["selected_airings"]
        + stage3_schedule["airings"]["val"]["selected_airings"]
        + expected_restored
    )
    check(
        "training_window_exact",
        schedule["window"] == ["2020-03-09 00:00:00", "2020-07-27 00:00:00"],
        schedule["window"],
    )
    check(
        "may18_boundary_restored",
        schedule["restored_may18_crossing_airings"] == expected_restored > 0,
        {
            "restored": schedule["restored_may18_crossing_airings"],
            "stage3_train_excluded": expected_restored,
        },
    )
    check(
        "full_airings_reconstructed_not_concatenated",
        schedule["airings"]["selected_airings"] == expected_airings,
        {
            "full_selected": schedule["airings"]["selected_airings"],
            "expected": expected_airings,
        },
    )
    airing_rows = _read_csv_rows(config.out("airings", "airings_history.csv"))
    train_end = config.windows["train"].end
    airing_ok = all(
        int(row["start_epoch"]) < train_end
        and int(row["end_epoch"]) <= train_end
        and int(row["duration"]) > 0
        and int(row["tv_show_id"]) > 0
        for row in airing_rows
    )
    check(
        "history_airings_strictly_before_cutoff",
        airing_ok and len(airing_rows) == schedule["airings"]["selected_airings"],
        {"rows": len(airing_rows), "train_end": train_end},
    )
    check(
        "no_crossing_final_cutoff_airings",
        schedule["airings"]["excluded"]["crossing_window_end"]
        == schedule["excluded_full_history_crossing_end"],
        schedule["airings"]["excluded"]["crossing_window_end"],
    )

    view_audit = read_json(config.out("audit", "history_view_audit.json"))
    check(
        "no_future_views",
        view_audit["no_future_observations"] is True
        and view_audit["max_observed_stop"] <= train_end
        and view_audit["rows_after_window"] == 0,
        {
            "max_observed_stop": view_audit["max_observed_stop"],
            "max_allowed": train_end,
            "rows_after": view_audit["rows_after_window"],
        },
    )
    check(
        "full_history_views_loaded",
        view_audit["segments_written"] == view_audit["total_rows"]
        and view_audit["segments_written"] > 0,
        view_audit["segments_written"],
    )

    candidates = read_json(config.out("candidates", "candidates_test.json"))
    stage3_candidates = read_json(stage3_config.out("candidates", "candidates_test.json"))
    candidate_ids = [int(x) for x in candidates["candidate_ids"]]
    check("test_candidate_count_6632", len(candidate_ids) == 6632 == candidates["candidate_count"])
    check(
        "test_candidates_match_frozen_stage3_pool",
        candidate_ids == [int(x) for x in stage3_candidates["ids"]],
    )
    candidate_airings = _read_csv_rows(config.out("airings", "candidate_airings_test.csv"))
    test_start = config.windows["test"].start
    test_end = config.windows["test"].end
    candidate_rule_ok = all(
        test_start <= int(row["start_epoch"]) < test_end
        and int(row["duration"]) > 0
        and int(row["tv_show_id"]) > 0
        for row in candidate_airings
    )
    check(
        "candidate_rule_start_in_test_window",
        candidate_rule_ok,
        {"rows": len(candidate_airings), "window": [test_start, test_end]},
    )

    counts = read_count_variants(
        config.out("interactions", "history_user_program_counts.csv")
    )
    popularity_ids, popularity_stats, popularity_rows = build_popularity(counts, candidate_ids)
    stored_popularity = [
        {
            "tv_show_id": int(row["tv_show_id"]),
            "distinct_training_users_primary": int(
                row["distinct_training_users_primary"]
            ),
            "qualifying_training_airings_primary": int(
                row["qualifying_training_airings_primary"]
            ),
        }
        for row in _read_csv_rows(config.out("candidates", "popularity_test.csv"))
    ]
    check(
        "popularity_refit_full_history",
        stored_popularity == popularity_rows
        and read_json(config.out("candidates", "popularity_fallback_test_top5.json"))[
            "ids"
        ]
        == popularity_ids[:5],
        {"top5": popularity_ids[:5], "candidate_rows": len(popularity_rows)},
    )

    cold = read_json(config.out("audit", "cold_start.json"))
    check(
        "cold_start_scopes_distinguished",
        cold["earlier_validation_training"]["cohort_size"] == 1260
        and cold["final_full_history"]["cohort_size"] == 1260
        and cold["earlier_validation_training"]["matched_to_full_history_views"] == 1256
        and cold["final_full_history"]["matched_to_full_history_views"] == 1260
        and cold["earlier_validation_training"]["distinct_primary80"]["zero"] == 18
        and cold["final_full_history"]["distinct_primary80"]["zero"] == 4,
        {
            "earlier_primary80_zero": cold["earlier_validation_training"]["distinct_primary80"]["zero"],
            "full_primary80_zero": cold["final_full_history"]["distinct_primary80"]["zero"],
            "earlier_matched": cold["earlier_validation_training"]["matched_to_full_history_views"],
            "full_matched": cold["final_full_history"]["matched_to_full_history_views"],
        },
    )

    provenance = read_json(config.out("provenance", "model_provenance.json"))
    final_config = read_json(config.out("final_config.json"))
    check(
        "frozen_parameters_match",
        final_config["frozen_parameters"]["repeat"] == REPEAT_PARAMS
        and final_config["frozen_parameters"]["association"] == AR_PARAMS
        and final_config["frozen_parameters"]["hybrid"] == HYBRID_PARAMS
        and final_config["frozen_parameters"]["popularity"] == POPULARITY_PARAMS,
    )
    check(
        "strategy_frozen",
        provenance["strategy"] == final_config["selected_strategy"] == SELECTED_STRATEGY
        and provenance["alternative"] == final_config["production_alternative"] == "repeat_50",
    )
    check(
        "full_history_refit_provenance",
        provenance["fitted_on"]
        == "full historical period, not Stage 3 weeks 11-20 models"
        and provenance["counts"]["restored_may18_crossing_airings"] == expected_restored
        and provenance["counts"]["test_candidates"] == 6632,
        provenance["counts"],
    )
    check(
        "validation_score_labeled_earlier",
        "0.378546" in final_config["notes"]["selection_score"]
        and "unknown" in final_config["notes"]["kaggle_score"].lower(),
        final_config["notes"],
    )

    submission_path = config.out("kuznetsov_vladimir_midterm_submission.csv")
    submission_sha = sha256_file(submission_path)
    format_audit = validate_submission(submission_path, template_users, candidate_ids)
    check("submission_format", format_audit["all_passed"], format_audit)
    check(
        "submission_sha_matches_provenance",
        submission_sha == provenance["submission"]["sha256"],
        {"actual": submission_sha, "stored": provenance["submission"]["sha256"]},
    )
    check(
        "submission_is_not_template",
        submission_path.resolve() != config.submission_file.resolve(),
    )

    runtime = read_json(config.out("audit", "runtime.json"))
    check(
        "actual_runtime_recorded",
        runtime["model_fit_and_predict_seconds"] > 0
        and runtime["pipeline_total_seconds"] > 0,
        runtime,
    )

    stage4_data = load_stage4_data(stage3_config)
    reproduced = fit_predict_hybrid(
        stage4_data["cohort"],
        stage4_data["count_variants"],
        stage4_data["user_to_train_idx"],
        stage4_data["val_candidates"],
        stage4_data["popularity_val"],
        stage4_data["popularity"],
    )["predictions"]
    stored_validation = read_json(
        stage4_config.out("methods", "hybrid_70_30", "top5.json")
    )
    reproduction_ok = reproduced == stored_validation
    reproduction = {
        "ok": reproduction_ok,
        "training_window": ["2020-03-09 00:00:00", "2020-05-18 00:00:00"],
        "frozen_strategy": HYBRID_PARAMS,
        "stage4_predictions": str(stage4_config.out("methods", "hybrid_70_30", "top5.json")),
        "prediction_hash": _canonical_hash(reproduced),
        "stored_prediction_hash": _canonical_hash(stored_validation),
        "users_compared": len(stage4_data["cohort"]),
    }
    write_json(config.out("audit", "validation_reproduction.json"), reproduction)
    check(
        "final_code_reproduces_stage4_hybrid_validation",
        reproduction_ok,
        {
            "users": len(stage4_data["cohort"]),
            "hash": reproduction["prediction_hash"],
        },
    )

    summary = {
        "all_passed": all(item["ok"] for item in checks),
        "checks_passed": sum(1 for item in checks if item["ok"]),
        "checks_total": len(checks),
        "checks": checks,
        "submission_path": str(submission_path),
        "submission_sha256": submission_sha,
        "submission_rows": len(template_users),
        "test_candidates": len(candidate_ids),
        "strategy": SELECTED_STRATEGY,
        "earlier_local_validation_map_at_5": 0.378546,
        "kaggle_score": None,
        "environment": environment_manifest(),
    }
    write_json(config.out("audit", "stage5_validation.json"), summary)
    print(
        f"all_passed={summary['all_passed']} checks={summary['checks_passed']}/{summary['checks_total']} "
        f"rows={summary['submission_rows']} candidates={summary['test_candidates']} "
        f"sha256={submission_sha}"
    )
    for item in checks:
        if not item["ok"]:
            print("FAIL", item)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
