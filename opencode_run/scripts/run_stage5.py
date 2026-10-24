#!/usr/bin/env python3
"""Freeze Stage 4 hybrid_70_30, refit on weeks 11-30, and write the submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    write_csv,
    write_json,
)
from stage4.ranking import top_k  # noqa: E402
from stage5.frozen import (  # noqa: E402
    AR_PARAMS,
    HYBRID_PARAMS,
    POPULARITY_PARAMS,
    PRODUCTION_ALTERNATIVE,
    REPEAT_PARAMS,
    SELECTED_STRATEGY,
)
from stage5.history import (  # noqa: E402
    build_history_interactions,
    build_history_schedule,
    build_history_views,
    build_popularity,
    build_test_candidates,
    cold_start_stats,
    read_count_variants,
    read_template_users,
)
from stage5.predict import (  # noqa: E402
    fit_predict_hybrid,
    popularity_alternative,
)
from stage5.submission import validate_submission, write_submission  # noqa: E402


def _validate_predictions(
    predictions: dict[str, list[int]],
    cohort: list[str],
    candidates: list[int],
    k: int = 5,
) -> None:
    allowed = set(candidates)
    for uid in cohort:
        items = predictions.get(uid, [])
        if len(items) != k or len(set(items)) != k or not set(items) <= allowed:
            raise RuntimeError(f"Invalid predictions for {uid}: {items}")


def _top5_from_scores(
    scores: dict[str, dict[int, float]],
    cohort: list[str],
    candidates: list[int],
    popularity: list[int],
    stats: dict[int, tuple[int, int]],
) -> dict[str, list[int]]:
    predictions: dict[str, list[int]] = {}
    for uid in cohort:
        top, _ = top_k(scores[uid], candidates, popularity, stats, k=5)
        predictions[uid] = top
    return predictions


def _save_method(
    config: Any,
    name: str,
    kind: str,
    params: dict[str, Any],
    predictions: dict[str, list[int]],
    details: dict[str, Any],
) -> None:
    out = config.out("methods", name)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", params)
    write_json(out / "top5_test.json", predictions)
    write_json(out / "details.json", details)


def _write_reproduction_instructions(config: Any) -> None:
    path = config.out("REPRODUCTION.md")
    path.write_text(
        "\n".join(
            [
                "# Stage 5 reproduction",
                "",
                "From the project root, run:",
                "",
                "```sh",
                "python3 scripts/run_stage5.py",
                "python3 scripts/validate_stage5.py",
                "python3 -m pytest -q",
                "```",
                "",
                "Inputs:",
                "- `sweettv-tv-program-recommender/dataset11-30.csv`",
                "- `sweettv-tv-program-recommender/export_arh_11-20-final.csv`",
                "- `sweettv-tv-program-recommender/export_arh_21-30-final.csv`",
                "- `sweettv-tv-program-recommender/export_arh_31-42-final.csv`",
                "- `sweettv-tv-program-recommender/submission.csv` (read-only template)",
                "",
                "Output:",
                f"- `{config.out('kuznetsov_vladimir_midterm_submission.csv')}`",
                "",
                "The pipeline reads no test-viewing behavior and does not modify the template or earlier outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "stage5.json"))
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    stage3_config = load_config(args.stage3_config)
    config.ensure_dirs()
    for extra in ("methods", "checkpoints", "provenance"):
        config.out(extra).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    raw_paths = {
        "view_file": config.view_file,
        "submission_template": config.submission_file,
        **{f"schedule_{name}": path for name, path in config.schedule_files.items()},
    }
    before = raw_manifest(raw_paths)
    write_json(config.out("audit", "raw_manifest_before.json"), before)
    write_json(config.out("audit", "environment.json"), environment_manifest())

    cohort = read_template_users(config)
    if len(cohort) != 1260 or len(set(cohort)) != 1260:
        raise RuntimeError(f"Unexpected template cohort: {len(cohort)} rows")
    write_json(
        config.out("audit", "template.json"),
        {
            "path": str(config.submission_file),
            "sha256": sha256_file(config.submission_file),
            "users": len(cohort),
            "unique_users": len(set(cohort)),
            "first_users": cohort[:5],
        },
    )

    print("Stage 5: reconstruct historical schedule", flush=True)
    history = build_history_schedule(config)
    print("Stage 5: reconstruct test candidates", flush=True)
    test = build_test_candidates(config)
    expected_candidates = read_json(stage3_config.out("candidates", "candidates_test.json"))
    if test["candidates"] != [int(x) for x in expected_candidates["ids"]]:
        raise RuntimeError("Stage 5 test candidates differ from frozen Stage 3 candidate pool")
    if len(test["candidates"]) != 6632:
        raise RuntimeError(f"Unexpected test candidate count: {len(test['candidates'])}")
    print("Stage 5: split raw views into the full history window", flush=True)
    views = build_history_views(config)
    print("Stage 5: build full-history interactions", flush=True)
    interaction_audit = build_history_interactions(config, history["airings"], views)
    counts = read_count_variants(
        config.out("interactions", "history_user_program_counts.csv")
    )
    id_maps = read_json(config.out("views", "id_maps_history.json"))["history"]
    history_user_ids = [str(uid) for uid in id_maps["user_ids"]]
    user_to_train_idx = {uid: idx for idx, uid in enumerate(history_user_ids)}

    popularity_ids, popularity_stats, popularity_rows = build_popularity(
        counts, test["candidates"]
    )
    write_csv(
        config.out("candidates", "popularity_test.csv"),
        [
            "tv_show_id",
            "distinct_training_users_primary",
            "qualifying_training_airings_primary",
        ],
        popularity_rows,
    )
    write_json(
        config.out("candidates", "popularity_fallback_test_top5.json"),
        {
            "rule": POPULARITY_PARAMS["ranking"],
            "scope": "full historical period",
            "ids": popularity_ids[:5],
            "source_counts": str(config.out("interactions", "history_user_program_counts.csv")),
        },
    )

    full_cold = cold_start_stats(cohort, user_to_train_idx, counts)
    stage3_counts = read_count_variants(
        stage3_config.out("interactions", "train_user_program_counts.csv")
    )
    stage3_ids = [
        str(uid)
        for uid in read_json(stage3_config.out("views", "id_maps.json"))["train"]["user_ids"]
    ]
    stage3_map = {uid: idx for idx, uid in enumerate(stage3_ids)}
    earlier_cold = cold_start_stats(cohort, stage3_map, stage3_counts)
    cold_payload = {
        "definitions": {
            "earlier_validation_training": "weeks 11-20 training interactions used for Stage 3/4 model selection",
            "final_full_history": "weeks 11-30 interactions strictly before 2020-07-27, used only for the Stage 5 final refit",
            "cold_start_zero": "no qualifying programme under the named coverage threshold in that training period",
        },
        "earlier_validation_training": {
            "source": str(stage3_config.out("interactions", "train_user_program_counts.csv")),
            **earlier_cold,
        },
        "final_full_history": {
            "source": str(config.out("interactions", "history_user_program_counts.csv")),
            **full_cold,
        },
    }
    write_json(config.out("audit", "cold_start.json"), cold_payload)

    print("Stage 5: fit frozen hybrid and alternatives on full history", flush=True)
    model_start = time.perf_counter()
    hybrid = fit_predict_hybrid(
        cohort,
        counts,
        user_to_train_idx,
        test["candidates"],
        popularity_ids,
        popularity_stats,
    )
    model_runtime = time.perf_counter() - model_start

    hybrid_predictions = hybrid["predictions"]
    repeat_scores = hybrid["repeat_scores"]
    ar_scores = hybrid["association_scores"]
    index = hybrid["association_index"]
    repeat_predictions = _top5_from_scores(
        repeat_scores, cohort, test["candidates"], popularity_ids, popularity_stats
    )
    ar_predictions = _top5_from_scores(
        ar_scores, cohort, test["candidates"], popularity_ids, popularity_stats
    )
    popularity_predictions, popularity_detail = popularity_alternative(
        cohort, test["candidates"], popularity_ids
    )
    for name, predictions in (
        ("hybrid_70_30", hybrid_predictions),
        ("repeat_50", repeat_predictions),
        ("ar_c5_conf_max_80", ar_predictions),
        ("popularity", popularity_predictions),
    ):
        _validate_predictions(predictions, cohort, test["candidates"])

    _save_method(
        config,
        "hybrid_70_30",
        "hybrid",
        HYBRID_PARAMS,
        hybrid_predictions,
        {
            **hybrid["details"],
            "selected_strategy": SELECTED_STRATEGY,
            "training_window": [config.windows["train"].start_iso, config.windows["train"].end_iso],
            "test_candidate_count": len(test["candidates"]),
        },
    )
    _save_method(
        config,
        "repeat_50",
        "repeat",
        REPEAT_PARAMS,
        repeat_predictions,
        {
            **hybrid["details"]["repeat"],
            "production_alternative": PRODUCTION_ALTERNATIVE,
            "count_variant": "sens50",
        },
    )
    _save_method(
        config,
        "ar_c5_conf_max_80",
        "association_rules",
        AR_PARAMS,
        ar_predictions,
        {
            **hybrid["details"]["association"],
            "association_index_users": int(index["n_users"]),
            "association_index_items": len(index["item_ids"]),
            "association_cooccurrence_nnz": int(index["cooccurrence"].nnz),
        },
    )
    _save_method(
        config,
        "popularity",
        "baseline",
        POPULARITY_PARAMS,
        popularity_predictions,
        {
            **popularity_detail,
            "top5_global": popularity_ids[:5],
        },
    )

    submission_path = config.out("kuznetsov_vladimir_midterm_submission.csv")
    write_submission(submission_path, cohort, hybrid_predictions)
    precheck = validate_submission(submission_path, cohort, test["candidates"])
    write_json(config.out("audit", "submission_precheck.json"), precheck)
    if not precheck["all_passed"]:
        raise RuntimeError(f"Submission precheck failed: {precheck}")

    after = raw_manifest(raw_paths)
    write_json(config.out("audit", "raw_manifest_after.json"), after)
    integrity_ok = before == after
    write_json(
        config.out("audit", "raw_integrity.json"),
        {"unchanged": integrity_ok, "before": before, "after": after},
    )
    if not integrity_ok:
        raise RuntimeError("Raw input files changed during Stage 5")

    submission_sha = sha256_file(submission_path)
    _write_reproduction_instructions(config)
    final_config = {
        "stage": 5,
        "protocol_version": config.raw["protocol_version"],
        "selected_strategy": SELECTED_STRATEGY,
        "production_alternative": PRODUCTION_ALTERNATIVE,
        "training_window": [
            config.windows["train"].start_iso,
            config.windows["train"].end_iso,
        ],
        "strictly_before": config.windows["train"].end_iso,
        "test_candidate_window": [
            config.windows["test"].start_iso,
            config.windows["test"].end_iso,
        ],
        "frozen_parameters": {
            "repeat": REPEAT_PARAMS,
            "association": AR_PARAMS,
            "hybrid": HYBRID_PARAMS,
            "popularity": POPULARITY_PARAMS,
        },
        "notes": {
            "selection_score": "earlier local validation MAP@5 = 0.378546 for hybrid_70_30",
            "kaggle_score": "unknown; no Kaggle score has been returned",
            "statistical_claim": "the small hybrid improvement is not statistically established or independently confirmed",
        },
    }
    write_json(config.out("final_config.json"), final_config)
    provenance = {
        "stage": 5,
        "strategy": SELECTED_STRATEGY,
        "alternative": PRODUCTION_ALTERNATIVE,
        "training_window": final_config["training_window"],
        "fitted_on": "full historical period, not Stage 3 weeks 11-20 models",
        "data_sources": before,
        "raw_integrity_unchanged": integrity_ok,
        "template_sha256": sha256_file(config.submission_file),
        "counts": {
            "cohort_users": len(cohort),
            "history_view_users": len(history_user_ids),
            "history_view_channels": len(views["channel_ids"]),
            "history_views_loaded": int(interaction_audit["views_loaded"]),
            "history_user_airing_rows": int(interaction_audit["user_airing_rows"]),
            "history_user_program_rows": int(interaction_audit["user_program_rows"]),
            "history_airings": int(history["airings"].audit["selected_airings"]),
            "restored_may18_crossing_airings": int(
                history["restored_may18_crossing_airings"]
            ),
            "test_candidates": len(test["candidates"]),
            "test_candidate_airings": int(test["airings"].audit["selected_airings"]),
        },
        "models": {
            "repeat_50": {
                "count_variant": "qualifying_airings_sensitivity",
                "coverage_threshold": ">=50%",
                "details": hybrid["details"]["repeat"],
            },
            "ar_c5_conf_max_80": {
                "parameters": AR_PARAMS,
                "details": {
                    **hybrid["details"]["association"],
                    "association_index_users": int(index["n_users"]),
                    "association_index_items": len(index["item_ids"]),
                    "association_cooccurrence_nnz": int(index["cooccurrence"].nnz),
                },
            },
            "popularity_fallback": {
                "parameters": POPULARITY_PARAMS,
                "top5_global": popularity_ids[:5],
                "distinct_training_users": popularity_rows[0][
                    "distinct_training_users_primary"
                ]
                if popularity_rows
                else 0,
            },
            "hybrid_70_30": {
                "parameters": HYBRID_PARAMS,
                "details": hybrid["details"]["fusion"],
            },
        },
        "submission": {
            "path": str(submission_path),
            "sha256": submission_sha,
            "rows": len(cohort),
            "header": "user_id,tv_show_id",
        },
        "runtime_seconds": {
            "model_fit_and_predict": round(model_runtime, 3),
            "pipeline_total": round(time.perf_counter() - started, 3),
        },
        "reproduction": {
            "command": "python3 scripts/run_stage5.py && python3 scripts/validate_stage5.py",
            "instructions": str(config.out("REPRODUCTION.md")),
        },
        "limitations": [
            "The earlier local validation MAP@5 is 0.378546; no Kaggle score is known.",
            "The hybrid gain over repeat_50 is small and not statistically established or independently confirmed.",
            "Test labels are unavailable locally; only format, candidate, and provenance checks are possible.",
        ],
    }
    write_json(config.out("provenance", "model_provenance.json"), provenance)
    write_json(
        config.out("audit", "runtime.json"),
        {
            "model_fit_and_predict_seconds": round(model_runtime, 3),
            "pipeline_total_seconds": round(time.perf_counter() - started, 3),
            "timing_scope": "single final refit and test prediction run on this machine",
        },
    )
    write_json(
        config.out("checkpoints", "stage5_complete.json"),
        {
            "status": "complete",
            "submission_path": str(submission_path),
            "submission_sha256": submission_sha,
            "test_candidates": len(test["candidates"]),
            "cohort": len(cohort),
            "strategy": SELECTED_STRATEGY,
        },
    )
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "sha256": submission_sha,
                "cohort": len(cohort),
                "test_candidates": len(test["candidates"]),
                "model_runtime_seconds": round(model_runtime, 3),
                "pipeline_runtime_seconds": round(time.perf_counter() - started, 3),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
