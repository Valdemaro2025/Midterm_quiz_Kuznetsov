#!/usr/bin/env python3
"""Run the predeclared Stage 4 bounded improvement study."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.evaluation import map_at_k  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402
from stage3.association import build_association_index  # noqa: E402
from stage4.association import (  # noqa: E402
    association_predictions,
    association_scores,
)
from stage4.content import content_predictions, content_scores  # noqa: E402
from stage4.data import load_stage4_data  # noqa: E402
from stage4.hybrid import fused_predictions  # noqa: E402
from stage4.repeat import repeat_predictions, repeat_scores  # noqa: E402

AR_PARAMS = {
    "min_count": 5,
    "min_confidence": 0.005,
    "min_lift": 1.0,
    "max_rules_per_antecedent": 50,
    "score_scheme": "confidence",
    "aggregate": "max",
    "max_antecedents_per_user": 50,
}


def _evaluate(data: dict[str, Any], predictions: dict[str, list[int]]) -> tuple[dict[str, Any], dict[str, float]]:
    result = map_at_k(
        predictions,
        data["labels"]["primary_top5"],
        users=data["cohort"],
        k=5,
    )
    aps = result.pop("aps")
    result["mean_ap_at_5"] = result.pop("map_at_k")
    result["mean_ap_at_5_with_positive"] = result.pop("map_with_positive")
    result["mean_ap_at_5_at_least_5_positive"] = result.pop(
        "map_with_at_least_k_positive"
    )
    return result, aps


def _validate(predictions: dict[str, list[int]], data: dict[str, Any]) -> None:
    allowed = set(data["val_candidates"])
    for uid in data["cohort"]:
        items = predictions.get(uid, [])
        if len(items) != 5 or len(set(items)) != 5 or not set(items) <= allowed:
            raise RuntimeError(f"Invalid predictions for {uid}: {items}")


def _save(
    stage4_config: Any,
    name: str,
    kind: str,
    params: dict[str, Any],
    predictions: dict[str, list[int]],
    data: dict[str, Any],
    runtime: float,
    details: dict[str, Any],
    shared_index_seconds: float | None = None,
) -> dict[str, Any]:
    _validate(predictions, data)
    metrics, aps = _evaluate(data, predictions)
    out = stage4_config.out("methods", name)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", params)
    write_json(out / "top5.json", predictions)
    write_json(out / "per_user_ap5.json", aps)
    write_json(out / "metrics.json", metrics)
    write_json(out / "details.json", details)
    return {
        "name": name,
        "kind": kind,
        "params": params,
        "status": "ok",
        "configuration_runtime_seconds": round(runtime, 3),
        "shared_index_runtime_seconds": (
            round(shared_index_seconds, 3) if shared_index_seconds is not None else None
        ),
        "total_runtime_seconds": round(
            runtime + (shared_index_seconds or 0.0), 3
        ),
        "metrics": metrics,
        "fallback": {
            "users_needing_fallback": details.get("users_needing_fallback"),
            "mean_scored_candidates_per_user": details.get(
                "mean_scored_candidates_per_user"
            ),
        },
        "artifacts": {
            "top5": str(out / "top5.json"),
            "per_user_ap5": str(out / "per_user_ap5.json"),
            "metrics": str(out / "metrics.json"),
            "details": str(out / "details.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)
    stage4_config.ensure_dirs()
    data = load_stage4_data(stage3_config)
    cohort = list(data["cohort"])
    candidates = list(data["val_candidates"])
    popularity = list(data["popularity_val"])
    stats = data["popularity"]

    individual_defs = [
        {
            "name": "repeat_50",
            "kind": "repeat",
            "params": {"count_variant": "sens50", "threshold": ">=50%", "tie_break": "training_popularity_then_id"},
        },
        {
            "name": "repeat_80",
            "kind": "repeat",
            "params": {"count_variant": "primary80", "threshold": ">=80%", "tie_break": "training_popularity_then_id"},
        },
        {
            "name": "cbf_category_log_pop",
            "kind": "cbf",
            "params": {"fields": ["category"], "profile_weight": "logcount", "use_idf": True, "actor_limit": 0, "profile_counts": "positive_airings", "tie_break": "training_popularity_then_id"},
        },
        {
            "name": "cbf_genre_cast_log_pop",
            "kind": "cbf",
            "params": {"fields": ["category", "genres", "decade", "director", "actors"], "profile_weight": "logcount", "use_idf": True, "actor_limit": 300, "profile_counts": "positive_airings", "tie_break": "training_popularity_then_id"},
        },
        {
            "name": "ar_c5_conf_max_80",
            "kind": "association_rules",
            "params": {**AR_PARAMS, "interaction_counts": "qualifying_airings_primary_>=80%", "tie_break": "program_id"},
        },
    ]
    write_json(
        stage4_config.out("search", "predeclared_individuals.json"),
        {
            "stage": 4,
            "declared_before_execution": True,
            "new_configuration_budget": 8,
            "individual_configurations": individual_defs,
            "fixed_inputs": (
                "Stage 3 frozen labels, validation candidate pool, cohorts, repeat "
                "policy, and fallback are unchanged."
            ),
        },
    )

    attempts: list[dict[str, Any]] = []
    write_json(
        stage4_config.out("search", "attempts.json"),
        {"stage": 4, "budget": 8, "attempts": attempts},
    )

    def record(attempt: dict[str, Any]) -> None:
        attempts.append(attempt)
        write_json(
            stage4_config.out("search", "attempts.json"),
            {"stage": 4, "budget": 8, "attempts": attempts},
        )

    results: dict[str, dict[str, Any]] = {}

    def run_def(definition: dict[str, Any], runner: Callable[[], tuple[dict[str, list[int]], dict[str, Any]]], shared: float | None = None) -> None:
        started = time.perf_counter()
        try:
            predictions, details = runner()
            runtime = time.perf_counter() - started
            attempt = _save(
                stage4_config,
                definition["name"],
                definition["kind"],
                definition["params"],
                predictions,
                data,
                runtime,
                details,
                shared,
            )
        except Exception as exc:
            attempt = {
                "name": definition["name"],
                "kind": definition["kind"],
                "params": definition["params"],
                "status": "failed",
                "configuration_runtime_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        record(attempt)
        results[definition["name"]] = attempt
        print(
            f"{definition['name']:26s} {attempt['status']:7s} "
            f"MAP@5={attempt.get('metrics', {}).get('mean_ap_at_5', float('nan')):.6f} "
            f"cfg={attempt['configuration_runtime_seconds']:.2f}s",
            flush=True,
        )

    # Repeat 50 and 80.
    for name, variant in (("repeat_50", "sens50"), ("repeat_80", "primary80")):
        run_def(
            next(item for item in individual_defs if item["name"] == name),
            lambda variant=variant: repeat_predictions(
                cohort,
                data["count_variants"][variant],
                data["user_to_train_idx"],
                candidates,
                popularity,
                stats,
            ),
        )

    # CBF variants.
    for name, fields, actor_limit in (
        ("cbf_category_log_pop", ["category"], 0),
        (
            "cbf_genre_cast_log_pop",
            ["category", "genres", "decade", "director", "actors"],
            300,
        ),
    ):
        run_def(
            next(item for item in individual_defs if item["name"] == name),
            lambda fields=fields, actor_limit=actor_limit: content_predictions(
                data,
                fields=fields,
                profile_weight="logcount",
                actor_limit=actor_limit,
                use_idf=True,
            ),
        )

    # AR index for >=80 interactions.
    data80 = dict(data)
    data80["user_items"] = data["count_variants"]["primary80"]
    started_index = time.perf_counter()
    index80 = build_association_index(data80)
    index80_seconds = time.perf_counter() - started_index
    run_def(
        next(item for item in individual_defs if item["name"] == "ar_c5_conf_max_80"),
        lambda: association_predictions(
            data80,
            index80,
            candidates,
            popularity,
            stats,
            tie_break="id",
            **AR_PARAMS,
        ),
        shared=index80_seconds,
    )

    successful_individuals = [
        results[item["name"]]
        for item in individual_defs
        if results.get(item["name"], {}).get("status") == "ok"
    ]
    if len(successful_individuals) != 5:
        raise RuntimeError("Individual study did not complete all five configurations")

    # Select strongest repeat and association methods from all evaluated individual
    # variants (Stage 3 plus Stage 4), using validation MAP@5 only.
    stage3_log = read_json(stage3_config.out("methods", "search_log.json"))
    stage3_by_name = {entry["name"]: entry for entry in stage3_log["entries"]}
    repeat_pool = {
        "stage3_repeat_any_positive": stage3_by_name["repeat_frequency"],
        "repeat_50": results["repeat_50"],
        "repeat_80": results["repeat_80"],
    }
    ar_pool = {
        "stage3_ar_c5_conf_max_any_positive": stage3_by_name["ar_c5_conf_max"],
        "ar_c5_conf_max_80": results["ar_c5_conf_max_80"],
    }
    best_repeat_name = max(
        repeat_pool,
        key=lambda name: (
            repeat_pool[name]["metrics"]["mean_ap_at_5"],
            -repeat_pool[name].get("configuration_runtime_seconds", 0.0),
            name,
        ),
    )
    best_ar_name = max(
        ar_pool,
        key=lambda name: (
            ar_pool[name]["metrics"]["mean_ap_at_5"],
            -ar_pool[name].get("configuration_runtime_seconds", 0.0),
            name,
        ),
    )

    repeat_variant = {
        "stage3_repeat_any_positive": "positive",
        "repeat_50": "sens50",
        "repeat_80": "primary80",
    }[best_repeat_name]
    repeat_selected_scores, repeat_detail = repeat_scores(
        cohort,
        data["count_variants"][repeat_variant],
        data["user_to_train_idx"],
        candidates,
    )
    if best_ar_name == "ar_c5_conf_max_80":
        ar_selected_data = data80
        ar_selected_index = index80
        ar_shared = index80_seconds
    else:
        ar_selected_data = data
        started_index = time.perf_counter()
        ar_selected_index = build_association_index(data)
        ar_shared = time.perf_counter() - started_index
    ar_selected_scores, ar_detail = association_scores(
        ar_selected_data, ar_selected_index, **AR_PARAMS
    )

    selection = {
        "rule": "highest validation MAP@5 among all individually evaluated variants; runtime only as tie-breaker",
        "best_repeat": {
            "name": best_repeat_name,
            "count_variant": repeat_variant,
            "map_at_5": repeat_pool[best_repeat_name]["metrics"]["mean_ap_at_5"],
            "details": repeat_detail,
        },
        "best_association": {
            "name": best_ar_name,
            "map_at_5": ar_pool[best_ar_name]["metrics"]["mean_ap_at_5"],
            "details": ar_detail,
            "shared_index_seconds": round(ar_shared, 3),
        },
        "stage3_ar_runtime_note": (
            "Stage 3 AR runtimes exclude the shared co-occurrence build; the measured "
            "shared build is recorded in diagnosis/category_cbf_ties.json."
        ),
    }
    write_json(stage4_config.out("search", "selection.json"), selection)

    hybrid_defs = [
        {
            "name": "hybrid_50_50",
            "kind": "hybrid",
            "params": {
                "repeat_source": best_repeat_name,
                "association_source": best_ar_name,
                "repeat_weight": 0.5,
                "association_weight": 0.5,
                "normalization": "per-user max normalization",
                "tie_break": "training_popularity_then_id",
            },
        },
        {
            "name": "hybrid_70_30",
            "kind": "hybrid",
            "params": {
                "repeat_source": best_repeat_name,
                "association_source": best_ar_name,
                "repeat_weight": 0.7,
                "association_weight": 0.3,
                "normalization": "per-user max normalization",
                "tie_break": "training_popularity_then_id",
            },
        },
    ]
    write_json(
        stage4_config.out("search", "predeclared_hybrids.json"),
        {
            "stage": 4,
            "declared_before_execution": True,
            "selection_rule": selection["rule"],
            "selected_repeat": best_repeat_name,
            "selected_association": best_ar_name,
            "hybrid_configurations": hybrid_defs,
            "fusion_rule": (
                "Per-user max-normalize each method's candidate scores, then take "
                "the weighted sum. No score rounding. Ties use training popularity "
                "then ascending program ID."
            ),
        },
    )

    for definition in hybrid_defs:
        run_def(
            definition,
            lambda definition=definition: fused_predictions(
                cohort,
                repeat_selected_scores,
                ar_selected_scores,
                candidates,
                popularity,
                stats,
                repeat_weight=definition["params"]["repeat_weight"],
                association_weight=definition["params"]["association_weight"],
            ),
        )

    successful = [entry for entry in attempts if entry["status"] == "ok"]
    comparison = [
        {
            "name": entry["name"],
            "kind": entry["kind"],
            "map_at_5": entry["metrics"]["mean_ap_at_5"],
            "map_with_positive": entry["metrics"]["mean_ap_at_5_with_positive"],
            "map_at_least_5_positive": entry["metrics"][
                "mean_ap_at_5_at_least_5_positive"
            ],
            "configuration_runtime_seconds": entry["configuration_runtime_seconds"],
            "shared_index_runtime_seconds": entry.get("shared_index_runtime_seconds"),
            "total_runtime_seconds": entry["total_runtime_seconds"],
            "users_needing_fallback": entry["fallback"]["users_needing_fallback"],
        }
        for entry in successful
    ]
    comparison.sort(key=lambda row: (-row["map_at_5"], row["name"]))
    write_json(stage4_config.out("search", "stage4_comparison.json"), comparison)
    write_json(
        stage4_config.out("checkpoints", "progress.json"),
        {
            "completed": [entry["name"] for entry in attempts],
            "stage4_attempt_count": len(attempts),
            "stage4_budget": 8,
            "statuses": {entry["name"]: entry["status"] for entry in attempts},
        },
    )
    print("\nStage 4 results")
    for row in comparison:
        print(
            f"{row['name']:30s} {row['kind']:18s} {row['map_at_5']:.6f} "
            f"cfg={row['configuration_runtime_seconds']:.2f}s "
            f"shared={row['shared_index_runtime_seconds']}"
        )
    print(f"selected repeat={best_repeat_name}; association={best_ar_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
