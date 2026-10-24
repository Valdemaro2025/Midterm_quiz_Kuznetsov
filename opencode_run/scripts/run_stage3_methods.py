#!/usr/bin/env python3
"""Evaluate four Stage 3 recommender families on frozen validation labels.

This script never refits on validation/test data and never writes a submission.
"""

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
from stage2.io_utils import write_json  # noqa: E402
from stage3.association import (  # noqa: E402
    association_predictions,
    build_association_index,
)
from stage3.baselines import popularity_predictions, repeat_predictions  # noqa: E402
from stage3.content import content_predictions  # noqa: E402
from stage3.data import load_stage3_data  # noqa: E402


def _validate_predictions(
    predictions: dict[str, list[int]], cohort: list[str], candidates: list[int], k: int = 5
) -> None:
    allowed = set(candidates)
    for uid in cohort:
        items = predictions[uid]
        if len(items) != k or len(set(items)) != k:
            raise RuntimeError(f"User {uid} does not have {k} unique predictions")
        if not set(items) <= allowed:
            raise RuntimeError(f"User {uid} has a prediction outside the candidate pool")


def _evaluate(
    data: dict[str, Any], predictions: dict[str, list[int]]
) -> tuple[dict[str, Any], dict[str, float]]:
    labels = data["labels"]["primary_top5"]
    result = map_at_k(predictions, labels, users=data["cohort"], k=5)
    aps = result.pop("aps")
    result["mean_ap_at_5"] = result.pop("map_at_k")
    result["mean_ap_at_5_with_positive"] = result.pop("map_with_positive")
    result["mean_ap_at_5_at_least_5_positive"] = result.pop(
        "map_with_at_least_k_positive"
    )
    result["cohort_with_positive_size"] = result.pop("cohort_with_positive_size")
    result["cohort_with_at_least_5_positive_size"] = result.pop(
        "cohort_with_at_least_k_positive_size"
    )
    return result, aps


def _write_result(
    config: Any,
    name: str,
    kind: str,
    params: dict[str, Any],
    predictions: dict[str, list[int]],
    data: dict[str, Any],
    runtime: float,
    detail: dict[str, Any],
) -> dict[str, Any]:
    _validate_predictions(predictions, data["cohort"], data["val_candidates"])
    metrics, aps = _evaluate(data, predictions)
    out = config.out("methods", name)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", params)
    write_json(out / "top5.json", predictions)
    write_json(out / "per_user_ap5.json", aps)
    write_json(out / "metrics.json", metrics)
    write_json(out / "details.json", detail)

    positive_size = metrics["cohort_with_positive_size"]
    ge5_size = metrics["cohort_with_at_least_5_positive_size"]
    return {
        "name": name,
        "kind": kind,
        "params": params,
        "seed": 0,
        "status": "ok",
        "runtime_seconds": round(runtime, 3),
        "metrics": metrics,
        "prediction_coverage": 1.0,
        "users_needing_fallback": detail.get("users_needing_fallback"),
        "mean_scored_candidates_per_user": detail.get(
            "mean_scored_candidates_per_user"
        ),
        "artifacts": {
            "top5": str(out / "top5.json"),
            "per_user_ap5": str(out / "per_user_ap5.json"),
            "metrics": str(out / "metrics.json"),
        },
        "cohort_sizes": {
            "all": len(data["cohort"]),
            "positive": positive_size,
            "at_least_5_positive": ge5_size,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "stage3.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    method_dir = config.out("methods")
    method_dir.mkdir(parents=True, exist_ok=True)

    data = load_stage3_data(config)
    cohort = data["cohort"]
    if len(cohort) != 1260:
        raise RuntimeError(f"Unexpected cohort size: {len(cohort)}")
    if len(data["val_candidates"]) != 6361:
        raise RuntimeError("Unexpected validation candidate count")
    if len(data["test_candidates"]) != 6632:
        raise RuntimeError("Unexpected test candidate count")

    log_path = method_dir / "search_log.json"
    search_log: list[dict[str, Any]] = []
    write_json(
        log_path,
        {
            "protocol_version": config.raw["protocol_version"],
            "labels_sha256": data["labels_manifest"]["label_sha256"],
            "search_budget": 12,
            "entries": search_log,
        },
    )

    def record(entry: dict[str, Any]) -> None:
        search_log.append(entry)
        write_json(
            log_path,
            {
                "protocol_version": config.raw["protocol_version"],
                "labels_sha256": data["labels_manifest"]["label_sha256"],
                "search_budget": 12,
                "entries": search_log,
            },
        )

    methods: list[tuple[str, str, dict[str, Any], Callable[[], tuple[dict[str, list[int]], dict[str, Any]]]]] = [
        (
            "popularity",
            "baseline",
            {"fallback": "global_train_qualifying_user_count"},
            lambda: popularity_predictions(
                cohort, data["val_candidates"], data["popularity_val"]
            ),
        ),
        (
            "repeat_frequency",
            "baseline",
            {"ranking": "training_positive_airings_then_popularity"},
            lambda: repeat_predictions(
                cohort,
                data["user_items"],
                data["user_to_train_idx"],
                data["val_candidates"],
                data["popularity_val"],
            ),
        ),
        (
            "cbf_category_log",
            "cbf",
            {
                "fields": ["category"],
                "profile_weight": "logcount",
                "actor_limit": 0,
                "use_idf": True,
            },
            lambda: content_predictions(
                data,
                fields=["category"],
                profile_weight="logcount",
                actor_limit=0,
            ),
        ),
        (
            "cbf_genre_log",
            "cbf",
            {
                "fields": ["category", "genres"],
                "profile_weight": "logcount",
                "actor_limit": 0,
                "use_idf": True,
            },
            lambda: content_predictions(
                data,
                fields=["category", "genres"],
                profile_weight="logcount",
                actor_limit=0,
            ),
        ),
        (
            "cbf_genre_decade_log",
            "cbf",
            {
                "fields": ["category", "genres", "decade"],
                "profile_weight": "logcount",
                "actor_limit": 0,
                "use_idf": True,
            },
            lambda: content_predictions(
                data,
                fields=["category", "genres", "decade"],
                profile_weight="logcount",
                actor_limit=0,
            ),
        ),
        (
            "cbf_genre_cast_log",
            "cbf",
            {
                "fields": ["category", "genres", "decade", "director", "actors"],
                "profile_weight": "logcount",
                "actor_limit": 300,
                "use_idf": True,
            },
            lambda: content_predictions(
                data,
                fields=["category", "genres", "decade", "director", "actors"],
                profile_weight="logcount",
                actor_limit=300,
            ),
        ),
        (
            "cbf_all_binary",
            "cbf",
            {
                "fields": ["category", "genres", "decade", "director", "actors"],
                "profile_weight": "binary",
                "actor_limit": 300,
                "use_idf": True,
            },
            lambda: content_predictions(
                data,
                fields=["category", "genres", "decade", "director", "actors"],
                profile_weight="binary",
                actor_limit=300,
            ),
        ),
    ]

    ar_index = build_association_index(data)
    ar_configs = [
        (
            "ar_c3_conf_loglift",
            {
                "min_count": 3,
                "min_confidence": 0.01,
                "min_lift": 1.0,
                "max_rules_per_antecedent": 50,
                "score_scheme": "conf_x_loglift",
                "aggregate": "sum",
                "max_antecedents_per_user": 50,
            },
        ),
        (
            "ar_c5_conf_sum",
            {
                "min_count": 5,
                "min_confidence": 0.02,
                "min_lift": 1.1,
                "max_rules_per_antecedent": 30,
                "score_scheme": "confidence",
                "aggregate": "sum",
                "max_antecedents_per_user": 30,
            },
        ),
        (
            "ar_c2_lift_sum",
            {
                "min_count": 2,
                "min_confidence": 0.01,
                "min_lift": 1.2,
                "max_rules_per_antecedent": 50,
                "score_scheme": "lift",
                "aggregate": "sum",
                "max_antecedents_per_user": 50,
            },
        ),
        (
            "ar_c5_conf_max",
            {
                "min_count": 5,
                "min_confidence": 0.005,
                "min_lift": 1.0,
                "max_rules_per_antecedent": 50,
                "score_scheme": "confidence",
                "aggregate": "max",
                "max_antecedents_per_user": 50,
            },
        ),
        (
            "ar_c10_conf_loglift_max",
            {
                "min_count": 10,
                "min_confidence": 0.02,
                "min_lift": 1.2,
                "max_rules_per_antecedent": 30,
                "score_scheme": "conf_x_loglift",
                "aggregate": "max",
                "max_antecedents_per_user": 30,
            },
        ),
    ]
    methods.extend(
        (
            name,
            "association_rules",
            params,
            lambda params=params: association_predictions(data, ar_index, **params),
        )
        for name, params in ar_configs
    )

    if len(methods) != 12:
        raise RuntimeError(f"Expected 12 configurations, got {len(methods)}")

    for name, kind, params, runner in methods:
        started = time.perf_counter()
        try:
            predictions, detail = runner()
            runtime = time.perf_counter() - started
            entry = _write_result(
                config, name, kind, params, predictions, data, runtime, detail
            )
        except Exception as exc:  # keep the search log complete on failures
            entry = {
                "name": name,
                "kind": kind,
                "params": params,
                "seed": 0,
                "status": "failed",
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        record(entry)
        write_json(
            config.out("checkpoints", "progress.json"),
            {
                "completed": [item["name"] for item in search_log],
                "remaining": [item[0] for item in methods[len(search_log):]],
                "statuses": {item["name"]: item["status"] for item in search_log},
            },
        )
        print(
            f"{name:28s} {entry['status']:7s} "
            f"MAP@5={entry.get('metrics', {}).get('mean_ap_at_5', float('nan')):.6f} "
            f"time={entry['runtime_seconds']:.2f}s",
            flush=True,
        )

    ok = [entry for entry in search_log if entry["status"] == "ok"]
    ok.sort(key=lambda entry: (-entry["metrics"]["mean_ap_at_5"], entry["name"]))
    comparison = [
        {
            "name": entry["name"],
            "kind": entry["kind"],
            "map_at_5": entry["metrics"]["mean_ap_at_5"],
            "map_with_positive": entry["metrics"]["mean_ap_at_5_with_positive"],
            "map_at_least_5_positive": entry["metrics"][
                "mean_ap_at_5_at_least_5_positive"
            ],
            "positive_size": entry["cohort_sizes"]["positive"],
            "at_least_5_positive_size": entry["cohort_sizes"]["at_least_5_positive"],
            "runtime_seconds": entry["runtime_seconds"],
            "prediction_coverage": entry["prediction_coverage"],
            "users_needing_fallback": entry["users_needing_fallback"],
        }
        for entry in ok
    ]
    write_json(method_dir / "comparison.json", comparison)
    write_json(method_dir / "search_log.json", {
        "protocol_version": config.raw["protocol_version"],
        "labels_sha256": data["labels_manifest"]["label_sha256"],
        "search_budget": 12,
        "attempted": len(search_log),
        "entries": search_log,
        "ranking": comparison,
    })

    print("\nname,kind,MAP@5,positive,>=5,time")
    for row in comparison:
        print(
            f"{row['name']},{row['kind']},{row['map_at_5']:.6f},"
            f"{row['map_with_positive']:.6f},{row['map_at_least_5_positive']:.6f},"
            f"{row['runtime_seconds']:.2f}"
        )
    print(f"\nArtifacts: {method_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
