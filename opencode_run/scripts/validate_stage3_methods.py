#!/usr/bin/env python3
"""Independently recompute Stage 3 validation metrics from saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.evaluation import map_at_k  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402
from stage2.labels import _canonical_hash  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "stage3.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    manifest = read_json(config.out("labels", "val_labels_manifest.json"))
    labels_path = config.out("labels", "val_labels_frozen.json")
    check("labels_frozen", manifest.get("frozen") is True)
    check(
        "labels_protocol",
        manifest.get("protocol_version") == config.raw["protocol_version"],
        manifest.get("protocol_version"),
    )
    labels = read_json(labels_path)
    check(
        "labels_sha256",
        _canonical_hash(labels) == manifest["label_sha256"],
    )
    primary_labels = labels["primary_top5"]

    cohort = [
        row["user_id"].strip()
        for row in __import__("csv").DictReader(
            config.submission_file.open(newline="", encoding="utf-8")
        )
    ]
    check("cohort_size", len(cohort) == 1260, len(cohort))

    views = read_json(config.out("audit", "view_split_audit.json"))
    check(
        "view_segment_conservation",
        views["total_rows"] + views["rows_split_across_windows"]
        == views["segments_written"]["train"] + views["segments_written"]["val"],
        {
            "total": views["total_rows"],
            "split": views["rows_split_across_windows"],
            "segments": views["segments_written"],
        },
    )
    check("cross_cutoff_rows_split", views["rows_split_across_windows"] > 0, views["rows_split_across_windows"])

    eligibility = read_json(config.out("audit", "candidate_eligibility.json"))
    check(
        "val_candidates",
        eligibility["val"]["candidate_count"] == 6361,
        eligibility["val"]["candidate_count"],
    )
    check(
        "test_candidates",
        eligibility["test"]["candidate_count"] == 6632,
        eligibility["test"]["candidate_count"],
    )
    check(
        "val_crossing_end_candidates",
        len(eligibility["val"]["candidate_ids_only_in_crossing_end_airings"]) == 6,
        eligibility["val"]["candidate_ids_only_in_crossing_end_airings"],
    )

    candidates = read_json(config.out("candidates", "candidates_val.json"))["ids"]
    allowed = set(candidates)
    log = read_json(config.out("methods", "search_log.json"))
    entries = log["entries"]
    check("search_attempted_at_most_12", len(entries) <= 12, len(entries))
    check("search_no_failures", all(e["status"] == "ok" for e in entries))
    kinds = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    check("two_baselines", kinds.get("baseline") == 2, kinds)
    check("five_cbf", kinds.get("cbf") == 5, kinds)
    check("five_association", kinds.get("association_rules") == 5, kinds)

    recomputed: list[dict[str, object]] = []
    for entry in entries:
        name = entry["name"]
        out = config.out("methods", name)
        top5 = read_json(out / "top5.json")
        aps = read_json(out / "per_user_ap5.json")
        metrics = read_json(out / "metrics.json")
        coverage_ok = all(
            len(items) == 5
            and len(set(items)) == 5
            and set(items) <= allowed
            for items in top5.values()
        )
        recomputed_result = map_at_k(top5, primary_labels, users=cohort, k=5)
        recomputed_aps = recomputed_result.pop("aps")
        metric_ok = (
            abs(metrics["mean_ap_at_5"] - recomputed_result["map_at_k"]) < 1e-12
            and abs(
                metrics["mean_ap_at_5_with_positive"]
                - recomputed_result["map_with_positive"]
            )
            < 1e-12
            and abs(
                metrics["mean_ap_at_5_at_least_5_positive"]
                - recomputed_result["map_with_at_least_k_positive"]
            )
            < 1e-12
        )
        aps_ok = aps == recomputed_aps
        check(
            f"predictions_and_metrics:{name}",
            coverage_ok and metric_ok and aps_ok,
            {
                "coverage_ok": coverage_ok,
                "metric_ok": metric_ok,
                "aps_ok": aps_ok,
                "map_at_5": recomputed_result["map_at_k"],
            },
        )
        recomputed.append(
            {
                "name": name,
                "kind": entry["kind"],
                "map_at_5": recomputed_result["map_at_k"],
                "map_with_positive": recomputed_result["map_with_positive"],
                "map_at_least_5_positive": recomputed_result[
                    "map_with_at_least_k_positive"
                ],
                "positive_size": len(
                    [u for u in cohort if primary_labels.get(u)]
                ),
                "at_least_5_positive_size": len(
                    [u for u in cohort if len(set(primary_labels.get(u, []))) >= 5]
                ),
            }
        )

    check("all_methods_recomputed", len(recomputed) == 12, len(recomputed))
    summary = {
        "all_passed": all(item["ok"] for item in checks),
        "checks_passed": sum(1 for item in checks if item["ok"]),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed_ranking": sorted(
            recomputed, key=lambda row: (-row["map_at_5"], row["name"])
        ),
    }
    write_json(config.out("audit", "stage3_methods_validation.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
