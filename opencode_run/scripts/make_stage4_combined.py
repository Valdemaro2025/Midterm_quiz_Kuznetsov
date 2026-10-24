#!/usr/bin/env python3
"""Combine Stage 3 and Stage 4 validation results without changing either search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)

    stage3_log = read_json(stage3_config.out("methods", "search_log.json"))
    stage4_log = read_json(stage4_config.out("search", "attempts.json"))
    diagnosis = read_json(stage4_config.out("diagnosis", "category_cbf_ties.json"))
    stage3_shared = diagnosis["association_runtime_accounting"][
        "shared_cooccurrence_build_seconds_measured_now"
    ]

    rows = []
    for entry in stage3_log["entries"]:
        if entry["status"] != "ok":
            continue
        is_ar = entry["kind"] == "association_rules"
        rows.append(
            {
                "source": "stage3",
                "name": entry["name"],
                "kind": entry["kind"],
                "map_at_5": entry["metrics"]["mean_ap_at_5"],
                "map_with_positive": entry["metrics"]["mean_ap_at_5_with_positive"],
                "map_at_least_5_positive": entry["metrics"][
                    "mean_ap_at_5_at_least_5_positive"
                ],
                "configuration_runtime_seconds": entry["runtime_seconds"],
                "shared_index_cost_seconds": stage3_shared if is_ar else None,
                "shared_cost_note": (
                    "one shared any-positive co-occurrence build; not included in the "
                    "reported per-configuration runtime"
                    if is_ar
                    else None
                ),
                "users_needing_fallback": entry.get("users_needing_fallback"),
            }
        )
    for entry in stage4_log["attempts"]:
        if entry["status"] != "ok":
            continue
        rows.append(
            {
                "source": "stage4",
                "name": entry["name"],
                "kind": entry["kind"],
                "map_at_5": entry["metrics"]["mean_ap_at_5"],
                "map_with_positive": entry["metrics"]["mean_ap_at_5_with_positive"],
                "map_at_least_5_positive": entry["metrics"][
                    "mean_ap_at_5_at_least_5_positive"
                ],
                "configuration_runtime_seconds": entry[
                    "configuration_runtime_seconds"
                ],
                "shared_index_cost_seconds": entry.get("shared_index_runtime_seconds"),
                "shared_cost_note": (
                    "shared >=80% co-occurrence build is paid once and reused by the "
                    "AR and hybrid configurations"
                    if entry.get("shared_index_runtime_seconds")
                    else None
                ),
                "users_needing_fallback": entry["fallback"]["users_needing_fallback"],
            }
        )

    rows.sort(key=lambda row: (-row["map_at_5"], row["source"], row["name"]))
    payload = {
        "stage3_attempts": len(stage3_log["entries"]),
        "stage4_attempts": len(stage4_log["attempts"]),
        "combined_attempts": len(stage3_log["entries"]) + len(stage4_log["attempts"]),
        "stage4_budget": 8,
        "labels_sha256": stage3_log["labels_sha256"],
        "note": (
            "Both searches use the same Stage 3 frozen validation labels, candidate "
            "pool, cohort, repeat policy, and fallback. Method selection and reported "
            "performance therefore use the same holdout."
        ),
        "rows": rows,
    }
    write_json(stage4_config.out("search", "combined_comparison.json"), payload)
    print(
        f"combined attempts={payload['combined_attempts']} "
        f"rows={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
