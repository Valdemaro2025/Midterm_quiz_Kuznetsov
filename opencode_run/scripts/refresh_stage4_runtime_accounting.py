#!/usr/bin/env python3
"""Correct Stage 4 hybrid runtime accounting without rerunning experiments."""

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
    parser.add_argument("--config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    config = load_config(args.config)

    attempts_path = config.out("search", "attempts.json")
    combined_path = config.out("search", "combined_comparison.json")
    stage4_path = config.out("search", "stage4_comparison.json")
    attempts = read_json(attempts_path)
    entries = {entry["name"]: entry for entry in attempts["attempts"]}

    repeat = entries["repeat_50"]
    ar = entries["ar_c5_conf_max_80"]
    repeat_component = float(repeat["configuration_runtime_seconds"])
    ar_component = float(ar["configuration_runtime_seconds"])
    shared_component = float(ar.get("shared_index_runtime_seconds") or 0.0)
    fusion_by_name = {
        "hybrid_50_50": float(entries["hybrid_50_50"]["configuration_runtime_seconds"]),
        "hybrid_70_30": float(entries["hybrid_70_30"]["configuration_runtime_seconds"]),
    }

    accounting_by_name = {}
    for name, fusion_seconds in fusion_by_name.items():
        total = repeat_component + ar_component + shared_component + fusion_seconds
        accounting = {
            "repeat_component_seconds_estimate": round(repeat_component, 3),
            "association_component_seconds_estimate": round(ar_component, 3),
            "shared_index_seconds": round(shared_component, 3),
            "fusion_only_seconds": round(fusion_seconds, 3),
            "component_attributed_total_seconds": round(total, 3),
            "reuse_note": (
                "Repeat scores, association rules/scores, and the shared co-occurrence "
                "index were computed once before hybrid scoring and reused. The "
                "component-attributed total is an estimate, not a proven upper bound and "
                "not a directly measured end-to-end run; it is not the incremental cost "
                "of fused scoring alone. Do not add the shared index twice if both "
                "hybrids are built in one run."
            ),
        }
        accounting_by_name[name] = accounting
        entry = entries[name]
        entry["fusion_only_runtime_seconds"] = round(fusion_seconds, 3)
        entry["runtime_accounting"] = accounting
        entry["total_runtime_seconds"] = accounting["component_attributed_total_seconds"]

    write_json(attempts_path, attempts)

    stage4_rows = read_json(stage4_path)
    for row in stage4_rows:
        accounting = accounting_by_name.get(row["name"])
        if not accounting:
            continue
        row["fusion_only_runtime_seconds"] = accounting["fusion_only_seconds"]
        row["component_attributed_runtime_seconds"] = accounting[
            "component_attributed_total_seconds"
        ]
        row["runtime_accounting"] = accounting
    write_json(stage4_path, stage4_rows)

    combined = read_json(combined_path)
    for row in combined["rows"]:
        accounting = accounting_by_name.get(row["name"])
        if not accounting:
            continue
        row["fusion_only_runtime_seconds"] = accounting["fusion_only_seconds"]
        row["component_attributed_runtime_seconds"] = accounting[
            "component_attributed_total_seconds"
        ]
        row["runtime_accounting"] = accounting
        row["runtime_note"] = (
            "Hybrid reporting uses component-attributed total, not fusion-only time."
        )
    combined["hybrid_runtime_correction"] = {
        "applied": True,
        "reason": "Hybrid attempts originally reported only fused-score computation.",
        "components": {
            "repeat_50_configuration_seconds_estimate": round(repeat_component, 3),
            "ar_c5_conf_max_80_configuration_seconds_estimate": round(ar_component, 3),
            "shared_index_seconds": round(shared_component, 3),
        },
        "hybrids": accounting_by_name,
    }
    write_json(combined_path, combined)

    diagnostic = {
        "stage": 4,
        "experiments_rerun": False,
        "correction": "hybrid runtime now includes component computations and fusion",
        "components": {
            "repeat_50_configuration_seconds_estimate": round(repeat_component, 3),
            "ar_c5_conf_max_80_configuration_seconds_estimate": round(ar_component, 3),
            "shared_index_seconds": round(shared_component, 3),
        },
        "hybrids": accounting_by_name,
        "interpretation": (
            "Component-attributed estimate: the selected repeat and AR configurations "
            "also ranked Top-5 predictions, while hybrids reused raw scores. It is not "
            "a proven upper bound or a directly measured end-to-end run. Fused scoring "
            "alone was 0.366 s and 0.369 s."
        ),
    }
    write_json(config.out("audit", "runtime_accounting.json"), diagnostic)
    print(diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
