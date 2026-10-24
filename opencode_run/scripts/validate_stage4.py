#!/usr/bin/env python3
"""Validate Stage 4 saved predictions, metrics, and required diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config  # noqa: E402
from stage2.evaluation import map_at_k  # noqa: E402
from stage2.io_utils import read_json, write_json  # noqa: E402
from stage4.data import load_stage4_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-config", default=str(ROOT / "config" / "stage3.json"))
    parser.add_argument("--stage4-config", default=str(ROOT / "config" / "stage4.json"))
    args = parser.parse_args()
    stage3_config = load_config(args.stage3_config)
    stage4_config = load_config(args.stage4_config)
    data = load_stage4_data(stage3_config)
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    stage3_log = read_json(stage3_config.out("methods", "search_log.json"))
    stage4_log = read_json(stage4_config.out("search", "attempts.json"))
    stage3_count = len(stage3_log["entries"])
    stage4_count = len(stage4_log["attempts"])
    check("stage3_log_preserved_12", stage3_count == 12, stage3_count)
    check("stage4_budget_at_most_8", stage4_count <= 8, stage4_count)
    check("stage4_all_attempts_ok", all(e["status"] == "ok" for e in stage4_log["attempts"]))
    check("combined_attempt_count", stage3_count + stage4_count == 19, stage3_count + stage4_count)

    kinds: dict[str, int] = {}
    for entry in stage4_log["attempts"]:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    check(
        "stage4_kind_counts",
        kinds == {"repeat": 2, "cbf": 2, "association_rules": 1, "hybrid": 2},
        kinds,
    )

    pre_individuals = read_json(stage4_config.out("search", "predeclared_individuals.json"))
    pre_hybrids = read_json(stage4_config.out("search", "predeclared_hybrids.json"))
    check("individuals_predeclared", pre_individuals["declared_before_execution"] is True)
    check("hybrids_predeclared", pre_hybrids["declared_before_execution"] is True)
    check(
        "hybrid_weight_sum",
        all(
            abs(
                item["params"]["repeat_weight"]
                + item["params"]["association_weight"]
                - 1.0
            )
            < 1e-12
            for item in pre_hybrids["hybrid_configurations"]
        ),
    )

    candidates = set(data["val_candidates"])
    recomputed: list[dict[str, object]] = []
    for entry in stage4_log["attempts"]:
        name = entry["name"]
        out = stage4_config.out("methods", name)
        top5 = read_json(out / "top5.json")
        aps = read_json(out / "per_user_ap5.json")
        metrics = read_json(out / "metrics.json")
        coverage_ok = all(
            len(items) == 5
            and len(set(items)) == 5
            and set(items) <= candidates
            for items in top5.values()
        )
        result = map_at_k(
            top5,
            data["labels"]["primary_top5"],
            users=data["cohort"],
            k=5,
        )
        result_aps = result.pop("aps")
        metric_ok = (
            abs(metrics["mean_ap_at_5"] - result["map_at_k"]) < 1e-12
            and abs(
                metrics["mean_ap_at_5_with_positive"]
                - result["map_with_positive"]
            )
            < 1e-12
            and abs(
                metrics["mean_ap_at_5_at_least_5_positive"]
                - result["map_with_at_least_k_positive"]
            )
            < 1e-12
        )
        check(
            f"recompute:{name}",
            coverage_ok and metric_ok and aps == result_aps,
            {
                "coverage_ok": coverage_ok,
                "metric_ok": metric_ok,
                "aps_ok": aps == result_aps,
                "map_at_5": result["map_at_k"],
                "attempt_map": entry["metrics"]["mean_ap_at_5"],
            },
        )
        recomputed.append(
            {
                "name": name,
                "kind": entry["kind"],
                "map_at_5": result["map_at_k"],
                "map_with_positive": result["map_with_positive"],
                "map_at_least_5_positive": result["map_with_at_least_k_positive"],
            }
        )

    diagnosis = read_json(stage4_config.out("diagnosis", "category_cbf_ties.json"))
    category = diagnosis["stage3_category_cbf"]
    check("category_cbf_six_lists", category["distinct_top5_lists"] == 6, category["distinct_top5_lists"])
    check(
        "all_users_have_exact_ties",
        category["exact_ties"]["users_with_more_than_5_tied"] == 1260,
        category["exact_ties"]["users_with_more_than_5_tied"],
    )
    check(
        "most_top5_are_lowest_tied_ids",
        category["exact_ties"]["users_whose_top5_are_exactly_lowest_5_tied_ids"] == 1250,
        category["exact_ties"]["users_whose_top5_are_exactly_lowest_5_tied_ids"],
    )
    runtime = diagnosis["association_runtime_accounting"]
    check(
        "association_shared_runtime_reported",
        runtime["stage3_reported_runtime_excludes_shared_index"] is True
        and runtime["shared_cooccurrence_build_seconds_measured_now"] > 0,
        runtime["shared_cooccurrence_build_seconds_measured_now"],
    )

    hybrid_entries = {
        entry["name"]: entry for entry in stage4_log["attempts"]
        if entry["kind"] == "hybrid"
    }
    hybrid_accounting_ok = True
    hybrid_accounting_detail: dict[str, object] = {}
    for name, entry in hybrid_entries.items():
        accounting = entry.get("runtime_accounting") or {}
        component_total = (
            float(accounting.get("repeat_component_seconds_estimate", 0.0))
            + float(accounting.get("association_component_seconds_estimate", 0.0))
            + float(accounting.get("shared_index_seconds", 0.0))
            + float(accounting.get("fusion_only_seconds", 0.0))
        )
        row_ok = (
            bool(accounting)
            and abs(float(entry["total_runtime_seconds"]) - component_total) < 1e-9
            and component_total > float(accounting.get("fusion_only_seconds", 0.0))
            and "estimate" in str(accounting.get("reuse_note", "")).lower()
            and not any(key.endswith("upper_bound") for key in accounting)
        )
        hybrid_accounting_ok = hybrid_accounting_ok and row_ok
        hybrid_accounting_detail[name] = {
            "ok": row_ok,
            "component_total": component_total,
            "reported_total": entry["total_runtime_seconds"],
        }
    check("hybrid_runtime_component_attributed_estimate", hybrid_accounting_ok, hybrid_accounting_detail)

    combined = read_json(stage4_config.out("search", "combined_comparison.json"))
    combined_hybrids = {
        row["name"]: row for row in combined["rows"] if row["kind"] == "hybrid"
    }
    check(
        "combined_hybrid_runtime_corrected",
        set(combined_hybrids) == set(hybrid_entries)
        and all(
            row.get("component_attributed_runtime_seconds")
            == hybrid_entries[row["name"]]["runtime_accounting"][
                "component_attributed_total_seconds"
            ]
            for row in combined_hybrids.values()
        ),
        {
            name: row.get("component_attributed_runtime_seconds")
            for name, row in combined_hybrids.items()
        },
    )

    two = read_json(stage4_config.out("two_user", "analysis.json"))
    heavy = max(
        data["cohort"],
        key=lambda uid: (
            data["history"][uid]["distinct_primary80"],
            data["history"][uid]["total_primary80"],
            -int(uid),
        ),
    )
    light = min(
        (uid for uid in data["cohort"] if data["history"][uid]["distinct_primary80"] > 0),
        key=lambda uid: (
            data["history"][uid]["distinct_primary80"],
            data["history"][uid]["total_primary80"],
            int(uid),
        ),
    )
    check("heavy_user_rule", two["heavy_user"]["user_id"] == heavy, heavy)
    check("light_user_rule", two["light_user"]["user_id"] == light, light)
    worked = two["worked_ap_at_5"]
    labels = set(data["labels"]["primary_top5"].get(worked["user_id"], []))
    hits = [
        rank for rank, show_id in enumerate(worked["predictions"], start=1)
        if show_id in labels
    ]
    expected_ap = (
        1.0 / hits[0] / min(5, len(labels)) if hits else 0.0
    )
    check(
        "worked_ap_arithmetic",
        abs(expected_ap - worked["ap_at_5"]) < 1e-12,
        {"expected": expected_ap, "stored": worked["ap_at_5"]},
    )
    check(
        "failure_case_has_zero_hits",
        two["failure_case"]["validation_hits"] == [],
        two["failure_case"]["validation_hits"],
    )
    flag_rows = []
    for method in two["methods"]:
        for rows in method["top5_by_user"].values():
            flag_rows.extend(rows)
    flag_rows.extend(two["failure_case"]["top5"])
    flag_ok = all(
        row["in_training_history"] == (row["training_counts"]["positive"] > 0)
        and row["personal_seen_any_positive_coverage"]
        == (row["training_counts"]["positive"] > 0)
        and row["personal_seen_at_least_50_percent_coverage"]
        == (row["training_counts"]["sens50"] > 0)
        and row["personal_seen_at_least_80_percent_coverage"]
        == (row["training_counts"]["primary80"] > 0)
        for row in flag_rows
    )
    check(
        "two_user_personal_training_history_thresholds",
        flag_ok and bool(two.get("training_history_flag_definition")),
        {
            "rows_checked": len(flag_rows),
            "flag_definition": two.get("training_history_flag_definition"),
        },
    )

    global_recomputed: dict[int, dict[str, int]] = {}
    for variant in ("positive", "sens50", "primary80"):
        for items in data["count_variants"][variant].values():
            for show_id, count in items.items():
                entry = global_recomputed.setdefault(
                    show_id,
                    {
                        "training_users_any_positive": 0,
                        "training_users_at_least_50": 0,
                        "training_users_at_least_80": 0,
                        "positive_airings": 0,
                        "sens50_airings": 0,
                        "primary80_airings": 0,
                    },
                )
                if variant == "positive":
                    entry["training_users_any_positive"] += 1
                    entry["positive_airings"] += int(count)
                elif variant == "sens50":
                    entry["training_users_at_least_50"] += 1
                    entry["sens50_airings"] += int(count)
                else:
                    entry["training_users_at_least_80"] += 1
                    entry["primary80_airings"] += int(count)
    empty = {
        "training_users_any_positive": 0,
        "training_users_at_least_50": 0,
        "training_users_at_least_80": 0,
        "positive_airings": 0,
        "sens50_airings": 0,
        "primary80_airings": 0,
    }
    global_ok = all(
        row["global_item_coverage"]
        == global_recomputed.get(row["tv_show_id"], empty)
        for row in flag_rows
    )
    check(
        "two_user_global_item_coverage_independent",
        global_ok and bool(two.get("global_item_coverage_definition")),
        {
            "rows_checked": len(flag_rows),
            "global_definition": two.get("global_item_coverage_definition"),
        },
    )

    summary = {
        "all_passed": all(item["ok"] for item in checks),
        "checks_passed": sum(1 for item in checks if item["ok"]),
        "checks_total": len(checks),
        "checks": checks,
        "stage3_attempts": stage3_count,
        "stage4_attempts": stage4_count,
        "combined_attempts": stage3_count + stage4_count,
        "recomputed_stage4": sorted(recomputed, key=lambda row: (-row["map_at_5"], row["name"])),
    }
    write_json(stage4_config.out("audit", "stage4_validation.json"), summary)
    print(
        f"all_passed={summary['all_passed']} "
        f"checks={summary['checks_passed']}/{summary['checks_total']} "
        f"stage3={stage3_count} stage4={stage4_count}"
    )
    for item in checks:
        if not item["ok"]:
            print("FAIL", item)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
