#!/usr/bin/env python3
"""Explain raw schedule id counts versus in-window candidate id counts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.config import load_config, parse_epoch  # noqa: E402
from stage2.io_utils import write_json  # noqa: E402


def scan(path: Path, start: int, end: int) -> dict:
    raw_ids: set[str] = set()
    owned_ids: set[str] = set()
    contained_ids: set[str] = set()
    before_ids: set[str] = set()
    after_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            at = parse_epoch(row["start_time"])
            show_id = (row.get("tv_show_id") or "").strip()
            if not show_id or show_id == "0":
                continue
            duration = int(float(row.get("duration", "")))
            if duration <= 0:
                continue
            raw_ids.add(show_id)
            if at < start:
                before_ids.add(show_id)
            elif at >= end:
                after_ids.add(show_id)
            else:
                owned_ids.add(show_id)
                if at + duration <= end:
                    contained_ids.add(show_id)
    return {
        "raw_ids": raw_ids,
        "owned_ids": owned_ids,
        "contained_ids": contained_ids,
        "before_ids": before_ids,
        "after_ids": after_ids,
    }


def main() -> int:
    config = load_config(ROOT / "config" / "stage2.json")
    result = {}
    for name in ("train", "val", "test"):
        window = config.windows[name]
        scanned = scan(config.schedule_files[name], window.start, window.end)
        expected = (
            scanned["owned_ids"] if name == "test" else scanned["contained_ids"]
        )
        with config.out("airings", f"airings_{name}.csv").open(newline="", encoding="utf-8") as handle:
            actual = {row["tv_show_id"] for row in csv.DictReader(handle)}
        crossing_only = scanned["owned_ids"] - scanned["contained_ids"]
        result[name] = {
            "window": [window.start_iso, window.end_iso],
            "raw_distinct_labelled_ids": len(scanned["raw_ids"]),
            "in_window_distinct_labelled_ids": len(scanned["owned_ids"]),
            "contained_in_window_distinct_ids": len(scanned["contained_ids"]),
            "crossing_window_end_only_ids": sorted(crossing_only, key=int),
            "airing_table_distinct_ids": len(actual),
            "scope_rule": (
                "test includes rows crossing the end; train/val include only airings fully contained in the window"
            ),
            "expected_vs_airing_table": {
                "missing_from_airing_table": sorted(expected - actual),
                "extra_in_airing_table": sorted(actual - expected),
            },
            "raw_ids_only_in_spillover": {
                "before_window": sorted(scanned["before_ids"] - scanned["owned_ids"], key=int),
                "after_window": sorted(scanned["after_ids"] - scanned["owned_ids"], key=int),
            },
            "explanation": (
                "Raw-file distinct ids exceed the candidate/airing-table count when ids occur only outside the "
                "nominal window or only in airings that cross and are excluded from train/val."
            ),
        }
    write_json(config.out("audit", "candidate_scope.json"), result)
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in item.items()
                    if key
                    in (
                        "raw_distinct_labelled_ids",
                        "in_window_distinct_labelled_ids",
                        "contained_in_window_distinct_ids",
                        "airing_table_distinct_ids",
                        "crossing_window_end_only_ids",
                        "raw_ids_only_in_spillover",
                        "expected_vs_airing_table",
                    )
                }
                for name, item in result.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
