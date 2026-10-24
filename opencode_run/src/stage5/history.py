"""Full-history reconstruction for the Stage 5 final refit."""

from __future__ import annotations

import csv
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from stage2.config import Stage2Config, Window, parse_epoch
from stage2.interactions import build_interactions
from stage2.io_utils import write_json
from stage2.schedule import (
    Airings,
    LoadedSchedule,
    ScheduleRow,
    aggregate_metadata,
    build_airings,
    load_schedule_file,
)
from stage2.views import split_view_into_windows


HISTORY_FILES = ("history_11_20", "history_21_30")


def read_template_users(config: Stage2Config) -> list[str]:
    with config.submission_file.open(newline="", encoding="utf-8") as handle:
        return [row["user_id"] for row in csv.DictReader(handle)]


def _schedule_window(config: Stage2Config, name: str) -> Window:
    bounds = config.raw["schedule_windows"][name]
    return Window(name, parse_epoch(bounds[0]), parse_epoch(bounds[1]))


def _row_identity(row: ScheduleRow) -> tuple[str, int, int, str]:
    return (row.channel_id, row.start, row.end, row.show_id)


def dedupe_history_rows(
    rows: Iterable[tuple[int, ScheduleRow]],
) -> tuple[list[ScheduleRow], dict[str, Any]]:
    """Keep one row per deterministic airing identity.

    Source priority is the tuple's first element; lower values win. Within a
    source, callers pass rows in source-row order. This makes cross-file
    ownership and duplicate resolution reproducible.
    """
    kept: dict[tuple[str, int, int, str], ScheduleRow] = {}
    exact_duplicates = 0
    identity_duplicates = 0
    conflicting_duplicates: list[dict[str, Any]] = []
    for priority, row in sorted(rows, key=lambda item: (item[0], item[1].source_row)):
        identity = _row_identity(row)
        previous = kept.get(identity)
        if previous is None:
            kept[identity] = row
            continue
        if previous.key == row.key:
            exact_duplicates += 1
        else:
            identity_duplicates += 1
            if len(conflicting_duplicates) < 10:
                conflicting_duplicates.append(
                    {
                        "identity": list(identity),
                        "kept_source": previous.source_file,
                        "kept_row": previous.source_row,
                        "dropped_source": row.source_file,
                        "dropped_row": row.source_row,
                    }
                )
    ordered = sorted(
        kept.values(),
        key=lambda row: (
            row.channel_id,
            row.start,
            row.end,
            row.show_id,
            row.source_file,
            row.source_row,
        ),
    )
    return ordered, {
        "identity": "(channel_id, start_epoch, end_epoch, tv_show_id)",
        "exact_duplicates_dropped": exact_duplicates,
        "identity_duplicates_dropped": identity_duplicates,
        "conflicting_identity_duplicates": conflicting_duplicates,
        "kept_rows": len(ordered),
        "priority": "history_11_20 before history_21_30; then source row order",
    }


def build_history_schedule(config: Stage2Config) -> dict[str, Any]:
    loaded = {
        name: load_schedule_file(
            config.schedule_files[name],
            name,
            _schedule_window(config, name),
        )
        for name in HISTORY_FILES
    }
    priority = {name: idx for idx, name in enumerate(HISTORY_FILES)}
    combined = [
        (priority[name], row)
        for name in HISTORY_FILES
        for row in loaded[name].owned_rows
    ]
    rows, dedupe_audit = dedupe_history_rows(combined)
    window = config.windows["train"]
    audit = {
        "window": [window.start_iso, window.end_iso],
        "files": {name: loaded[name].audit for name in HISTORY_FILES},
        "dedupe": dedupe_audit,
        "source_rows": {
            name: len(loaded[name].owned_rows) for name in HISTORY_FILES
        },
        "combined_rows_before_dedupe": len(combined),
    }
    shuffled = LoadedSchedule(
        "history",
        window,
        rows,
        [row for name in HISTORY_FILES for row in loaded[name].spillover_rows],
        audit,
    )
    airings = build_airings(
        shuffled,
        include_crossing_end=False,
        airing_csv=config.out("airings", "airings_history.csv"),
    )
    boundary = parse_epoch("2020-05-18T00:00:00")
    restored = [row for row in airings.rows if row.start < boundary < row.end]
    audit["airings"] = airings.audit
    audit["restored_may18_crossing_airings"] = len(restored)
    audit["restored_may18_examples"] = [row.as_dict() for row in restored[:5]]
    audit["excluded_full_history_crossing_end"] = airings.audit["excluded"][
        "crossing_window_end"
    ]
    metadata = aggregate_metadata(
        airings,
        config.out("metadata", "metadata_history.csv"),
    )
    audit["metadata"] = metadata
    write_json(config.out("audit", "history_schedule_audit.json"), audit)
    return {
        "airings": airings,
        "audit": audit,
        "dedupe": dedupe_audit,
        "restored_may18_crossing_airings": len(restored),
    }


def build_test_candidates(config: Stage2Config) -> dict[str, Any]:
    window = config.windows["test"]
    loaded = load_schedule_file(
        config.schedule_files["test"],
        "test",
        _schedule_window(config, "test"),
    )
    airings = build_airings(
        loaded,
        include_crossing_end=True,
        airing_csv=config.out("airings", "candidate_airings_test.csv"),
    )
    candidates = sorted({int(row.show_id) for row in airings.rows})
    audit = {
        "window": [window.start_iso, window.end_iso],
        "rule": config.raw["test_candidate_rule"],
        "source_rows": loaded.audit,
        "airings": airings.audit,
        "candidate_count": len(candidates),
        "candidate_ids": candidates,
    }
    write_json(config.out("candidates", "candidates_test.json"), audit)
    return {"airings": airings, "candidates": candidates, "audit": audit}


def build_history_views(config: Stage2Config) -> dict[str, Any]:
    window = config.windows["train"]
    windows = {"history": window}
    user_ids: list[str] = []
    user_index: dict[str, int] = {}
    channel_ids: list[str] = []
    channel_index: dict[str, int] = {}
    users = array("i")
    channels = array("i")
    starts = array("q")
    stops = array("q")
    audit = {
        "view_file": str(config.view_file),
        "window": [window.start_iso, window.end_iso],
        "total_rows": 0,
        "parse_failures": 0,
        "nonpositive_view_duration": 0,
        "rows_before_window": 0,
        "rows_after_window": 0,
        "rows_clipped_at_window_end": 0,
        "rows_clipped_at_window_start": 0,
        "segments_written": 0,
    }
    with config.view_file.open(newline="", encoding="utf-8", errors="replace") as handle:
        for raw in csv.DictReader(handle):
            audit["total_rows"] += 1
            try:
                start = parse_epoch(raw["start_time"])
                stop = parse_epoch(raw["stop_time"])
            except (KeyError, ValueError):
                audit["parse_failures"] += 1
                continue
            if stop <= start:
                audit["nonpositive_view_duration"] += 1
                continue
            user_id = (raw.get("user_id") or "").strip()
            channel_id = (raw.get("vsetv_id") or "").strip()
            if not user_id or not channel_id:
                audit["parse_failures"] += 1
                continue
            if stop <= window.start:
                audit["rows_before_window"] += 1
            if start >= window.end:
                audit["rows_after_window"] += 1
            pieces = split_view_into_windows(start, stop, windows)
            piece = pieces.get("history")
            if piece is None:
                continue
            piece_start, piece_stop = piece
            if piece_start > start:
                audit["rows_clipped_at_window_start"] += 1
            if piece_stop < stop:
                audit["rows_clipped_at_window_end"] += 1
            idx = user_index.get(user_id)
            if idx is None:
                idx = len(user_ids)
                user_index[user_id] = idx
                user_ids.append(user_id)
            ch_idx = channel_index.get(channel_id)
            if ch_idx is None:
                ch_idx = len(channel_ids)
                channel_index[channel_id] = ch_idx
                channel_ids.append(channel_id)
            users.append(idx)
            channels.append(ch_idx)
            starts.append(piece_start)
            stops.append(piece_stop)
            audit["segments_written"] += 1
    npz_path = config.out("views", "views_history.npz")
    np.savez_compressed(
        npz_path,
        user_idx=np.frombuffer(users, dtype=np.int32),
        channel_idx=np.frombuffer(channels, dtype=np.int32),
        start=np.frombuffer(starts, dtype=np.int64),
        stop=np.frombuffer(stops, dtype=np.int64),
    )
    write_json(
        config.out("views", "id_maps_history.json"),
        {"history": {"user_ids": user_ids, "channel_ids": channel_ids}},
    )
    audit.update(
        {
            "users": len(user_ids),
            "channels": len(channel_ids),
            "npz": str(npz_path),
            "max_observed_stop": int(np.max(stops)) if stops else None,
            "max_allowed_stop": window.end,
            "no_future_observations": bool(stops) and int(np.max(stops)) <= window.end,
        }
    )
    write_json(config.out("audit", "history_view_audit.json"), audit)
    return {
        "user_ids": user_ids,
        "channel_ids": channel_ids,
        "user_index": user_index,
        "channel_index": channel_index,
        "audit": audit,
    }


def build_history_interactions(
    config: Stage2Config,
    airings: Airings,
    views: Mapping[str, Any],
) -> dict[str, Any]:
    audit = build_interactions(
        config,
        "history",
        airings,
        views["channel_ids"],
        "history",
    )
    return audit


def read_count_variants(path: Path) -> dict[str, dict[int, dict[int, int]]]:
    variants: dict[str, dict[int, dict[int, int]]] = {
        "positive": defaultdict(dict),
        "sens50": defaultdict(dict),
        "primary80": defaultdict(dict),
    }
    fields = {
        "positive": "positive_airings",
        "sens50": "qualifying_airings_sensitivity",
        "primary80": "qualifying_airings_primary",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            user_idx = int(row["user_idx"])
            show_id = int(row["tv_show_id"])
            for variant, field in fields.items():
                value = int(row[field])
                if value > 0:
                    variants[variant][user_idx][show_id] = value
    return {
        name: {user: dict(items) for user, items in users.items()}
        for name, users in variants.items()
    }


def build_popularity(
    count_variants: Mapping[str, Mapping[int, Mapping[int, int]]],
    candidate_ids: Iterable[int],
) -> tuple[list[int], dict[int, tuple[int, int]], list[dict[str, Any]]]:
    user_sets: dict[int, set[int]] = defaultdict(set)
    airing_counts: Counter[int] = Counter()
    for user_idx, items in count_variants["primary80"].items():
        for show_id, count in items.items():
            user_sets[show_id].add(user_idx)
            airing_counts[show_id] += int(count)
    rows = []
    for show_id in sorted({int(x) for x in candidate_ids}):
        rows.append(
            {
                "tv_show_id": show_id,
                "distinct_training_users_primary": len(user_sets.get(show_id, set())),
                "qualifying_training_airings_primary": int(
                    airing_counts.get(show_id, 0)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["distinct_training_users_primary"],
            -row["qualifying_training_airings_primary"],
            row["tv_show_id"],
        )
    )
    stats = {
        int(row["tv_show_id"]): (
            int(row["distinct_training_users_primary"]),
            int(row["qualifying_training_airings_primary"]),
        )
        for row in rows
    }
    return [int(row["tv_show_id"]) for row in rows], stats, rows


def cold_start_stats(
    cohort: Iterable[str],
    user_to_train_idx: Mapping[str, int],
    count_variants: Mapping[str, Mapping[int, Mapping[int, int]]],
) -> dict[str, Any]:
    rows = []
    for uid in cohort:
        idx = user_to_train_idx.get(uid)
        counts = {
            variant: count_variants[variant].get(idx, {}) if idx is not None else {}
            for variant in ("positive", "sens50", "primary80")
        }
        rows.append(
            {
                "user_id": uid,
                "matched_to_full_history_views": idx is not None,
                "distinct_positive": len(counts["positive"]),
                "distinct_sens50": len(counts["sens50"]),
                "distinct_primary80": len(counts["primary80"]),
                "total_positive": sum(counts["positive"].values()),
                "total_sens50": sum(counts["sens50"].values()),
                "total_primary80": sum(counts["primary80"].values()),
            }
        )
    distributions = {}
    for field in ("distinct_positive", "distinct_sens50", "distinct_primary80"):
        sizes = [row[field] for row in rows]
        distributions[field] = {
            "zero": sum(1 for value in sizes if value == 0),
            "one_to_four": sum(1 for value in sizes if 0 < value < 5),
            "at_least_five": sum(1 for value in sizes if value >= 5),
            "positive": sum(1 for value in sizes if value > 0),
        }
    return {
        "cohort_size": len(rows),
        "matched_to_full_history_views": sum(
            1 for row in rows if row["matched_to_full_history_views"]
        ),
        "distinct_positive": distributions["distinct_positive"],
        "distinct_sens50": distributions["distinct_sens50"],
        "distinct_primary80": distributions["distinct_primary80"],
        "per_user": rows,
    }
