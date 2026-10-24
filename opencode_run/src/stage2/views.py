"""Split raw viewing intervals into observation windows.

A view is intersected with every applicable window.  A view crossing a cutoff
therefore contributes its pre-cutoff seconds to training and its observed
post-cutoff seconds to validation.  No interval is extended and no unobserved
seconds are invented.

Each window has its own ID maps.  In particular, the training axes are built
only from training segments; validation-only users or channels never enter the
training matrices.
"""

from __future__ import annotations

import csv
from array import array
from typing import Any, Mapping

import numpy as np

from .config import Stage2Config, Window, parse_epoch
from .io_utils import write_json


def split_view_into_windows(
    start: int,
    stop: int,
    windows: Mapping[str, Window],
) -> dict[str, tuple[int, int]]:
    """Return positive-length intersections of ``[start, stop)`` with each window."""
    if stop <= start:
        return {}
    pieces: dict[str, tuple[int, int]] = {}
    for name, window in windows.items():
        piece_start = max(start, window.start)
        piece_stop = min(stop, window.end)
        if piece_stop > piece_start:
            pieces[name] = (piece_start, piece_stop)
    return pieces


def split_views(config: Stage2Config) -> dict[str, Any]:
    windows = {name: config.windows[name] for name in ("train", "val")}
    user_ids: dict[str, list[str]] = {name: [] for name in windows}
    user_index: dict[str, dict[str, int]] = {name: {} for name in windows}
    channel_ids: dict[str, list[str]] = {name: [] for name in windows}
    channel_index: dict[str, dict[str, int]] = {name: {} for name in windows}
    arrays = {
        name: {
            "user": array("i"),
            "channel": array("i"),
            "start": array("q"),
            "stop": array("q"),
        }
        for name in windows
    }
    audit: dict[str, Any] = {
        "view_file": str(config.view_file),
        "total_rows": 0,
        "parse_failures": 0,
        "nonpositive_view_duration": 0,
        "rows_before_train": 0,
        "rows_after_val": 0,
        "rows_crossing_train_end": 0,
        "rows_crossing_val_end": 0,
        "rows_split_across_windows": 0,
        "clipped_at_train_end": 0,
        "clipped_at_val_end": 0,
        "segments_written": {name: 0 for name in windows},
        "written": {name: 0 for name in windows},
        "windows": {
            name: [windows[name].start_iso, windows[name].end_iso] for name in windows
        },
        "id_scope": {
            "train": "training segments only",
            "val": "validation segments only",
        },
    }

    with config.view_file.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
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

            if stop <= windows["train"].start:
                audit["rows_before_train"] += 1
            if start >= windows["val"].end:
                audit["rows_after_val"] += 1
            if start < windows["train"].end < stop:
                audit["rows_crossing_train_end"] += 1
            if start < windows["val"].end < stop:
                audit["rows_crossing_val_end"] += 1

            pieces = split_view_into_windows(start, stop, windows)
            if len(pieces) > 1:
                audit["rows_split_across_windows"] += 1
            for name, (piece_start, piece_stop) in pieces.items():
                if name == "train" and piece_stop == windows["train"].end and stop > piece_stop:
                    audit["clipped_at_train_end"] += 1
                if name == "val" and piece_stop == windows["val"].end and stop > piece_stop:
                    audit["clipped_at_val_end"] += 1
                idx = user_index[name].get(user_id)
                if idx is None:
                    idx = len(user_ids[name])
                    user_index[name][user_id] = idx
                    user_ids[name].append(user_id)
                ch_idx = channel_index[name].get(channel_id)
                if ch_idx is None:
                    ch_idx = len(channel_ids[name])
                    channel_index[name][channel_id] = ch_idx
                    channel_ids[name].append(channel_id)
                bucket = arrays[name]
                bucket["user"].append(idx)
                bucket["channel"].append(ch_idx)
                bucket["start"].append(piece_start)
                bucket["stop"].append(piece_stop)
                audit["segments_written"][name] += 1

    audit["written"] = dict(audit["segments_written"])
    audit["train_users"] = len(user_ids["train"])
    audit["val_users"] = len(user_ids["val"])
    audit["train_channels"] = len(channel_ids["train"])
    audit["val_channels"] = len(channel_ids["val"])

    for name, bucket in arrays.items():
        path = config.out("views", f"views_{name}.npz")
        np.savez_compressed(
            path,
            user_idx=np.frombuffer(bucket["user"], dtype=np.int32),
            channel_idx=np.frombuffer(bucket["channel"], dtype=np.int32),
            start=np.frombuffer(bucket["start"], dtype=np.int64),
            stop=np.frombuffer(bucket["stop"], dtype=np.int64),
        )
        audit[f"{name}_npz"] = str(path)

    id_maps = {
        name: {
            "user_ids": user_ids[name],
            "channel_ids": channel_ids[name],
        }
        for name in windows
    }
    write_json(config.out("views", "id_maps.json"), id_maps)
    audit["id_maps"] = str(config.out("views", "id_maps.json"))
    write_json(config.out("audit", "view_split_audit.json"), audit)
    return {
        "audit": audit,
        "windows": {
            name: {
                "user_ids": user_ids[name],
                "channel_ids": channel_ids[name],
                "user_index": user_index[name],
                "channel_index": channel_index[name],
            }
            for name in windows
        },
    }


def load_views(config: Stage2Config, name: str) -> dict[str, np.ndarray]:
    path = config.out("views", f"views_{name}.npz")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def load_id_maps(config: Stage2Config, name: str) -> tuple[list[str], list[str]]:
    import json

    payload = json.loads(config.out("views", "id_maps.json").read_text(encoding="utf-8"))
    item = payload[name]
    return item["user_ids"], item["channel_ids"]
