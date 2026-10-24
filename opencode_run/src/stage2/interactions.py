"""User-airing coverage construction.

For every user and airing, the covered time is the union of the user's view
intervals intersected with that airing.  A view may cover several consecutive
airings; all of them receive their own coverage.  Intervals are half-open, and
overlapping/adjacent view intervals are unioned before summing so no second is
counted twice.  Unobserved gaps are never counted in the primary result.
"""

from __future__ import annotations

import json
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Stage2Config
from .intervals import merge_intervals, overlap_seconds, qualify
from .io_utils import write_json
from .schedule import Airings, ChannelIndex


def overlapping_airings(
    index: ChannelIndex,
    start: int,
    end: int,
) -> Iterable[tuple[int, int]]:
    """Yield ``(airing_idx, overlap_seconds)`` for every overlapping airing.

    A segment tree over the sorted starts stores the maximum end in each range.
    This handles deliberately overlapping schedule rows without the classic
    "stop at the first short earlier airing" bug.  The query cost is
    O(log n + k log n)-ish for k returned overlaps, rather than scanning every
    schedule row on the channel.
    """
    if end <= start or not index.starts:
        return
    starts = index.starts
    ends = index.ends
    tree = index.max_end_tree
    size = len(tree) // 2
    n = len(starts)
    stack = [(1, 0, size)]
    while stack:
        node, lo, hi = stack.pop()
        if lo >= n or starts[lo] >= end or tree[node] <= start:
            continue
        if hi - lo == 1:
            if ends[lo] > start:
                overlap = min(end, ends[lo]) - max(start, starts[lo])
                if overlap > 0:
                    yield index.airing_indices[lo], overlap
            continue
        mid = (lo + hi) // 2
        stack.append((2 * node + 1, mid, hi))
        stack.append((2 * node, lo, mid))


def _process_user(
    user_idx: int,
    user_starts: list[int],
    user_stops: list[int],
    user_channels: list[int],
    channel_ids: list[str],
    airings: Airings,
    primary_qual: float,
    sensitivity_qual: float,
    bridged_gap: int,
) -> tuple[dict[int, int], dict[int, int], dict[str, int]]:
    covered: dict[int, int] = {}
    bridged: dict[int, int] = {}
    stats = Counter()
    # group by channel within the user
    order = sorted(range(len(user_starts)), key=lambda k: (user_channels[k], user_starts[k]))
    k = 0
    while k < len(order):
        channel_idx = user_channels[order[k]]
        channel_id = channel_ids[channel_idx]
        group_positions = []
        while k < len(order) and user_channels[order[k]] == channel_idx:
            group_positions.append(order[k])
            k += 1
        intervals = [(user_starts[p], user_stops[p]) for p in group_positions]
        index = airings.by_channel.get(channel_id)
        if index is None:
            seconds = sum(e - s for s, e in intervals)
            if channel_id in airings.source_channels:
                stats["views_on_channel_with_schedule_but_no_labelled_airings"] += len(intervals)
                stats["seconds_on_channel_with_schedule_but_no_labelled_airings"] += seconds
            else:
                stats["views_on_channel_absent_from_schedule"] += len(intervals)
                stats["seconds_on_channel_absent_from_schedule"] += seconds
            continue
        merged = merge_intervals(intervals, max_gap=0)
        stats["merged_primary_sessions"] += len(merged)
        for s, e in merged:
            found = False
            for airing_idx, overlap in overlapping_airings(index, s, e):
                found = True
                covered[airing_idx] = covered.get(airing_idx, 0) + overlap
            if not found:
                stats["merged_sessions_without_labelled_airing_overlap"] += 1
                stats["seconds_without_labelled_airing_overlap"] += e - s
        if bridged_gap > 0:
            bridged_intervals = merge_intervals(intervals, max_gap=bridged_gap)
            stats["merged_bridged_sessions"] += len(bridged_intervals)
            for s, e in bridged_intervals:
                for airing_idx, overlap in overlapping_airings(index, s, e):
                    bridged[airing_idx] = bridged.get(airing_idx, 0) + overlap
        else:
            bridged = dict(covered)
    return covered, bridged, dict(stats)


def build_interactions(
    config: Stage2Config,
    window_name: str,
    airings: Airings,
    channel_ids: list[str],
    output_prefix: str,
    compute_bridged_sensitivity: bool = True,
) -> dict[str, Any]:
    """Build user-airing rows and user-program qualifying counts for a window."""
    started = time.perf_counter()
    data = np.load(config.out("views", f"views_{window_name}.npz"))
    user = data["user_idx"].astype(np.int64)
    channel = data["channel_idx"].astype(np.int64)
    start = data["start"].astype(np.int64)
    stop = data["stop"].astype(np.int64)
    order = np.lexsort((start, channel, user))
    user = user[order]
    channel = channel[order]
    start = start[order]
    stop = stop[order]

    out_user = array("i")
    out_airing = array("i")
    out_covered = array("q")
    out_duration = array("q")
    out_show = array("q")
    out_bridged = array("q")

    counts: dict[tuple[int, int], list[int]] = {}
    # list fields: [qual_primary, qual_sensitivity, positive, bridged_qual_primary, bridged_positive, covered_total]
    audit: dict[str, Any] = {
        "window": window_name,
        "views_loaded": int(len(user)),
        "users_with_views": int(len(np.unique(user))),
        "channels_with_views": int(len(np.unique(channel))),
        "user_airing_rows": 0,
        "user_program_rows": 0,
        "views_on_channel_absent_from_schedule": 0,
        "seconds_on_channel_absent_from_schedule": 0,
        "views_on_channel_with_schedule_but_no_labelled_airings": 0,
        "seconds_on_channel_with_schedule_but_no_labelled_airings": 0,
        "merged_primary_sessions": 0,
        "merged_bridged_sessions": 0,
        "merged_sessions_without_labelled_airing_overlap": 0,
        "seconds_without_labelled_airing_overlap": 0,
        "coverage_primary_qualifying_airings": 0,
        "coverage_sensitivity_qualifying_airings": 0,
        "positive_coverage_airings": 0,
        "bridged_primary_qualifying_airings": 0,
        "bridged_positive_airings": 0,
        "airing_duration_leq_zero_excluded": 0,
        "show_id_zero_excluded": 0,
        "coverage_quantiles": {},
        "runtime_seconds": 0.0,
        "outputs": {},
    }

    bounds = np.flatnonzero(np.r_[True, user[1:] != user[:-1], True])
    n_users = len(bounds) - 1
    for user_number, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        user_idx = int(user[a])
        u_starts = start[a:b].tolist()
        u_stops = stop[a:b].tolist()
        u_channels = channel[a:b].tolist()
        covered, bridged, stats = _process_user(
            user_idx,
            u_starts,
            u_stops,
            u_channels,
            channel_ids,
            airings,
            config.primary_coverage,
            config.sensitivity_coverage,
            config.merge_gap_sensitivity_seconds if compute_bridged_sensitivity else 0,
        )
        for key in (
            "views_on_channel_absent_from_schedule",
            "seconds_on_channel_absent_from_schedule",
            "views_on_channel_with_schedule_but_no_labelled_airings",
            "seconds_on_channel_with_schedule_but_no_labelled_airings",
            "merged_primary_sessions",
            "merged_bridged_sessions",
            "merged_sessions_without_labelled_airing_overlap",
            "seconds_without_labelled_airing_overlap",
        ):
            audit[key] += stats.get(key, 0)

        # Ordered iteration gives deterministic output rows.
        for airing_idx in sorted(covered):
            airing = airings.rows[airing_idx]
            if airing.duration <= 0:
                audit["airing_duration_leq_zero_excluded"] += 1
                continue
            if not airing.show_id or airing.show_id == "0":
                audit["show_id_zero_excluded"] += 1
                continue
            covered_sec = covered[airing_idx]
            if covered_sec <= 0:
                continue
            b_sec = bridged.get(airing_idx, 0)
            out_user.append(user_idx)
            out_airing.append(airing_idx)
            out_covered.append(covered_sec)
            out_duration.append(airing.duration)
            out_show.append(int(airing.show_id))
            out_bridged.append(b_sec)
            audit["user_airing_rows"] += 1
            is_positive = True
            is_primary = qualify(covered_sec, airing.duration, config.primary_coverage)
            is_sensitivity = qualify(covered_sec, airing.duration, config.sensitivity_coverage)
            if is_positive:
                audit["positive_coverage_airings"] += 1
            if is_primary:
                audit["coverage_primary_qualifying_airings"] += 1
            if is_sensitivity:
                audit["coverage_sensitivity_qualifying_airings"] += 1
            b_primary = qualify(b_sec, airing.duration, config.primary_coverage)
            if b_primary:
                audit["bridged_primary_qualifying_airings"] += 1
            if b_sec > 0:
                audit["bridged_positive_airings"] += 1
            key = (user_idx, int(airing.show_id))
            entry = counts.get(key)
            if entry is None:
                entry = [0, 0, 0, 0, 0, 0]
                counts[key] = entry
            entry[0] += int(is_primary)
            entry[1] += int(is_sensitivity)
            entry[2] += int(is_positive)
            entry[3] += int(b_primary)
            entry[4] += int(b_sec > 0)
            entry[5] += covered_sec
        if user_number % 500 == 0:
            print(f"  {window_name}: processed {user_number}/{n_users} users", flush=True)

    out_npz = config.out("interactions", f"{output_prefix}_user_airing.npz")
    np.savez_compressed(
        out_npz,
        user_idx=np.frombuffer(out_user, dtype=np.int32),
        airing_idx=np.frombuffer(out_airing, dtype=np.int32),
        covered_sec=np.frombuffer(out_covered, dtype=np.int64),
        airing_duration=np.frombuffer(out_duration, dtype=np.int64),
        show_id=np.frombuffer(out_show, dtype=np.int64),
        bridged_covered_sec=np.frombuffer(out_bridged, dtype=np.int64),
    )

    # Materialise user-program counts as a compact CSV in a deterministic order.
    rows = []
    for (user_idx, show_id), entry in sorted(counts.items()):
        if entry[0] == 0 and entry[1] == 0 and entry[2] == 0:
            continue
        rows.append(
            {
                "user_idx": user_idx,
                "tv_show_id": show_id,
                "qualifying_airings_primary": entry[0],
                "qualifying_airings_sensitivity": entry[1],
                "positive_airings": entry[2],
                "bridged_qualifying_airings_primary": entry[3],
                "bridged_positive_airings": entry[4],
                "covered_seconds_total": entry[5],
            }
        )
    counts_csv = config.out("interactions", f"{output_prefix}_user_program_counts.csv")
    from .io_utils import write_csv

    write_csv(
        counts_csv,
        [
            "user_idx",
            "tv_show_id",
            "qualifying_airings_primary",
            "qualifying_airings_sensitivity",
            "positive_airings",
            "bridged_qualifying_airings_primary",
            "bridged_positive_airings",
            "covered_seconds_total",
        ],
        rows,
    )
    audit["user_program_rows"] = len(rows)
    audit["user_airing_npz"] = str(out_npz)
    audit["user_program_csv"] = str(counts_csv)
    audit["outputs"] = {
        "user_airing_npz": str(out_npz),
        "user_program_csv": str(counts_csv),
    }
    audit["runtime_seconds"] = round(time.perf_counter() - started, 3)
    cov = (
        np.frombuffer(out_covered, dtype=np.int64).astype(np.float64)
        / np.maximum(1, np.frombuffer(out_duration, dtype=np.int64))
    )
    if len(cov):
        audit["coverage_quantiles"] = {
            "min": float(np.min(cov)),
            "p10": float(np.percentile(cov, 10)),
            "median": float(np.median(cov)),
            "p90": float(np.percentile(cov, 90)),
            "max": float(np.max(cov)),
        }
    write_json(config.out("audit", f"interaction_audit_{window_name}.json"), audit)
    return audit
