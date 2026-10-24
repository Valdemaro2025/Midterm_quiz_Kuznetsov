"""Schedule loading, boundary ownership, airing tables, and metadata.

Ownership rule
--------------
Each schedule file has a nominal half-open week window.  A row belongs to the
window containing its *start* time; rows whose start is outside that nominal
window are spillover and are dropped from the owned schedule.  Within the
owned schedule, a view may overlap several airings, and all of them are kept.
Cross-file conflicts near boundaries are audited; exact-row deduplication alone
is not assumed to resolve them.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Stage2Config, Window, format_epoch, parse_epoch
from .intervals import overlap_seconds
from .io_utils import write_csv, write_json


SCHEDULE_FIELDS = [
    "source_file",
    "source_row",
    "channel_id",
    "channel_title",
    "start_time",
    "start_epoch",
    "end_epoch",
    "duration",
    "tv_show_id",
    "tv_show_title",
    "tv_show_category",
    "tv_show_genre_1",
    "tv_show_genre_2",
    "tv_show_genre_3",
    "year_of_production",
    "director",
    "actors",
]


@dataclass
class ScheduleRow:
    source_file: str
    source_row: int
    channel_id: str
    channel_title: str
    start: int
    end: int
    duration: int
    show_id: str
    title: str
    category: str
    genre_1: str
    genre_2: str
    genre_3: str
    year: str
    director: str
    actors: str

    @property
    def key(self) -> tuple[str, int, int, str, str]:
        return (
            self.channel_id,
            self.start,
            self.duration,
            self.show_id,
            self.title,
        )

    def as_dict(self, airing_idx: int | None = None) -> dict[str, Any]:
        row = {
            "source_file": self.source_file,
            "source_row": self.source_row,
            "channel_id": self.channel_id,
            "channel_title": self.channel_title,
            "start_time": format_epoch(self.start),
            "start_epoch": self.start,
            "end_epoch": self.end,
            "duration": self.duration,
            "tv_show_id": self.show_id,
            "tv_show_title": self.title,
            "tv_show_category": self.category,
            "tv_show_genre_1": self.genre_1,
            "tv_show_genre_2": self.genre_2,
            "tv_show_genre_3": self.genre_3,
            "year_of_production": self.year,
            "director": self.director,
            "actors": self.actors,
        }
        if airing_idx is not None:
            row["airing_idx"] = airing_idx
        return row


@dataclass
class LoadedSchedule:
    name: str
    window: Window
    owned_rows: list[ScheduleRow]
    spillover_rows: list[ScheduleRow]
    audit: dict[str, Any]


@dataclass
class ChannelIndex:
    starts: list[int]
    ends: list[int]
    airing_indices: list[int]
    max_end_tree: list[int]


@dataclass
class Airings:
    name: str
    rows: list[ScheduleRow]
    by_channel: dict[str, ChannelIndex]
    source_channels: set[str]
    audit: dict[str, Any]


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_schedule_file(
    path: Path,
    name: str,
    window: Window,
    boundary_hours: int = 24,
) -> LoadedSchedule:
    owned: list[ScheduleRow] = []
    spillover: list[ScheduleRow] = []
    boundary_epoch = boundary_hours * 3600
    audit: dict[str, Any] = {
        "file": str(path),
        "window": [window.start_iso, window.end_iso],
        "total_rows": 0,
        "parse_start_failures": 0,
        "invalid_duration": 0,
        "zero_show_id": 0,
        "owned_rows": 0,
        "spillover_before": 0,
        "spillover_after": 0,
        "crossing_window_end": 0,
        "crossing_window_start": 0,
        "owned_labelled_positive_duration": 0,
        "boundary_rows": 0,
    }
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=1):
            audit["total_rows"] += 1
            try:
                start = parse_epoch(raw["start_time"])
            except ValueError:
                audit["parse_start_failures"] += 1
                continue
            duration = _parse_int(raw.get("duration", ""))
            if duration is None or duration <= 0:
                audit["invalid_duration"] += 1
                continue
            end = start + duration
            show_id = (raw.get("tv_show_id") or "").strip()
            if not show_id or show_id == "0":
                audit["zero_show_id"] += 1
            item = ScheduleRow(
                source_file=name,
                source_row=row_number,
                channel_id=(raw.get("channel_id") or "").strip(),
                channel_title=(raw.get("channel_title") or "").strip(),
                start=start,
                end=end,
                duration=duration,
                show_id=show_id,
                title=(raw.get("tv_show_title") or "").strip(),
                category=(raw.get("tv_show_category") or "").strip(),
                genre_1=(raw.get("tv_show_genre_1") or "").strip(),
                genre_2=(raw.get("tv_show_genre_2") or "").strip(),
                genre_3=(raw.get("tv_show_genre_3") or "").strip(),
                year=(raw.get("year_of_production") or "").strip(),
                director=(raw.get("director") or "").strip(),
                actors=(raw.get("actors") or "").strip(),
            )
            if start < window.start:
                audit["spillover_before"] += 1
                if start >= window.start - boundary_epoch:
                    spillover.append(item)
                if end > window.start:
                    audit["crossing_window_start"] += 1
                continue
            if start >= window.end:
                audit["spillover_after"] += 1
                if start < window.end + boundary_epoch:
                    spillover.append(item)
                continue
            audit["owned_rows"] += 1
            if end > window.end:
                audit["crossing_window_end"] += 1
            if show_id and show_id != "0":
                audit["owned_labelled_positive_duration"] += 1
            owned.append(item)
            if start < window.start + boundary_epoch or start >= window.end - boundary_epoch:
                audit["boundary_rows"] += 1
    audit["owned_labelled_fraction"] = (
        audit["owned_labelled_positive_duration"] / audit["owned_rows"]
        if audit["owned_rows"]
        else 0.0
    )
    return LoadedSchedule(name, window, owned, spillover, audit)


def _count_overlaps(
    rows: Iterable[ScheduleRow],
    require_different_show: bool = False,
) -> dict[str, Any]:
    """Count overlapping airings on the same channel via a sweep."""
    by_channel: dict[str, list[ScheduleRow]] = defaultdict(list)
    for row in rows:
        by_channel[row.channel_id].append(row)
    pairs = 0
    conflicting = 0
    example: dict[str, Any] | None = None
    for channel_id, channel_rows in by_channel.items():
        channel_rows.sort(key=lambda r: (r.start, r.end, r.show_id))
        active: list[ScheduleRow] = []
        for row in channel_rows:
            active = [a for a in active if a.end > row.start]
            for other in active:
                if other.show_id == row.show_id and not require_different_show:
                    pairs += 1
                elif other.show_id != row.show_id:
                    pairs += 1
                    conflicting += 1
                    if example is None:
                        example = {
                            "channel_id": channel_id,
                            "a": other.as_dict(),
                            "b": row.as_dict(),
                        }
            active.append(row)
    return {"overlapping_pairs": pairs, "conflicting_show_id_pairs": conflicting, "example": example}


def audit_boundaries(loaded: dict[str, LoadedSchedule]) -> dict[str, Any]:
    """Audit exact duplicates and overlapping airings across adjacent files."""
    result: dict[str, Any] = {"adjacent_boundaries": []}
    order = ["train", "val", "test"]
    for left_name, right_name in zip(order[:-1], order[1:]):
        left = loaded[left_name]
        right = loaded[right_name]
        boundary = left.window.end
        left_rows = [
            r
            for r in left.owned_rows
            if r.start >= boundary - 24 * 3600
        ] + [r for r in left.spillover_rows if r.start >= boundary - 24 * 3600]
        right_rows = [
            r
            for r in right.owned_rows
            if r.start < boundary + 24 * 3600
        ] + [r for r in right.spillover_rows if r.start < boundary + 24 * 3600]
        left_keys = {r.key for r in left_rows}
        right_keys = {r.key for r in right_rows}
        exact_duplicates = len(left_keys & right_keys)
        by_channel_left: dict[str, list[ScheduleRow]] = defaultdict(list)
        by_channel_right: dict[str, list[ScheduleRow]] = defaultdict(list)
        for r in left_rows:
            by_channel_left[r.channel_id].append(r)
        for r in right_rows:
            by_channel_right[r.channel_id].append(r)
        overlapping = 0
        conflicting = 0
        examples = []
        for channel_id in set(by_channel_left) & set(by_channel_right):
            a_rows = by_channel_left[channel_id]
            b_rows = by_channel_right[channel_id]
            for a in a_rows:
                for b in b_rows:
                    ov = overlap_seconds(a.start, a.end, b.start, b.end)
                    if ov > 0:
                        overlapping += 1
                        if a.show_id != b.show_id:
                            conflicting += 1
                            if len(examples) < 3:
                                examples.append(
                                    {
                                        "channel_id": channel_id,
                                        "overlap_seconds": ov,
                                        "left": a.as_dict(),
                                        "right": b.as_dict(),
                                    }
                                )
        result["adjacent_boundaries"].append(
            {
                "left_file": left_name,
                "right_file": right_name,
                "boundary": format_epoch(boundary),
                "left_rows_within_24h": len(left_rows),
                "right_rows_within_24h": len(right_rows),
                "exact_key_duplicates": exact_duplicates,
                "overlapping_pairs": overlapping,
                "conflicting_show_id_pairs": conflicting,
                "examples": examples,
            }
        )
    result["ownership_rule"] = (
        "A row is owned by the nominal half-open window containing its start "
        "time. Rows starting outside that window are spillover and are dropped. "
        "Within a window all overlapping airings are retained for matching; "
        "cross-file boundary conflicts are reported above."
    )
    return result


def _build_max_end_tree(ends: list[int]) -> list[int]:
    """Segment tree over sorted starts holding the maximum end in each range."""
    size = 1
    while size < len(ends):
        size *= 2
    tree = [-1] * (2 * size)
    tree[size : size + len(ends)] = ends
    for node in range(size - 1, 0, -1):
        tree[node] = max(tree[2 * node], tree[2 * node + 1])
    return tree


def build_airings(
    loaded: LoadedSchedule,
    include_crossing_end: bool,
    airing_csv: Path,
) -> Airings:
    """Build deterministic airing indices for one window."""
    selected = []
    excluded = {
        "zero_show_id": 0,
        "nonpositive_duration": 0,
        "crossing_window_end": 0,
        "non_labelled": 0,
    }
    for row in loaded.owned_rows:
        if not row.show_id or row.show_id == "0":
            excluded["zero_show_id"] += 1
            continue
        if row.duration <= 0:
            excluded["nonpositive_duration"] += 1
            continue
        if not include_crossing_end and row.end > loaded.window.end:
            excluded["crossing_window_end"] += 1
            continue
        selected.append(row)
    selected.sort(key=lambda r: (r.channel_id, r.start, r.show_id, r.source_row))
    by_channel: dict[str, ChannelIndex] = {}
    rows_out = []
    for idx, row in enumerate(selected):
        index = by_channel.get(row.channel_id)
        if index is None:
            index = ChannelIndex([], [], [], [])
            by_channel[row.channel_id] = index
        index.starts.append(row.start)
        index.ends.append(row.end)
        index.airing_indices.append(idx)
        rows_out.append(row)
    for index in by_channel.values():
        index.max_end_tree = _build_max_end_tree(index.ends)
    write_csv(
        airing_csv,
        ["airing_idx"] + SCHEDULE_FIELDS,
        (row.as_dict(idx) for idx, row in enumerate(selected)),
    )
    audit = {
        "window": [loaded.window.start_iso, loaded.window.end_iso],
        "include_crossing_end": include_crossing_end,
        "selected_airings": len(selected),
        "channels": len(by_channel),
        "distinct_show_ids": len({r.show_id for r in selected}),
        "excluded": excluded,
        "csv": str(airing_csv),
    }
    source_channels = {row.channel_id for row in loaded.owned_rows}
    return Airings(loaded.name, rows_out, by_channel, source_channels, audit)


def aggregate_metadata(airings: Airings, out_csv: Path) -> dict[str, Any]:
    """Aggregate programme metadata from the schedule rows of one window only."""
    by_show: dict[str, list[ScheduleRow]] = defaultdict(list)
    for row in airings.rows:
        by_show[row.show_id].append(row)
    rows_out = []
    for show_id, rows in sorted(by_show.items(), key=lambda kv: kv[0]):
        title_counts = Counter(r.title for r in rows if r.title)
        category_counts = Counter(r.category for r in rows if r.category)
        year_counts = Counter(r.year for r in rows if r.year)
        director_counts = Counter(r.director for r in rows if r.director)
        genre_counts = Counter()
        actor_counts = Counter()
        for r in rows:
            for genre in (r.genre_1, r.genre_2, r.genre_3):
                if genre:
                    genre_counts[genre] += 1
            if r.actors:
                for actor in r.actors.split(","):
                    actor = actor.strip()
                    if actor:
                        actor_counts[actor] += 1

        def mode(counter: Counter[str]) -> str:
            if not counter:
                return ""
            return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        rows_out.append(
            {
                "tv_show_id": show_id,
                "canonical_title": mode(title_counts),
                "title_variants": len(title_counts),
                "category": mode(category_counts),
                "category_variants": len(category_counts),
                "genres": "|".join(
                    g for g, _ in sorted(genre_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
                "year_of_production": mode(year_counts),
                "director": mode(director_counts),
                "actors": "|".join(
                    a for a, _ in sorted(actor_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
                "airings": len(rows),
                "channels": len({r.channel_id for r in rows}),
            }
        )
    write_csv(
        out_csv,
        [
            "tv_show_id",
            "canonical_title",
            "title_variants",
            "category",
            "category_variants",
            "genres",
            "year_of_production",
            "director",
            "actors",
            "airings",
            "channels",
        ],
        rows_out,
    )
    return {
        "csv": str(out_csv),
        "distinct_show_ids": len(rows_out),
        "show_ids_with_multiple_titles": sum(1 for r in rows_out if r["title_variants"] > 1),
        "show_ids_with_multiple_categories": sum(
            1 for r in rows_out if r["category_variants"] > 1
        ),
    }


def build_all_schedules(config: Stage2Config) -> dict[str, Any]:
    """Build separate interaction and candidate airing tables.

    Interaction tables are conservative: only airings fully contained in the
    window can become training interactions or validation labels.

    Candidate availability is wider: the same rule is used for validation and
    test -- labelled, positive-duration airings starting inside the target
    window, including airings that end beyond it.
    """
    loaded = {
        name: load_schedule_file(path, name, config.windows[name])
        for name, path in config.schedule_files.items()
    }
    boundary_audit = audit_boundaries(loaded)
    schedule_audit: dict[str, Any] = {
        "files": {name: item.audit for name, item in loaded.items()},
        "boundaries": boundary_audit,
        "within_window_overlaps": {},
        "airings": {},
        "candidate_airings": {},
        "metadata": {},
        "rules": {
            "interaction": "labelled, positive-duration airing with start in window and end <= window end",
            "candidate": "labelled, positive-duration airing with start in target window, including crossing end",
            "training_id_scope": "training segments only",
        },
    }

    interaction_airings = {}
    for name in ("train", "val"):
        airings = build_airings(
            loaded[name],
            include_crossing_end=False,
            airing_csv=config.out("airings", f"airings_{name}.csv"),
        )
        interaction_airings[name] = airings
        schedule_audit["airings"][name] = airings.audit
        schedule_audit["within_window_overlaps"][name] = _count_overlaps(
            airings.rows, require_different_show=True
        )

    candidate_airings = {}
    for name in ("val", "test"):
        airings = build_airings(
            loaded[name],
            include_crossing_end=True,
            airing_csv=config.out("airings", f"candidate_airings_{name}.csv"),
        )
        candidate_airings[name] = airings
        schedule_audit["candidate_airings"][name] = airings.audit
        schedule_audit["within_window_overlaps"][f"candidate_{name}"] = _count_overlaps(
            airings.rows, require_different_show=True
        )

    schedule_audit["metadata"]["train"] = aggregate_metadata(
        interaction_airings["train"],
        config.out("metadata", "metadata_train.csv"),
    )
    write_json(config.out("audit", "schedule_audit.json"), schedule_audit)
    return {
        "loaded": loaded,
        "airings": interaction_airings,
        "candidate_airings": candidate_airings,
        "audit": schedule_audit,
    }
