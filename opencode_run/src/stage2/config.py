"""Configuration and time parsing for Stage 2 preprocessing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


TS_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S")
EPOCH = datetime(1970, 1, 1)


def parse_datetime(value: str) -> datetime:
    """Parse the formats used by the config, view log, and schedule."""
    value = value.strip()
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp: {value!r}")


def parse_epoch(value: str) -> int:
    return int((parse_datetime(value) - EPOCH).total_seconds())


def format_epoch(epoch: int) -> str:
    return (EPOCH + timedelta(seconds=epoch)).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int

    @property
    def start_iso(self) -> str:
        return format_epoch(self.start)

    @property
    def end_iso(self) -> str:
        return format_epoch(self.end)

    def contains_start(self, epoch: int) -> bool:
        return self.start <= epoch < self.end


@dataclass(frozen=True)
class Stage2Config:
    config_path: Path
    project_root: Path
    data_dir: Path
    output_dir: Path
    view_file: Path
    submission_file: Path
    schedule_files: dict[str, Path]
    windows: dict[str, Window]
    primary_coverage: float
    sensitivity_coverage: float
    merge_gap_primary_seconds: int
    merge_gap_sensitivity_seconds: int
    top_k: int
    tie_break: str
    primary_evaluation_cohort: str
    allow_repeats_initial: bool
    popularity_fallback: str
    schedule_ownership: str
    interaction_airing_rule: str
    test_candidate_rule: str
    raw: dict[str, Any]

    def out(self, *parts: str) -> Path:
        return self.output_dir.joinpath(*parts)

    def ensure_dirs(self) -> None:
        for name in (
            "audit",
            "airings",
            "interactions",
            "labels",
            "candidates",
            "metadata",
            "views",
            "worked",
        ):
            self.out(name).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> Stage2Config:
    path = Path(path).resolve()
    project_root = path.parent.parent
    raw = json.loads(path.read_text(encoding="utf-8"))
    data_dir = project_root / raw["data_dir"]
    windows = {
        name: Window(name, parse_epoch(bounds[0]), parse_epoch(bounds[1]))
        for name, bounds in raw["windows"].items()
    }
    schedule_files = {
        name: data_dir / filename for name, filename in raw["schedule_files"].items()
    }
    return Stage2Config(
        config_path=path,
        project_root=project_root,
        data_dir=data_dir,
        output_dir=project_root / raw["output_dir"],
        view_file=data_dir / raw["view_file"],
        submission_file=data_dir / raw["submission_file"],
        schedule_files=schedule_files,
        windows=windows,
        primary_coverage=float(raw["primary_coverage"]),
        sensitivity_coverage=float(raw["sensitivity_coverage"]),
        merge_gap_primary_seconds=int(raw["merge_gap_primary_seconds"]),
        merge_gap_sensitivity_seconds=int(raw["merge_gap_sensitivity_seconds"]),
        top_k=int(raw["top_k"]),
        tie_break=raw["tie_break"],
        primary_evaluation_cohort=raw["primary_evaluation_cohort"],
        allow_repeats_initial=bool(raw["allow_repeats_initial"]),
        popularity_fallback=raw["popularity_fallback"],
        schedule_ownership=raw["schedule_ownership"],
        interaction_airing_rule=raw["interaction_airing_rule"],
        test_candidate_rule=raw["test_candidate_rule"],
        raw=raw,
    )
