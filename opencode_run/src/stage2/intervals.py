"""Half-open interval arithmetic used for airing coverage.

All intervals are ``[start, end)`` in integer epoch seconds.  Positive-length
intervals only are meaningful; invalid intervals are ignored by callers or
reported by the pipeline.
"""

from __future__ import annotations

from typing import Iterable

Interval = tuple[int, int]


def overlap_seconds(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the intersection of two half-open intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def merge_intervals(intervals: Iterable[Interval], max_gap: int = 0) -> list[Interval]:
    """Union half-open intervals, optionally bridging a positive gap.

    ``max_gap=0`` merges overlapping and exactly adjacent intervals only.  A
    positive ``max_gap`` is a sensitivity-analysis device; it must not be used
    to claim that the bridged seconds were observed.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    cleaned = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not cleaned:
        return []
    merged: list[list[int]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1] + max_gap:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def covered_seconds(intervals: Iterable[Interval], start: int, end: int) -> int:
    """Actual observed seconds inside ``[start, end)`` without double counting."""
    if end <= start:
        return 0
    clipped = []
    for s, e in intervals:
        s2, e2 = max(int(s), start), min(int(e), end)
        if e2 > s2:
            clipped.append((s2, e2))
    return sum(e - s for s, e in merge_intervals(clipped, max_gap=0))


def covered_seconds_with_gap(
    intervals: Iterable[Interval],
    start: int,
    end: int,
    max_gap: int,
) -> tuple[int, int, int]:
    """Return ``(actual, bridged, bridged_gap)`` for one airing.

    ``actual`` never counts unobserved gaps.  ``bridged`` is the sensitivity
    quantity obtained by merging observed sub-intervals separated by at most
    ``max_gap`` seconds; it therefore includes unobserved gap seconds.  The
    returned gap is ``bridged - actual``.
    """
    if end <= start:
        return 0, 0, 0
    clipped = []
    for s, e in intervals:
        s2, e2 = max(int(s), start), min(int(e), end)
        if e2 > s2:
            clipped.append((s2, e2))
    actual = sum(e - s for s, e in merge_intervals(clipped, max_gap=0))
    bridged_intervals = merge_intervals(clipped, max_gap=max_gap)
    bridged = sum(e - s for s, e in bridged_intervals)
    return actual, bridged, max(0, bridged - actual)


def qualify(covered: int, duration: int, fraction: float) -> bool:
    """Integer-safe ``covered / duration >= fraction`` for a positive duration."""
    if duration <= 0:
        return False
    # Config fractions are simple decimals; compare with a tiny tolerance by
    # cross-multiplying after rounding to parts-per-million.
    ppm = int(round(fraction * 1_000_000))
    return covered * 1_000_000 >= duration * ppm
