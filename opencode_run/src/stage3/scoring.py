"""Deterministic Top-5 construction with a common candidate pool and fallback."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def top_k_from_scores(
    scores: Mapping[int, float],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    k: int = 5,
) -> tuple[list[int], dict[str, int | bool]]:
    """Rank scored candidates, then fill from global popularity.

    All IDs are validated against the common candidate pool. Ties are broken by
    ascending show ID. The result always has exactly ``k`` unique IDs unless the
    pool itself is smaller than ``k``.
    """
    allowed = set(int(x) for x in candidate_ids)
    ordered: list[tuple[float, int]] = []
    for raw_id, raw_score in scores.items():
        show_id = int(raw_id)
        score = float(raw_score)
        if show_id in allowed and score > 0.0:
            ordered.append((score, show_id))
    ordered.sort(key=lambda pair: (-pair[0], pair[1]))

    selected: list[int] = []
    seen: set[int] = set()
    for _, show_id in ordered:
        if show_id not in seen:
            seen.add(show_id)
            selected.append(show_id)
        if len(selected) >= k:
            break

    fallback_used = 0
    for raw_id in popularity_ids:
        if len(selected) >= k:
            break
        show_id = int(raw_id)
        if show_id in allowed and show_id not in seen:
            seen.add(show_id)
            selected.append(show_id)
            fallback_used += 1

    if len(selected) < k:
        for raw_id in sorted(allowed):
            if len(selected) >= k:
                break
            if raw_id not in seen:
                seen.add(raw_id)
                selected.append(raw_id)
                fallback_used += 1

    return selected, {
        "scored_candidates": len(ordered),
        "fallback_filled": fallback_used,
        "used_fallback": fallback_used > 0,
    }


def top_k_from_vector(
    row: Iterable[float],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    k: int = 5,
) -> tuple[list[int], dict[str, int | bool]]:
    scores = {
        show_id: float(score)
        for show_id, score in zip(candidate_ids, row)
        if float(score) > 0.0
    }
    return top_k_from_scores(scores, candidate_ids, popularity_ids, k=k)
