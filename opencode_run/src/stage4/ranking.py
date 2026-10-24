"""Stage 4 ranking with an explicit popularity tie-break."""

from __future__ import annotations

from typing import Mapping, Sequence


def top_k(
    scores: Mapping[int, float],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
    k: int = 5,
    tie_break: str = "popularity",
) -> tuple[list[int], dict[str, int | bool]]:
    allowed = set(int(x) for x in candidate_ids)
    ranked: list[tuple[float, int, int, int]] = []
    for raw_id, raw_score in scores.items():
        show_id = int(raw_id)
        score = float(raw_score)
        if show_id not in allowed or score <= 0.0:
            continue
        users, airings = popularity_stats.get(show_id, (0, 0))
        ranked.append((score, users, airings, show_id))

    if tie_break == "popularity":
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    elif tie_break == "id":
        ranked.sort(key=lambda row: (-row[0], row[3]))
    else:
        raise ValueError(f"Unknown tie break: {tie_break}")

    selected: list[int] = []
    seen: set[int] = set()
    for _, _, _, show_id in ranked:
        if show_id not in seen:
            seen.add(show_id)
            selected.append(show_id)
        if len(selected) >= k:
            break

    fallback = 0
    for raw_id in popularity_ids:
        if len(selected) >= k:
            break
        show_id = int(raw_id)
        if show_id in allowed and show_id not in seen:
            seen.add(show_id)
            selected.append(show_id)
            fallback += 1
    if len(selected) < k:
        for show_id in sorted(allowed):
            if len(selected) >= k:
                break
            if show_id not in seen:
                seen.add(show_id)
                selected.append(show_id)
                fallback += 1
    return selected, {
        "scored_candidates": len(ranked),
        "fallback_filled": fallback,
        "used_fallback": fallback > 0,
    }


def top_k_from_vector(
    row: Sequence[float],
    candidate_ids: Sequence[int],
    popularity_ids: Sequence[int],
    popularity_stats: Mapping[int, tuple[int, int]],
    k: int = 5,
    tie_break: str = "popularity",
) -> tuple[list[int], dict[str, int | bool]]:
    scores = {
        show_id: float(score)
        for show_id, score in zip(candidate_ids, row)
        if float(score) > 0.0
    }
    return top_k(scores, candidate_ids, popularity_ids, popularity_stats, k, tie_break)
