"""AP@5 / MAP@5 evaluation with explicit edge-case conventions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def ap_at_k(
    predictions: Sequence[int],
    relevant: Iterable[int],
    k: int = 5,
) -> float:
    """Average precision at K for one user.

    Conventions:
    - ``relevant`` is a set; duplicates in it do not increase ``m``.
    - A duplicate prediction is kept in its rank position but counts as a
      miss after its first occurrence.
    - If ``m == 0``, AP is 0 by definition.
    - If fewer than K predictions are supplied, missing positions are misses;
      the denominator remains ``min(m, K)``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = {int(x) for x in relevant}
    m = len(relevant_set)
    if m == 0:
        return 0.0
    seen: set[int] = set()
    hits = 0
    score = 0.0
    for rank, raw_item in enumerate(list(predictions)[:k], start=1):
        item = int(raw_item)
        if item in seen:
            is_relevant = False
        else:
            is_relevant = item in relevant_set
        seen.add(item)
        if is_relevant:
            hits += 1
            score += hits / rank
    return score / min(m, k)


def map_at_k(
    predictions_by_user: Mapping[str, Sequence[int]],
    labels_by_user: Mapping[str, Iterable[int]],
    users: Iterable[str] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """MAP@K with cohort breakdowns.

    The primary AP is zero for a user with no positives.  Results are also
    reported separately for users with >=1 positive and users with >=K positives.
    """
    user_list = list(users) if users is not None else sorted(labels_by_user)
    aps: dict[str, float] = {}
    with_positive: list[float] = []
    with_ge_k: list[float] = []
    no_positive = 0
    for user in user_list:
        relevant = labels_by_user.get(user, [])
        m = len(set(int(x) for x in relevant))
        ap = ap_at_k(predictions_by_user.get(user, []), relevant, k=k)
        aps[user] = ap
        if m == 0:
            no_positive += 1
        else:
            with_positive.append(ap)
            if m >= k:
                with_ge_k.append(ap)
    total = len(user_list)
    return {
        "k": k,
        "cohort_size": total,
        "map_at_k": sum(aps.values()) / total if total else 0.0,
        "users_with_no_positive": no_positive,
        "map_with_positive": (
            sum(with_positive) / len(with_positive) if with_positive else 0.0
        ),
        "cohort_with_positive_size": len(with_positive),
        "map_with_at_least_k_positive": (
            sum(with_ge_k) / len(with_ge_k) if with_ge_k else 0.0
        ),
        "cohort_with_at_least_k_positive_size": len(with_ge_k),
        "aps": aps,
    }


def label_size_cohorts(labels_by_user: Mapping[str, Iterable[int]]) -> dict[str, int]:
    sizes = [len(set(int(x) for x in labels)) for labels in labels_by_user.values()]
    return {
        "total": len(sizes),
        "zero": sum(1 for n in sizes if n == 0),
        "one_to_four": sum(1 for n in sizes if 0 < n < 5),
        "at_least_five": sum(1 for n in sizes if n >= 5),
        "positive": sum(1 for n in sizes if n > 0),
    }
