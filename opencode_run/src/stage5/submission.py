"""Submission writing and strict format validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SUBMISSION_HEADER = ["user_id", "tv_show_id"]


def write_submission(
    path: Path,
    template_users: Sequence[str],
    predictions: Mapping[str, Sequence[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUBMISSION_HEADER)
        for uid in template_users:
            writer.writerow([uid, " ".join(str(int(x)) for x in predictions[uid])])
    tmp.replace(path)


def validate_submission(
    path: Path,
    template_users: Sequence[str],
    candidate_ids: Iterable[int],
) -> dict:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    with path.open("rb") as handle:
        raw = handle.read()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    check("header_exact", lines[0] == "user_id,tv_show_id" if lines else False, lines[:1])
    check("no_bom", not raw.startswith(b"\xef\xbb\xbf"))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    check("two_columns", all(len(row) == 2 for row in rows), [len(r) for r in rows if len(r) != 2][:10])
    header = rows[0] if rows else []
    data_rows = rows[1:]
    check("exact_header_fields", header == SUBMISSION_HEADER, header)
    check("row_count_1260", len(data_rows) == len(template_users) == 1260, len(data_rows))
    users = [row[0] for row in data_rows]
    check("unique_users", len(users) == len(set(users)), len(users) - len(set(users)))
    check("template_order_exact", users == list(template_users))
    check("no_missing_user", all(uid.strip() for uid in users))
    check("user_ids_string_exact", all(uid == expected for uid, expected in zip(users, template_users)))
    allowed = {int(x) for x in candidate_ids}
    all_five = True
    all_distinct = True
    all_positive = True
    all_candidate = True
    malformed: list[dict] = []
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != 2 or not row[1].strip():
            all_five = False
            malformed.append({"row": row_number, "value": row[1] if len(row) > 1 else None})
            continue
        tokens = row[1].split(" ")
        try:
            ids = [int(token) for token in tokens]
        except ValueError:
            all_five = False
            malformed.append({"row": row_number, "value": row[1]})
            continue
        all_five = all_five and len(tokens) == 5
        all_distinct = all_distinct and len(ids) == len(set(ids))
        all_positive = all_positive and all(value > 0 for value in ids)
        all_candidate = all_candidate and all(value in allowed for value in ids)
        if len(tokens) != 5 or len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
            if len(malformed) < 10:
                malformed.append({"row": row_number, "value": row[1]})
    check("five_tokens_each", all_five, malformed)
    check("distinct_ids_each", all_distinct)
    check("positive_integer_ids", all_positive)
    check("candidate_pool_membership", all_candidate)
    check("no_extra_index_column", len(header) == 2 and header[0] == "user_id")
    return {
        "all_passed": all(item["ok"] for item in checks),
        "checks_passed": sum(1 for item in checks if item["ok"]),
        "checks_total": len(checks),
        "checks": checks,
        "row_count": len(data_rows),
        "header": header,
    }
