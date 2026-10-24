# Stage 13 reproduction and resume

From the project root:

```sh
python3 -B outputs/stage13/code/study.py prepare
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py outputs/stage10/code/test_refinement.py outputs/stage11/code/test_ease.py outputs/stage11/code/test_fusion.py outputs/stage12/code/test_counts.py outputs/stage13/code/test_refinements.py --junitxml=outputs/stage13/audit/tests.xml
python3 -B outputs/stage13/code/study.py evaluate
python3 -B outputs/stage13/code/study.py select
python3 -B outputs/stage13/code/final_refit.py
python3 -B outputs/stage13/code/audit.py
```

Completed declaration, split result files, per-method completion markers,
selection, historical reproduction records and final refit records are reused
on resume. Do not remove completed results to expand the search or count retries
as new configurations. The final CSV is not overwritten. Audit-only operation
uses `audit.py`. For a clean reproduction, preserve the completed run first and
use a fresh workspace containing the original Stage 2–12 artifacts and raw
inputs plus Stage 13 code/documentation.

`prepare` records the best-scored Stage 12 evidence, snapshots every previous
workspace file/raw input, audits exact duplicates and freezes all five variants
and the conservative rule before evaluation. All writes stay under Stage 13;
bytecode and pytest cache writes are disabled.

## Computation

`study.py` reuses the read-only Stage 12 lambda1000 completed EASE score caches
and exact Stage 9 pre-ranking scores. It verifies the unchanged training-only
log-count scalar and exact control predictions on both frozen Stage 8 splits.
No historical EASE fitting is repeated. New opportunity counts are constructed
from both historical schedule files with the established rules; factors match
the verified Stage 9 factors. No external diagnostic file is used.

The shared `predict`/`component` functions implement both historical and final
prediction. A–D do not use EASE opportunity factors. E performs:

`completed raw score -> clip at zero -> multiply opportunity factor -> candidate-only per-user max normalization`

It never changes learned coefficients, fabricates repeat/EASE evidence, or
renormalizes S9. Positive EASE evidence without historical opportunities is an
error. The fixed input, lambda, count normalization, candidate pool, fallbacks,
seen-item policy and deterministic ranking are retained.

`select` applies only the declared gains versus Stage 12 and 12-week interval
screen, then maximin gain and lower measured cost. Variant E is the only winner.
`final_refit.py` freezes E, checks exact historical prediction equivalence using
the same prediction functions, then recomputes the weighted matrix, scalar,
EASE coefficients, S9 statistics and all opportunity factors for the full
20-week training / 12-week target. It first reproduces the existing full Stage
9 and Stage 12 controls exactly and checks all 6,632 target candidates.

The only output CSV is:

`outputs/stage13/kuznetsov_vladimir_midterm_submission_refined_ease.csv`

## Artifacts

- `declaration.json`, `duplicate_audit.json`, `score_evidence.json`, source attribution.
- `horizon12/`, `original10/`: predictions, independent per-user AP, three-cohort
  metrics, fallback, timing, recommendation changes, subgroup diagnostics,
  opportunity counts and paired descriptive intervals.
- `comparison.json`, `selection.json`, `selected_configuration.json`.
- `final_reproduction/`: exact selected historical prediction checks.
- `final/`: full training scalar/count audit, opportunity counts/factors,
  model audits, runtime scopes, submission checks, Stage 12 comparison and digest.
- `audit/validation.json`, test XML and test execution metadata.
- `preservation_before.json`, `preservation_after.json`, `code_provenance.json`.
- `attempts.jsonl`, `checkpoints/`, `search_history.json`: 58 unique configurations;
  reused models, control checks, reproductions and final refit add no configurations.

The code follows the frozen conditional refit authorization. There is no upload,
publication, session-export editing or final report. Stage 12 remains the best
scored submission until a better returned score is confirmed. See
`PROVENANCE.md` for Codex attribution and overlapping-history uncertainty.
