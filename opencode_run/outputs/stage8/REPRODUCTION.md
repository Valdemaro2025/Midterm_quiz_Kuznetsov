# Stage 8 reproduction and audit

Run from the project root. All writes remain under `outputs/stage8`; prior source,
predictions, labels, submissions and search logs are read-only.

```sh
python3 -B outputs/stage8/code/study.py prepare
python3 -B outputs/stage8/code/study.py evaluate
python3 -B outputs/stage8/code/checks.py all
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py --junitxml=outputs/stage8/audit/tests_passed.xml
```

Completed splits and diagnostics are checkpointed and skipped on resume. A clean
rebuild should use a fresh workspace containing original Stage 2–7 artifacts,
Stage 8 source/documentation, and the externally supplied diagnostic JSON at its
recorded path. Preserve this completed run before rebuilding. `prepare` freezes
the four configurations before evaluation and records earlier-artifact hashes.

## Input construction

Both raw historical schedules are streamed with nominal start-time ownership.
Only labelled positive-duration airings enter these models. Identity duplicates
are resolved deterministically by `(channel,start,end,program)` with the first
historical file then source-row priority. Training schedule airings and observed
coverage must be fully contained before the training cutoff. Target opportunities
are selected by starts in the target window, including crossing-end airings.

The 12-week split uses train `[2020-03-09,2020-05-04)` and validation
`[2020-05-04,2020-07-27)`. It includes **23,639 target airings from weeks 19–20**
and **121,238 from weeks 21–30**, yielding **7,039 candidate IDs**, 678 of which
would be absent using only the latter file. The same 1,260 users are retained.
The positive-label and >=5-label cohorts contain 1,240 and 1,199 users.

Stage 5's full-history coverage cache supplies actual unioned viewing seconds
for fully contained airings. Clipping observations at these split boundaries
cannot change their intersection with fully contained airings. Airings crossing
cutoffs are excluded from interaction labels/signals. New labels are verified
separately with integer `5*covered >= 4*original_duration` comparisons. The old
10-week labels/candidates and four control predictions reproduce exactly.

## Models and accounting

Four new configurations A–D are declared in `plan.json`; no other model settings
are searched. A changes only the original fusion weights. B adjusts original
repeat_50 by the declared square-root clipped weekly opportunity ratio before
normalization, preserving zero scores for programs without personal repeat
evidence. C uses weekday/weekend by 4-hour start bins and smoothing 20. D first
forms the original hybrid score, then max-normalizes it and C as two sources.
Cold users follow the same training-popularity fallback; repeats remain eligible.

`search_attempts.jsonl` distinguishes the four new configurations from original
control refits and repeated backtest fits. `search_history.json` retains pointers
and hashes for the full 31 earlier configurations plus the four new ones (35).
Model C is evaluated before D on each split. No final refit or submission is made.

## Uncertainty and independent checks

Per-user predictions and AP, three-cohort metrics, fallback counts, runtimes,
prediction changes and paired bootstrap intervals are saved for both splits.
Bootstrap uses 10,000 paired-user resamples, seed 137, percentile 95% intervals,
and the unchanged template user order. Intervals are conditional descriptive,
not adjusted for model selection. Both backtests reuse overlapping historical
data and are not untouched tests. Tiny positive point estimates are not a reason
to upload; horizon matching also does not resolve the earlier local/Kaggle
ordering disagreement.

`checks.py` freshly verifies the external **Codex** review's maps, win/loss/tie
counts, submission diversity/personal-history statistics, drift counts and
fifth-label tie sensitivity for the four controls. Its bootstrap endpoints
reproduce in lexicographic user-ID order with the same seed; the first
template-order diagnostic is preserved separately. This is a resampling-order
reconciliation, not additional model fitting. The external report does not state
its quantile probability levels; newly computed schedule-rate quantiles explicitly
use 10/25/50/75/90 percentiles rather than assuming an undocumented convention.

`SOURCES.md` distinguishes official rendered-page facts, user confirmation,
external-review attribution, actual access in this session, and local label
assumptions. Headerless score columns remain unlabeled. No external implementation
is imported, and no externally reported Kaggle value is called a local execution.

Key outputs: `comparison.json`, `recommendation.json`, `audit/validation.json`,
`diagnostics/external_review_verification.json`, split-specific results and
bootstrap files, and `checkpoints/`.
