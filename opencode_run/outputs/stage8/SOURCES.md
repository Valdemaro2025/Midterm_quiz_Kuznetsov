# Stage 8 source and protocol clarification

## Official competition description

URL: https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data

The user confirms that the rendered official page specifies:
- >=80% screen-time coverage;
- frequency-based selection of **exactly five** preferred programs;
- a 12-week forecast (weeks 31–42);
- public/private partitions **by users, 50/50**.

The external Codex review independently reports reading that rendered page on
2026-10-23. In this OpenCode session, a direct web fetch returned only the page
shell/title; no rendered-browser execution is claimed here. The facts above are
attributed to the user's rendered-page confirmation, corroborated by that review.

The description does **not** fully specify fragmentation aggregation or ties
between equal program frequencies. Actual unioned observed seconds divided by
original airing duration, excluding unobserved gaps, and ascending program-ID
label tie-breaking remain documented **local implementation assumptions**.
They are not newly verified official details. Local users with fewer than five
qualifying items retain fewer labels; zero/positive/>=5 cohorts remain explicit.
Official selection of exactly five does not establish a local padding rule.

The 50/50 partition fact does not identify which of the previously displayed,
headerless score columns is Public or Private. Their labels remain unconfirmed.

## External diagnostic review

Author/source: **external Codex review**, not an OpenCode execution or session export.

Path:
`/Users/vladimirkuznecov/Documents/Codex/2026-10-21/new-chat-5/outputs/sweettv_independent_review_20261023.json`

Its original computations are attributed to that external review. The Stage 8
`diagnostics/external_review_verification.json` records fresh, separately executed
checks against saved local predictions, coverage caches, schedules and submissions.
External-reported Kaggle values cannot be independently verified from local files;
the Stage 7 values in that review are retained as external reports, not new results
executed or uploaded in this session. No external implementation is imported.

## Interpretation

The new train [2020-03-09, 2020-05-04), validation [2020-05-04, 2020-07-27)
split has 8 training weeks and **12 validation weeks**. Its candidates include
weeks 19–20 from the first historical schedule and weeks 21–30 from the second.
It is a retrospective backtest on overlapping history, not an independent
untouched test. The old 10-week labels and all earlier metrics remain unchanged.

Paired user bootstrap intervals are conditional descriptive uncertainty, not
model-selection-adjusted confidence statements or independent Kaggle validation.
The four configurations are fixed before their Stage 8 evaluation; this stage
itself is informed by prior diagnostics and Kaggle feedback. Four comparisons,
reused users, and overlapping periods create model-selection uncertainty. A tiny
positive estimate alone is insufficient evidence for final refitting or upload.
