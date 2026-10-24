# Stage 12: frequency-preserving EASE input

## Assistance and convention

An external AI assistant, **Codex**, suggested testing whether preserving viewing
frequency improves EASE. This suggestion is not described as personal analysis
or verification by the human user. The implementation, experiments and automated
checks were performed in this OpenCode session, using project data and code.
Personal user verification remains pending. No external diagnostic files were
read and no outside implementation was imported.

The unchanged solver follows Harald Steck, *Embarrassingly Shallow Autoencoders
for Sparse Data*, WWW 2019, https://arxiv.org/abs/1905.03375, as implemented and
checked in Stage 11. This stage changes only the input representation:

`n_ui = number of distinct >=80% actual-coverage training airings`

`Z_ui = log1p(n_ui)`

`scale = sqrt(nnz(Z) / sum(Z_ui**2)); X = scale * Z`

The global scaling is an **explicit experiment convention**, not a claim about
the paper's standard preprocessing or an optimal normalization. It is fitted
only on training positives across all qualifying users and programs. There is
no target-data or label contribution to this scalar. Empty support has scalar
zero, an empty matrix and zero EASE scores, with no division by zero or solve.

## Frozen design and unchanged method

The four lambda/beta configurations and selection rule were saved before model
evaluation in `declaration.json`. All 49 prior configurations and their
declarations were audited. Binary and log-count inputs are not duplicates even
when lambda/beta match. Four new unique configurations bring the total to **53**;
reproductions, shared fits and the selected final refit are not new configurations.

Both Stage 8 splits, labels, candidates, users and cohorts are unchanged. Stage 9
and matching Stage 11 binary predictions were reused without control-model
refits; their AP and cohort metrics were independently recomputed. A beta-zero
Stage 9 score/ranking check reproduces the saved predictions exactly.

Training axes, float64 arithmetic, Cholesky solver, column-wise inverse division,
zero diagonal, negative learned coefficients, candidate filtering and completed
score clipping/normalization remain the Stage 11 implementation. Weighted query
rows use the same globally scaled log-count X. Previously watched programs stay
eligible. No schedule adjustment is added to EASE, and S9 is not additionally
max-normalized. The fusion is exactly `(1-beta)*S9 + beta*positive_normalized_EASE`.

Distinct repeated airings count separately. Duplicate representations of the
same `(user_idx,airing_idx)` do not. Conflicting duplicate representations are
rejected. Independent checks reconstruct counts from actual unioned viewing
seconds and original airing durations, excluding cross-cutoff airings and
unobserved gaps, and compare them with the verified training count caches.

## Training matrix statistics

| Window | Users | Programs | nnz | Qualifying airings | Scalar |
|---|---:|---:|---:|---:|---:|
| 8-week training for 12-week backtest | 4,680 | 2,022 | 82,809 | 237,136 | 0.8336929887150271 |
| Original 10-week training | 4,737 | 2,489 | 114,754 | 316,314 | 0.8583889535264608 |
| Final full 20-week training | 4,841 | 3,864 | 228,410 | 611,449 | 0.8906002416547741 |

Count histograms/quantiles, matrix dimensions, nnz, squared norms and scalars
are stored under each split's `models/input_statistics.json` and `final/`.
Squared Frobenius norms equal nnz within float64 tolerance. The all-counts-one
case agrees with the corresponding binary matrix within numerical tolerance.

## Findings and conservative selection

Every tested count-input point estimate improves over its matched binary EASE
result on both splits. Standalone EASE improves most, by about .029–.041 MAP@5,
but still substantially underperforms Stage 9. Both beta=.10 fusions pass the
saved practical screening rule versus Stage 9.

The selected candidate is **lambda=1000, beta=.10**, because its smaller
cross-split gain (.00228748) exceeds that of lambda=100 (.00214065). This is the
predeclared maximin rule; the slightly higher 12-week point estimate for
lambda=100 does not override it.

For the selected fusion, gains versus Stage 9 are:
- 12-week: .00377293, descriptive interval [.00111892, .00635188].
- Original 10-week: .00228748, interval [-.00018476, .00478373].

The rule requires a positive interval lower bound only on the 12-week split;
the original-split interval includes zero. The matched count-minus-binary fusion
intervals also include zero. Consequently frequency preservation helped locally
in point estimates, with a large standalone improvement, but the small fusion
advantage is not independently established. These are overlapping, repeatedly
used historical periods with selection-unadjusted descriptive intervals. No
diagnostic partition or returned Kaggle score was used for selection.

Relevant-pair diagnostics retain the >=50% personal-history definition. For the
selected fusion versus Stage 9, without-history relevant hits gain/lose 26/0 on
the 12-week split and 10/4 on the original split; with-history gains/losses are
58/63 and 58/62. AP can improve by reordering existing hits even when net hit
counts decline. These diagnostics do not create or select additional models.

## Authorized final refit and verification

The selected configuration was frozen, then the final prediction implementation
reproduced its saved Top-5 lists exactly on both historical splits. It was
refitted once on [2020-03-09,2020-07-27) for [2020-07-27,2020-10-19). The scalar
was recalculated from all full-history training counts. Stage 9 opportunity
factors, association and fallback statistics were recomputed on those 20/12-week
windows. The full S9 component reproduced the immutable Stage 9 CSV exactly
before adding EASE. Historical boundary restoration, final-cutoff exclusions and
the 6,632-candidate pool were retained. No future viewing behavior was read.

The only new CSV is
`outputs/stage12/kuznetsov_vladimir_midterm_submission_frequency_ease.csv`.
Its digest and actual runtime scopes are in `final/provenance.json` and
`final/runtime.json`. Stage 9 remains the designated best **scored** submission
(Private .41656, Public .43185); the new file has no returned score and was not
uploaded. Earlier files, raw inputs and session exports were not modified.

Automated verification passed 100 tests and 158 substantive audit checks.
One audit-report serialization attempt failed on a NumPy boolean; it was fixed
by converting check values to native bool, without changing models or repeating
fits. This is recorded in `audit/verification_attempts.json`. These checks are
not claimed as personal verification by the user.
