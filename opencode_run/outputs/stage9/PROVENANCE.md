# Stage 9 provenance and interpretation

This is the fixed full-history refit of Stage 8 **B_schedule_adjusted_hybrid70**,
not a new improvement study. No threshold, clipping bound, smoothing, fusion
weight, fallback or ranking setting is searched or changed. The unique
configuration history stays **35**; reproductions and this final refit add zero.

## Source attribution correction

The previously rendered official Kaggle data-page verification is attributed to
the **external Codex review**, not to personal verification by the human user.
The external diagnostic JSON records its rendered accessibility-tree inspection
on 2026-10-23. It reports >=80% screen-time coverage, frequency-based selection
of exactly five programs, a 12-week forecast, and a 50/50 public/private split by
users. This OpenCode session does not claim a rendered-browser verification.
Earlier provenance files are preserved unchanged; this note clarifies the
attribution for Stage 9.

Officially incompletely specified details remain local assumptions: actual union
of observed viewing intervals divided by original airing duration, no unobserved
gap seconds, and ascending program-ID label tie-breaking. Local users with fewer
than five qualifying labels retain fewer labels during backtests. The returned
Kaggle score-column identities remain **unconfirmed** because their headers were
cropped; the official partition sizes do not resolve their displayed order.

## Computation and data lineage

The same final prediction function is applied to both Stage 8 historical splits
and to full history. It reproduces the saved B top-five lists exactly, and MAP@5
is independently recomputed from saved predictions. The source implementation
uses the previously evaluated association fitting, independent component max
normalization, 70/30 fusion and deterministic ranking/fallback code. The new
opportunity factors are computed from full-history/test schedule counts; neither
historical validation factor table is reused for final predictions.

Historical opportunities obey nominal file ownership, deterministic airing
identity deduplication and positive-labelled eligibility. Training airings start
inside [2020-03-09, 2020-07-27) and must end by the cutoff. This retains the 66
restored May 18 internal-boundary airings and excludes 58 final-boundary
crossings. Target airings start in [2020-07-27, 2020-10-19), including end
crossings. The historical window is 20 weeks and the target is 12 weeks,
calculated from timestamp differences. Opportunity identities and candidates are
checked against Stage 5 before generating the CSV. Any inconsistency stops the
run rather than silently changing the protocol.

Stage 5 full-history union-coverage and program-count caches are reused. Actual
>=50% and >=80% counts are separately verified from the airing-level coverage
cache. Association and popularity statistics are fitted from that full observed
history only. No target viewing behavior is read. Only the target schedule is
used for target opportunities/candidate eligibility.

`preservation_before.json` / `preservation_after.json` hash **all existing
workspace files outside Stage 9**, including raw inputs, root-level submission
copies and earlier artifacts. `code_provenance.json` records source digests.
`configuration.json` and the fixed Stage 8 parameter-file digest identify the
configuration. Exact timing scopes are in `final/runtime.json`; schedule parsing,
cache I/O and direct fit/prediction are distinguished.

## Unresolved limitations and designation

The local backtests overlap earlier study data, are subject to model-selection
uncertainty, and do not independently establish Kaggle generalization. Previous
local improvements did not always transfer to the returned Kaggle scores. Stage
9 has no returned Kaggle score. Stage 5 remains the designated best scored
submission until a better returned score is confirmed. This stage performs no
upload, publishing or final-report generation.
