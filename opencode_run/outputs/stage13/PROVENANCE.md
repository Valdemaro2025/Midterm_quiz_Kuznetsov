# Stage 13: bounded refinement of the scored Stage 12 model

## Evidence and assistance

The user confirms Stage 12's returned scores: **Public .44041, Private .42267**.
Stage 12 is the designated best scored submission, with preserved SHA-256
`5128cb80f1fa160b08a26a67539cd3a8059640dac6bb8038b55e43ccbd639aee`.
These observations are recorded only in new Stage 13 files; historical metadata
is not rewritten.

**Codex, an external AI assistant, suggested this follow-up.** Calculations,
implementation and automated verification were performed in this OpenCode
session using project data and code, without external diagnostic-file
dependencies. Personal user analysis or verification is not claimed; personal
verification remains pending. Returned Kaggle feedback motivates the follow-up,
but neither score is an optimization target or independent confirmation.

## Frozen design and component reuse

The declaration and selection rule were saved before evaluation. All 53 prior
configurations and their declarations were audited. The five specified variants
are distinct from prior configurations; in particular, binary EASE beta .20/.30
is not a duplicate of globally scaled log-count EASE at those betas. No slots
were replaced. Five new unique configurations bring the total to **58**.

The control is exact Stage 12: >=80% qualifying-airing log-count input, global
training-only scaling, lambda 1000 and beta .10. The Stage 9 score remains its
exact pre-ranking .70 normalized opportunity-adjusted repeat50 plus .30
normalized association80, with no additional normalization. EASE has no
opportunity adjustment in the control.

Stage 12's verified raw completed weighted-EASE scores and Stage 9 component
scores were reused. No historical EASE model was refitted for this study.
Control lists reproduce exactly on both splits, and their three-cohort metrics
were independently recomputed from saved predictions rather than rounded values.

A–D change beta alone (.05, .15, .20, .30). E retains beta .10 and changes only
EASE opportunity adjustment. It clips completed EASE predictions to zero,
multiplies positive candidate scores by the established square-root clipped
weekly opportunity ratio, then maximum-normalizes across eligible candidates.
Negative learned coefficients remain intact during prediction. Zero EASE
evidence remains zero, and a positive score without a valid historical
opportunity raises an inconsistency. No such inconsistency occurred.

The historical/target opportunity counts were recomputed with the established
ownership, deterministic identity deduplication, eligibility and boundary rules.
Factors reproduce Stage 9's established factors exactly. The 12-week split uses
8 training weeks and 12 target weeks; the original split uses 10 and 10 weeks.
EASE opportunity adjustment is a forecasting heuristic, not calibrated
probability. It was not combined with the alternative betas.

## Result and decision

All four beta-only changes have lower point MAP@5 than Stage 12 on both splits.
This does not prove beta .10 is generally optimal; it is the result of the fixed
declared grid.

Variant **E** is the only qualifying configuration:
- 12-week MAP@5 .41206261022927687, gain .002131613756613756,
  descriptive 95% interval [.0010310350529100522, .003284408068783068].
- Original 10-week MAP@5 .39266490299823636, gain .0025634920634920633,
  interval [.0012804232804232805, .003899537037037036].

It passes the saved >=.002 gain requirement on both splits and the positive
12-week interval lower-bound requirement. There was no selection tie.
Both periods overlap earlier study data and have been repeatedly used. These
10,000-resample, seed-137, template-order intervals are descriptive and
selection-unadjusted, not independent statistical confirmation. Diagnostic
repeat-history partitions did not create or select additional models.

## Authorized final refit

E was frozen and the shared final prediction function reproduced both selected
historical Top-5 lists exactly, reusing the valid historical EASE models. The
selected model was then refitted once on the full 20-week observed history for
the 12-week target. Its log-count normalization scalar was recalculated from
full training data; S9 and EASE opportunity factors were recomputed from those
exact windows. All training users/programs, original >=80% association settings,
fallback and ranking remain unchanged.

The full reconstructed Stage 9 and unadjusted Stage 12 controls both match their
existing submission lists exactly before the new EASE adjustment. Internal
historical-boundary airings are restored and final-cutoff crossings excluded as
in verified Stage 5 preprocessing. The 6,632 test candidates are unchanged.
No future viewing behavior enters any computation.

The only new CSV is
`outputs/stage13/kuznetsov_vladimir_midterm_submission_refined_ease.csv`.
Its digest is in `final/provenance.json`. It differs from Stage 12 for 133 users'
lists and 3 first recommendations, with mean Top-5 overlap 4.951587301587302.
It recommends 190 distinct programs versus 189 for Stage 12.

Stage 12 remains the best **scored** submission. Stage 13 has no returned score
and has not been uploaded. No publication, session-export editing or final
report was performed. All earlier workspace files and raw inputs are checked
byte-for-byte. Verification is automated, not personal user verification.
