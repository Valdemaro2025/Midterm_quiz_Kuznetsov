# Stage 14: retrospective preference-aware association study

## Assistance, evidence and frozen scope

**Codex, an external AI assistant, suggested this follow-up.** Implementation,
calculations and automated verification were performed in this OpenCode session
using project data/code, without external diagnostic-file dependencies. These
checks are not personal verification by the human user; personal verification
remains pending.

The user confirms Stage 13's returned scores: **Public .44099, Private .42315**.
They are recorded only as observations in new Stage 14 metadata, not optimization
targets. Stage 13 remains the designated best scored submission, with unchanged
SHA-256 `628edd64a9c4ecc26df1f0c2174d7378db4c15ebc4f01615a51f841ff44779d4`.

The original association implementations were inspected before evaluation.
Their source digests and exact ordering conventions are recorded in
`association_inspection.json`. The complete six-configuration declaration,
formulas and hybrid-only selection rule were frozen before evaluation. All 58
previous configurations/declarations were audited. The six recommendation
configurations are new, bringing the total to **64**. Standalone methods count
toward the budget; shared AR fits and repeated backtests do not. No slots were
replaced and no settings or combinations were added.

## Model conventions

The Stage 13 control reproduces exactly on both splits. Although its score is
algebraically .63 R + .27 AR + .10 E, the implementation preserves the evaluated
nested arithmetic `.90*(.70*R + .30*AR) + .10*E` to avoid changing exact-score
ranking through reassociation. Repeat and EASE components, lambda1000, the
training-only count scalar, opportunity factors, target eligibility, fallback
and deterministic ranking are unchanged. There is no final combined-score max
normalization.

A multiplies the original completed maximum-confidence AR candidate scores by
the established opportunity factor, then maximum-normalizes the eligible
positive scores per user. It does not relearn or reprune the original rules.

B/C construct binary float64 A from any >=80% training evidence and fractional
float64 Y from full-training-catalogue Top-K counts at K=5/10. Users with <=K
positive programs give weight 1 to all. Otherwise programs above the Kth count
get 1; the boundary tie shares the remaining membership mass uniformly; below
the boundary gets 0. There is no ID-based preference in this membership rule.
The original ID/column tie ordering for rule truncation and personal antecedent
selection remains unchanged.

`C=A.T@Y` is asymmetric fractional mass. Confidence divides by antecedent A
support; lift uses the **Y consequent marginal**, not the A marginal. C is not
rounded. All statistics use the full training catalogue and the exact original
usable-primary80 training-user population: 4,680 users / 2,022 programs on the
12-week split, and 4,737 / 2,489 on the original split. Y is constructed before
target filtering. Rule filtering then uses candidate eligibility, C>=5,
confidence>=.005, lift>=1, self-rule exclusion and the original 50-rule limit.
Personal antecedents remain count-ordered with the original 50-antecedent cap;
aggregation remains maximum confidence. Missing evidence or denominators
produce no rule. Previously watched candidates remain allowed.

These are **retrospective training-preference associations**, not calibrated
probabilities of future Top-5 membership. B/C do not receive AR opportunity
adjustment; variants were not combined.

## Rule counts

| AR model | 12-week rules | Original 10-week rules |
|---|---:|---:|
| A: original, opportunity-adjusted scores | 31,070 | 51,600 |
| B: preferred K=5 | 6,835 | 10,167 |
| C: preferred K=10 | 12,983 | 19,641 |

Saved matrices, membership statistics, complete pruned rules, raw/normalized
cohort scores and rule counts are under each split's `AR_models/` directory.
Each fitted/adjusted AR signal is shared by its standalone and hybrid results.
Runtime records distinguish shared model work from direct prediction time.

## Results and diagnostics

No hybrid passes the original conservative rule. A improves the 12-week point
MAP by .00325198 with descriptive interval [.00088677, .00562436], but its
original-split point decreases by .00048523. B/C have smaller 12-week gains and
also decrease on the original split. No winner was refitted and no CSV is made.

Preferred standalone components have higher point scores than A here, but A
also differs in opportunity adjustment. This is not a clean causal estimate of
preferred targets versus unadjusted-original standalone AR. In any case, all
three standalone results remain well below the scored hybrid, and standalone
results are diagnostic only, not conditional-refit candidates.

Familiarity means personal >=50% qualifying repeat history. For the B hybrid,
relevant Top-5 hits without that history gain/lose 31/10 on the 12-week split
(net +21), and 19/2 on the original split (net +17). Familiar pairs gain/lose
90/99 and 89/94 (net -9 / -5). For C, unfamiliar net gains are +17 / +13, with
familiar net losses -6 / -4. Gains on unfamiliar items therefore do not establish
a reliable overall improvement; AP also depends on the ordering of familiar
hits. All pair counts, distinct-program counts and AP contributions are saved
separately. Diagnostic partitions were not used for additional models or
selection.

The periods overlap earlier studies and have repeatedly used the same users and
labels. The 10,000 paired-user resamples use seed 137 and template order. The
intervals are descriptive and selection-unadjusted, not independent confirmation.
No extra K values, thresholds, weights, aggregations or searches were run.

## Verification and designation

127 automated tests and 158 audit checks passed. Small-reference tests cover
membership boundaries, fractional support, asymmetric C, the correct Y marginal,
rule and antecedent limits, missing histories, normalization order and temporal
isolation. The audits independently verify saved memberships, support vectors,
C columns, complete rule pruning, user aggregation, prediction scores, AP/MAP
and descriptive intervals. They also verify unchanged repeat/EASE components,
exact control lists, all earlier files/raw inputs and the best submission hash.

Stage 13 stays best scored. No new submission, upload, publication, session-export
editing or final report is performed.
