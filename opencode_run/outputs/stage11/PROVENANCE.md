# Stage 11: EASE assistance, implementation and automated verification

## Assistance and sources

An **external AI assistant, Codex**, suggested investigating EASE after discussing
the existing results with the user. That suggestion is not presented as the
user's personal analysis. Implementation, diagnostic calculations, model fits,
evaluation and automated checks were performed in this OpenCode session using
project data. **Personal user verification remains pending** unless the user
actually performs and reports it.

Primary source consulted in this session:

Harald Steck, *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW 2019.
- https://arxiv.org/abs/1905.03375
- https://arxiv.org/html/1905.03375v1
- DOI: https://doi.org/10.1145/3308558.3313710

Equations 4 and 8 provide the inverse and column-wise coefficient division;
Algorithm 1 describes the zero diagonal. Sections 4.2 and 5.2 explain why negative
weights must be retained. The implementation here uses a float64 Cholesky solve
against the identity rather than importing outside model code. A separately
calculated constrained-ridge reference is used in the small-matrix tests.

No external Codex diagnostic JSON or outside-project implementation was read or
used. Earlier project files were hashed for preservation; their external-review
findings were not inputs to these Stage 11 calculations. No AI-session export
was fabricated, reconstructed or edited. `attempts.jsonl` is a genuine local
program-attempt log, not a replacement for the AI-session history.

## Resume state and unchanged declaration

At resumption, the actual checkpoint was `eight_configurations_declared`. Both
diagnoses were complete. All five code files, including `test_fusion.py`, were
present. The attempt log contained four diagnostic events and **no EASE fit or
configuration evaluations**. The reported certificate error had not produced a
recorded model failure. The saved diagnoses and declaration were neither rerun
nor overwritten. Their hashes are checked against `resume_state.json`.

All resumed work ran locally. No certificate settings were changed and TLS
verification was not disabled.

## Fixed model and data conventions

X is binary float64 CSR, from every training user/program with at least one
>=80% actual-coverage airing. Repeated qualifying interactions collapse to one.
Noncohort training users and noncandidate historical programs remain included:

| Split | Learned users | Learned programs | Noncohort users | Noncandidate programs |
|---|---:|---:|---:|---:|
| 12-week | 4,680 | 2,022 | 3,441 | 401 |
| Original 10-week | 4,737 | 2,489 | 3,495 | 623 |

G = X.T @ X is float64. For lambda 100 and 1000, a positive-definite Cholesky
solve obtains P = inverse(G + lambda I). B divides **columns** by -P_jj and sets
its diagonal to zero. Negative coefficients are retained. Prediction completes
X_user @ B before eligible candidate filtering, score clipping and per-user
positive-score max normalization. Previously watched programs stay eligible.
Missing learned items and users without usable history get zero EASE scores.

EASE receives no new schedule adjustment. Fusion is exactly
`(1-beta)*S9 + beta*normalized_positive_EASE`, with **no additional normalization
of S9**. Beta zero reproduces both saved Stage 9 historical prediction lists
exactly. The original popularity fallback and deterministic ranking remain.

Training matrices and mappings come only from pre-cutoff observations. Labels
are evaluation-only. The established 1,260 users, candidate sets, temporal
boundaries and primary labels are unchanged. All previous CBF and association
code and comparison evidence remain preserved.

Conservative pre-allocation estimates were approximately **520 MiB / 658 MiB**,
below the fixed 4,096 MiB budget. Both lambdas fit sequentially on each split;
each fitted model served four beta variants. There were four shared EASE fits,
16 candidate/split evaluations, eight new unique configurations and no actual
model failures. The total preserved unique configuration count is **49**.
Resource limits did not require a substitution or replacement experiment.

## Diagnostics and interpretation

Personal repeat history means >=50% actual-coverage evidence. Global learned-item
history means >=80% evidence from any training user. Pair counts are not distinct
program counts; each is saved separately and distinct counts are not additive
across user-pair groups.

For the 12-week split, 2,075 relevant pairs lack personal repeat50 history, only
51 of which reach Stage 9's Top-5. Of these personally repeat-unobserved pairs,
1,663 have qualifying primary80 evidence from other users. For the original
split the corresponding counts are 1,996, 32 and 1,740. There are 412 / 256
globally primary80-unobserved relevant pairs that EASE cannot learn.

With lambda=100, beta=.10, relevant pairs without personal repeat50 history gain
22 hits and lose 2 on the 12-week split, and gain 11 and lose 4 on the original
split. Pairs with personal repeat50 history gain/lose 50/67 and 47/55. Globally
unobserved relevant-pair hit counts do not change. This illustrates a tradeoff,
not proof of generalization. AP can also improve by reordering existing hits,
so net-hit change and MAP change need not have the same sign.

The fewer-than-20 history subset is defined by distinct eligible programs with
personal >=50% evidence. Its points improve for beta=.10, but the diagnostic
subsets are not used to create models or to select a winner.

## Decision and limitations

**No configuration qualifies under the saved conservative rule.** At beta=.10,
12-week point gains are about .002166 (lambda=100) and .002458 (lambda=1000), but
their intervals include zero and original-split gains are only .001414 and
.001040. Larger EASE weights are worse; standalone EASE substantially
underperforms the frequency/schedule baseline.

No full-history refit or submission CSV is generated. Stage 9 stays the best
scored submission (Private .41656, Public .43185). Returned Kaggle scores are
not optimization targets.

The two historical periods overlap and have been repeatedly used. Paired-user
intervals use 10,000 resamples, seed 137, in template order; they are descriptive
and selection-unadjusted, not independent confirmation. Frequency ties and
fragmentation conventions remain the previously documented local assumptions.
The result supports limited complementary discovery but does not establish a
reliable improvement sufficient for another submission.
