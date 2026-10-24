# Stage 14 reproduction and resume

From the project root:

```sh
python3 -B outputs/stage14/code/study.py prepare
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py outputs/stage10/code/test_refinement.py outputs/stage11/code/test_ease.py outputs/stage11/code/test_fusion.py outputs/stage12/code/test_counts.py outputs/stage13/code/test_refinements.py outputs/stage14/code/test_preference_ar.py --junitxml=outputs/stage14/audit/tests.xml
python3 -B outputs/stage14/code/study.py evaluate
python3 -B outputs/stage14/code/study.py select
python3 -B outputs/stage14/code/audit.py all
```

The declaration, completed split results, shared AR-score files, per-method
completion markers and selection are reused on resume. Do not delete them to
force new experiments or count interrupted retries twice. Audit-only operation
uses `audit.py verify`; `audit.py final` records the conditional decision.

For a clean reproduction, preserve this completed run first and use a fresh
workspace with all original Stage 2–13 artifacts/raw inputs and the Stage 14
code/documentation. Every write remains under Stage 14. Bytecode and pytest
cache writes are disabled to preserve prior files. No external diagnostic JSON
or outside-project implementation is accessed.

## Computation

`prepare` inspects and records the current association source conventions, hashes
all earlier workspace files, audits 58 prior configurations and declarations,
then freezes the exact six recommendation configurations and hybrid-only rule
before evaluation.

`preference_ar.py` implements A/Y matrices, full-catalogue fractional memberships,
asymmetric float64 C, confidence and lift with support_Y, original rule limits
and count-ordered personal antecedents. It saves matrices, marginals and complete
retained rules. The original CBF and AR implementations remain unchanged.

`study.py` reuses verified Stage 13 repeat and adjusted log-count EASE components.
The control lists and independently recomputed three-cohort metrics must match
the exact saved Stage 13 values, not rounded user-provided approximations. A
post-adjusts original completed AR scores. B/C learn preferred-consequent rules
at K=5/10 with no AR adjustment. Each model is evaluated individually before
its AR-only replacement hybrid. Standalone uses the same candidate-normalized
AR component and established fallback/ranking.

`select` applies the conservative criteria only to A/B/C hybrids: >=.002 gains
on both splits, positive 12-week interval lower bound, then maximin gain and
lower measured cost. Standalone results and diagnostic familiarity partitions
are not selection candidates/targets. In this run none qualifies. `audit.py
final` therefore records no refit and produces no CSV. The gated conditional
refit implementation would rebuild all full-history statistics and check the
historical predictions first if a hybrid had qualified.

`audit.py verify` separately checks full-catalogue memberships with a partition
reference, A/Y support vectors, C-column sums, all rule thresholds and correct
consequent lift, complete top-50 pruning, original personal antecedent ordering,
aggregation, unchanged repeat/EASE prediction arithmetic, saved AP/MAP and paired
intervals. It does not rerun any completed recommendation experiment.

## Outputs

- `declaration.json`, `association_inspection.json`, `duplicate_audit.json`.
- `score_evidence.json`, `current_best_submission.json`, source attribution.
- `horizon12/`, `original10/`: all six predictions, independent per-user AP,
  three-cohort metrics, fallback, rule counts, runtimes, recommendation changes,
  familiar/unobserved diagnostics and paired bootstrap intervals.
- `AR_models/` within each split: A/B/C raw and normalized scores; B/C A/Y/C
  matrices, full marginals, retained rules and model statistics.
- `comparison.json`, `selection.json`, `final_decision.json`.
- `audit/validation.json`, test XML, code provenance and preservation manifests.
- `attempts.jsonl`, `checkpoints/`, `search_history.json`: 64 unique configurations;
  standalone counts toward budget, shared fits and repeated backtests do not.

No Stage 14 submission exists for this negative outcome. Stage 13 remains the
best scored submission. No upload, publication, session-export editing or final
report is performed. All verification claims are automated, not personal user
verification. See `PROVENANCE.md` for Codex attribution and uncertainty limits.
