# Stage 10 bounded refinement reproduction

From the project root:

```sh
python3 -B outputs/stage10/code/study.py prepare
python3 -B outputs/stage10/code/study.py evaluate
python3 -B outputs/stage10/code/study.py select
python3 -B outputs/stage10/code/study.py final
python3 -B outputs/stage10/code/study.py verify
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py outputs/stage10/code/test_refinement.py --junitxml=outputs/stage10/audit/tests.xml
```

`prepare` snapshots all previous workspace files/raw inputs, records confirmed
score evidence and the best-scored designation, checks exact prior duplicates,
and freezes all six configurations and the conservative selection rule before
evaluation. Every model attempt is logged. Completed phases resume without
repeating evaluations; `all` runs the five script phases. For a fresh rebuild,
preserve the completed run and use a fresh workspace containing the original
Stage 2–9 files plus Stage 10 code and documentation.

The existing Stage 8 preprocessing is reused. One association fit and repeat map
serve all candidates within each split. No new labels, thresholds, clipping
bounds, alpha values, weights or models are introduced. The control reproduces
Stage 9/B's saved historical lists exactly. Repeated backtest fits and any exact
duplicate reuse are explicitly separated from new unique configurations.

`select` uses only local saved AP/MAP and intervals: >=0.002 gain on each split,
positive lower bound on the 12-week paired 95% interval, then largest minimum
gain, fewer independent changed parameters, and lower actual end-to-end cost on
fresh exact reproduction fits if needed. Returned Kaggle scores are not targets.

`final` generates at most one CSV only when a candidate passes the rule. In this
completed study none passed, so the command records a no-refit decision and does
not create a CSV. If a qualifying run existed, the same prediction implementation
would first reproduce the selected historical lists, then refit only that fixed
candidate on 20 historical weeks for the 12-week target and validate all 6,632
eligible candidates against the immutable template.

Main artifacts:
- `declaration.json`, `duplicate_audit.json`, `attempts.jsonl`, `search_history.json`.
- `score_evidence.json`, `current_best_submission.json`, `PROVENANCE.md`.
- `horizon12/` and `original10/`: predictions, independent AP, three-cohort
  metrics, factors, runtime, fallback, descriptive intervals and list changes.
- `comparison.json`, `selection.json`, `final_decision.json`.
- `preservation_before.json`, `preservation_after.json`, `code_provenance.json`.
- `audit/validation.json`, test XML, and progress checkpoints.

All writes stay under Stage 10; bytecode and pytest cache writes are disabled.
No upload, publication or final report is performed. Stage 9 remains the best
scored submission. See the provenance note for overlapping-history and
selection-unadjusted uncertainty caveats.
