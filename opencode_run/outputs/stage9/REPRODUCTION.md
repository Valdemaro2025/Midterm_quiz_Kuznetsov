# Fixed Stage 9 refit reproduction

Run from the project root with the preserved Stage 2–8 artifacts and raw inputs:

```sh
python3 -B outputs/stage9/code/refit.py prepare
python3 -B outputs/stage9/code/refit.py reproduce
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py --junitxml=outputs/stage9/audit/tests.xml
python3 -B outputs/stage9/code/refit.py refit
python3 -B outputs/stage9/code/refit.py verify
```

`prepare` freezes B and snapshots every previous workspace file, including raw
data and all submission copies. The two reproduction fits and final fit use the
same prediction function, not a separate approximation. Both saved Stage 8
top-five lists must match exactly before the final CSV may be written.

Completed phases resume without reevaluating models. `all` runs prepare,
reproduction, fixed refit and verification; tests are the separate command above.
To verify an existing completed run, use only `verify`. The script refuses to
overwrite an existing final CSV outside the recorded completed-refit resume path.

For a fresh reproduction, preserve this completed run and use a fresh workspace
containing all original Stage 2–8 files plus Stage 9 code/documentation. There is
no new configuration search. The only submission output is:

`outputs/stage9/kuznetsov_vladimir_midterm_submission_schedule.csv`

Important artifacts:
- `configuration.json`: exact frozen settings and timestamp-derived windows.
- `reproduction/equivalence.json`: both split equivalence and independent metrics.
- `final/opportunity_counts_and_factors.json`: per-program counts, rates and factors.
- `final/opportunity_summary.json`: count totals, zero-history cases and clipping summary.
- `final/model_provenance.json`, `code_provenance.json`, `PROVENANCE.md`: lineage and attribution.
- `final/predictions.json`, `final/fallback.json`, `final/cold_start.json`.
- `comparison_with_stage5.json`: changed lists/first items, overlap and program diversity.
- `final/runtime.json`: actual directly timed scopes.
- `audit/submission_checks.json`, `audit/validation.json`: format and substantive checks.
- `attempts.jsonl`, `checkpoints/`: fixed refit/reproduction attempts and progress.

No upload or publishing occurs. Stage 5 remains the best scored submission, and
the unique configuration count remains 35.
