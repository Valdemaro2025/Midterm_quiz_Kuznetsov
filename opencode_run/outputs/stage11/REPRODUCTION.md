# Resume and verify the completed Stage 11 study

From the project root, use:

```sh
python3 -B outputs/stage11/code/run.py evaluate
python3 -B outputs/stage11/code/run.py select
python3 -B outputs/stage11/code/audit.py all
```

Completed split result files cause `evaluate` to skip those evaluations. The
saved selection is reused. **Do not rerun the completed diagnosis or overwrite
the frozen declaration.** The existing checkpoint, attempt prefix, declaration
and diagnostic hashes are protected in `resume_state.json`. Unique configuration
names are counted once, not once per split, model fit or interrupted retry.

Relevant implementation and existing tests:

```sh
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py outputs/stage10/code/test_refinement.py outputs/stage11/code/test_ease.py outputs/stage11/code/test_fusion.py --junitxml=outputs/stage11/audit/tests.xml
```

This completed run passed 88 tests. Pytest cache and bytecode writes are disabled
so that prior artifacts remain unchanged. For audit-only operation, run:

```sh
python3 -B outputs/stage11/code/audit.py verify
```

## Inputs and reproducibility

Only original project files and primary-paper mathematics are used. Do not import
an external Codex diagnostic file or an outside implementation. The diagnosis
was completed before EASE implementation; the exact eight lambda/beta pairs and
selection rule were declared before model evaluation. A fresh reproduction must
preserve this completed run in a separate workspace and retain its frozen
declaration, diagnosis, source files and original Stage 2–10 data. There is no
reason to restart the completed study or add a replacement configuration.

The same original Stage 8 12-week and 10-week splits are used. Verified existing
unioned-coverage caches supply pre-cutoff >=80% training interactions; the matrix
includes all qualifying training users and historical items. Stage 9 component
scores are reused without additional normalization. Target eligibility, fallback
and ranking are unchanged. EASE is trained sequentially at lambda 100 and 1000,
reusing each fit for beta .10, .20, .30 and 1.00.

`ease_model.py` contains the float64 matrix, Cholesky inverse, column-wise
coefficient division and completed-score clipping/normalization. `test_ease.py`
uses separate per-column constrained-ridge solutions as a reference, and checks
duplicates, overflow, signs, zero diagonal, mappings and temporal isolation.
`test_fusion.py` checks that S9 is not renormalized, beta zero is exact, missing
EASE history falls back correctly, and subgroup AP attribution is additive.

No candidate passed the declaration's rule, so `audit.py final` records a
no-refit decision and produces no CSV. The conditional refit code is gated by
that selection and never changes the best-scored designation before a returned
score. There is no Stage 11 submission path/hash to report for this outcome.

## Important artifacts

- `declaration.json`, `duplicate_audit.json`, `source_attribution.json`.
- `diagnostics/`: original independent coverage diagnosis, unchanged.
- `resume_state.json`, `attempts.jsonl`, `checkpoints/`: genuine resume evidence.
- `horizon12/`, `original10/`: all predictions, independent per-user AP and
  three-cohort metrics, subgroup gains/losses, fallback, timing and list changes.
- `models/` within each split: binary X, mappings, resource estimates, coefficient
  and solver audits, raw/normalized eligible score caches.
- `paired_bootstrap.json` within each split: all three cohort intervals.
- `comparison.json`, `selection.json`, `final_decision.json`.
- `audit/validation.json`, test XML, `code_provenance.json`, `search_history.json`.
- `preservation_before.json` / `preservation_after.json`: previous files and raw
  inputs, including the immutable Stage 9 CSV.

These program logs are not an AI-session export. No session history is fabricated
or edited. No TLS/certificate setting is changed. No upload, publishing or final
report is performed. See `PROVENANCE.md` for assistance attribution and uncertainty.
