# Stage 12 reproduction and resume

From the project root:

```sh
python3 -B outputs/stage12/code/study.py prepare
python3 -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests outputs/stage6/code/test_study.py outputs/stage8/code/test_study.py outputs/stage9/code/test_refit.py outputs/stage10/code/test_refinement.py outputs/stage11/code/test_ease.py outputs/stage11/code/test_fusion.py outputs/stage12/code/test_counts.py --junitxml=outputs/stage12/audit/tests.xml
python3 -B outputs/stage12/code/study.py evaluate
python3 -B outputs/stage12/code/study.py select
python3 -B outputs/stage12/code/final_refit.py
python3 -B outputs/stage12/code/audit.py
```

For this completed run, completed declarations, split results, model score
caches, per-method completion markers, selection and final-refit records are
reused on resume. Do not remove them to force new experiments. The final CSV
is never overwritten outside the recorded completed-refit resume path. Unique
configuration names are counted once across splits, retries and reproductions.

To verify only, run `audit.py`. Tests and verification do not refit successful
historical configurations. For a clean reproduction, preserve this run first and
use a fresh workspace containing the original Stage 2–11 artifacts/raw inputs
plus the Stage 12 code/documentation. Do not replace the original files.

## Inputs and computation

`study.py prepare` freezes the four requested configurations and selection rule,
audits prior duplicates and hashes every prior workspace file/raw input. All
writes remain under Stage 12. No external diagnostic JSON or outside implementation
is accessed. Existing Stage 11 training-only data loaders, solver, completed-score
normalization and fusion functions are reused read-only.

`count_input.py` reconstructs qualifying counts from unioned coverage caches,
counts distinct training user-airing identities, and checks duplicate
representations. Its transform builds the same learned axes/support as binary
EASE, computes `log1p(n)` in float64 and learns one global training scalar
`sqrt(nnz / sum(log1p(n)**2))`. It never row-normalizes or column-normalizes X.
The zero-positive case is explicit. New Gram validation checks squared column
norms rather than the binary-only column-sum identity. The matrix multiplication
and Cholesky/column-wise coefficient solver remain unchanged.

`evaluate` fits lambdas 100 and 1000 sequentially per split, reusing each fit for
beta .10 and 1.00. Stage 9 and matching binary EASE predictions are reused,
not refitted. Their AP and three-cohort metrics are independently recomputed.
All per-user scores, matched differences, fallback statistics, frequency
distributions, input scalars, timings and relevant-pair diagnostics are saved.
Paired descriptive intervals use 10,000 resamples, seed 137, in template order.

`select` uses only gains versus Stage 9 and the frozen conservative rule, not
matched-binary differences, diagnostic subsets or Kaggle scores. It selected
lambda=1000, beta=.10 by the largest minimum gain across the two splits.

`final_refit.py` uses the same weighted prediction implementation for selected
historical equivalence and final full-history training. Both historical lists
must match exactly before writing the CSV. It recomputes the input scalar from
all full-history counts and rebuilds Stage 9 factors for the exact 20/12-week
windows. The full-history S9 component must reproduce the best Stage 9 CSV.
It checks the 6,632 candidates and complete historical/target schedule identities.

## Key artifacts

- `declaration.json`, `duplicate_audit.json`, `source_attribution.json`.
- `preservation_before.json`, `preservation_after.json`, `code_provenance.json`.
- `horizon12/`, `original10/`: predictions, independent AP/metrics, matched binary
  controls/differences, bootstrap intervals, subgroups, input count audits and model
  score caches.
- `comparison.json`, `selection.json`, `selected_configuration.json`.
- `final_reproduction/`: exact selected historical Top-5 equivalence.
- `final/`: full-history scalar/statistics, parameters, model audits, fallback,
  cold-start, Stage 9 comparison, runtime and submission checks.
- `audit/validation.json`, `audit/tests.xml`, `audit/verification_attempts.json`.
- `attempts.jsonl`, `checkpoints/`, `search_history.json`: 53 unique configurations;
  refits, reused controls and verification retries do not add configurations.

The only submission output is:

`outputs/stage12/kuznetsov_vladimir_midterm_submission_frequency_ease.csv`

No upload, publication, session-export editing or final report is performed.
Stage 9 remains best scored until a better returned score establishes otherwise.
See `PROVENANCE.md` for assistance attribution, normalization-convention limits,
overlapping-history uncertainty and the unconfirmed generalization of local gains.
