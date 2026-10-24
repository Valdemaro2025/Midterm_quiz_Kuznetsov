# Stage 6 reproduction

All paths and commands are relative to the project root. Prior outputs are read-only.

## Verify the completed study

```sh
python3 outputs/stage6/code/study.py verify
python3 -m pytest -q tests outputs/stage6/code/test_study.py
```

## Rebuild in a fresh workspace

Keep the original Stage 2–5 outputs and Stage 6 source files. Archive the existing
Stage 6 generated artifacts before rebuilding; never replace earlier-stage artifacts.
In a fresh `outputs/stage6` containing only `code/` and this file, run:

```sh
python3 outputs/stage6/code/study.py prepare
python3 outputs/stage6/code/study.py primary
python3 outputs/stage6/code/study.py robustness
python3 outputs/stage6/code/study.py final
python3 outputs/stage6/code/study.py verify
python3 -m pytest -q tests outputs/stage6/code/test_study.py
```

`prepare` persists the 11-configuration predeclaration and hashes every prior
output and prior source file before evaluation. It refuses to replace an existing
plan. Phase completion files allow resuming with `study.py all`. Attempts are
append-only in `search_attempts.jsonl`; reproduction, robustness, and final refit
are distinct from new unique configurations. The twelfth slot stays unused.

The primary split reads the existing Stage 3 user–airing coverage cache and frozen
labels/candidates. The later split reads Stage 5 full-history coverage, retaining
only fully contained training/validation airings. For these airings, intersecting
unioned views yields exactly the same coverage as clipping views to each split;
cross-cutoff airings are excluded. The original candidate-airing cache supplies
the later candidate pool including validation-end-crossing airings. Popularity,
association rules, repeat signals, and labels are rebuilt for the later cutoff.

Coverage is actual unioned `covered_sec / airing_duration`; bridged gap seconds
are never read. Recency is `2**(-(cutoff-airing_end)/(86400*half_life_days))`.
No future observations enter training. Max normalization and ranking/fallback
reuse the evaluated Stage 4 implementation, with previously seen items allowed.

The single selected challenger is refitted from Stage 5's verified full-history
cache. The cache restores 66 airings crossing May 18 and excludes 58 crossing
July 27. The final output is:

`outputs/stage6/kuznetsov_vladimir_midterm_submission_challenger.csv`

See `final/configuration.json` for its SHA-256 and exact configuration,
`final/cold_start.json` for full-history cold-start statistics,
`audit/validation.json` for independent saved-prediction metric checks, and
`outcome.json` for both splits and the upload recommendation. No upload is performed.

The later split overlaps the original holdout: it is a temporal robustness check,
not an independent untouched test. The observed Stage 5 Kaggle values are 0.40974
and 0.42586 with unconfirmed column labels; they are not used in model selection.
