# Stage 7: fixed no-decay ablation

This follow-up was **motivated by the returned Kaggle scores**, not preplanned.
Stage 5 stays designated best. Returned score-column labels are unconfirmed.

From the project root:

```sh
python3 -B outputs/stage7/code/ablation.py all
```

The script checks all recorded prior method configurations for an exact match.
Existing exact results are reused if present. It evaluates only the fixed >=20%
coverage-fraction, no-decay, 85/15 repeat/association hybrid. The already evaluated
Stage 6 no-decay individual repeat is a subcomponent reproduction, not a second
new configuration. The two temporal evaluations and full-history refit share one
configuration; no thresholds or weights are selected from their outcomes.

The existing Stage 3 and Stage 6 split labels, cohorts and candidate pools remain
unchanged. Stage 5's verified union-coverage cache is reused for the later training
window and final refit. Fully contained training airings only are aggregated at
each cutoff; bridged gap seconds are never counted. Cutoff-specific >=80%
association rules and popularity are rebuilt with the frozen settings. Previously
seen items stay eligible and max-normalization/ranking/fallback are unchanged.

`all` resumes completed phases without reevaluating them. To verify only:

```sh
python3 -B outputs/stage7/code/ablation.py verify
```

For a clean reproduction, use a fresh workspace containing the original Stage
2–6 artifacts and this Stage 7 code/documentation. Preserve the completed run
before rebuilding. All new outputs remain under `outputs/stage7`.

Key artifacts:
- `kaggle_negative_result.json`: both score pairs, negative result, unconfirmed labels.
- `current_best_submission.json`: Stage 5 designation and original SHA-256.
- `plan.json`, `deduplication.json`, `search_attempts.jsonl`: fixed study and attempts.
- `primary/`, `robustness/`: predictions, per-user AP, three-cohort metrics,
  fallback, runtime, parameters and provenance.
- `final/`: full-history parameters, cold-start statistics and submission checks.
- `kuznetsov_vladimir_midterm_submission_ablation.csv`: separate ablation CSV.
- `audit/validation.json`: independent metrics and byte-preservation checks.
- `outcome.json`: local effects of removing decay and unchanged best designation.

No upload or publishing is performed. The later split is an overlapping temporal
robustness check, not independent untouched evidence. Stage 7 has no Kaggle score.
