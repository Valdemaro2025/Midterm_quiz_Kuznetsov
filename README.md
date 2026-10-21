# SWEET.TV: reproducible TV programme recommendation

Individual LLM4Rec midterm project by Vladimir Kuznetsov.

Repository supplied by the student: https://github.com/Valdemaro2025/Midterm_quiz_Kuznetsov . Code, results and the exact final CSV have been published to the main branch. Input: the official Kaggle SWEET.TV competition archive. No external competition solution code or additional viewing data was used.

## Status

Preprocessing, three required-family algorithms (content filtering, implicit matrix factorization, singleton association rules), baselines, model comparison and submission generation are implemented. The current CSV has **not yet been submitted to Kaggle**; local MAP@5 is not a Kaggle public/private score. The final PDF must later include the actual score screenshot and repository URL. A byte-identical snapshot of the native Codex JSONL session is supplied separately from the public repository. Result JSON files are not a substitute for that log; refresh the native snapshot after further work.

## Install and reproduce

Use Python 3.11+ in a virtual environment. Install `requirements.txt`. Unpack the five official CSVs into a `data/` directory. The raw data remains local and is excluded from Git.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py --data-dir data --cache-dir cache
python test_preparation.py
python test_models.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiments.py --cache-dir cache --output-dir results
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python improve_strategy.py --cache-dir cache --output-dir results
python verify_join.py --data-dir data --cache-dir cache --output results/join_verification.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python make_submission.py --cache-dir cache --results-dir results --output kuznetsov_vladimir_midterm_submission.csv
python summarize_results.py --cache-dir cache --results-dir results
```

The execution host already supplied the dependencies in two Python installations: preprocessing used Python 3.12.14 / NumPy 2.3.5 / pandas 2.2.3; models used Python 3.11.9 / NumPy 2.4.6 / SciPy 1.17.1 / scikit-learn 1.9.0. The commands above also work in one environment containing all listed dependencies. Exact host versions and timings are saved in the result JSONs. No package installation was needed on the author's execution host.

## Data and join

- Parse `user_id` as unsigned 64-bit integers internally, and preserve its original decimal string in the submitted CSV. IDs exceed the signed 64-bit range; float conversion would corrupt them.
- `vsetv_id` equals schedule `channel_id`. Treat times as the supplied local wall time, without inventing a timezone conversion.
- Use half-open intervals `[start, stop)`. Merge overlapping or adjacent user/channel view intervals, then intersect with each labelled programme airing.
- Compute programme end as `schedule.start_time + duration`. The misspelled view-log field `duraton` is audited, but overlap uses the actual start/stop timestamps.
- ID 0 is unlabelled airtime, never a recommendation item. Nonpositive programme durations are discarded.
- At the 18 May and 27 July boundaries, the schedule file for the later period owns airings STARTING in that period. Spillover from the previous file is excluded. Broadcasts crossing a validation/training boundary are excluded from both training and validation labels.
- Sum distinct watched seconds per user/airing. Main positive interactions and validation labels require coverage >=80%. The controlled improvement also tests >=20% and >=50% partial-view evidence for training ONLY, weighted by coverage. The validation target always retains the 80% definition.

## Evaluation

Train: airings ending at or before 2020-05-18 00:00. Validate: airings starting at or after that time and ending by 2020-07-27 00:00. Only the named validation schedule supplies candidates; the final run uses the labelled programme IDs in the test schedule. Recommending previously watched programmes is allowed in every method because repeated viewing defines the target.

Target: the five distinct programme IDs with the largest counts of high-coverage airings, numeric ID ascending on frequency ties. AP@5 uses `sum(P@rank * relevant_hit)/min(5, number_of_relevant_items)`. Primary comparison uses template users with at least five distinct validation positives; all-positive and all-template means are also disclosed, including zero-positive users in the latter.

Every method ranks positive scores, breaks exact ties by numeric programme ID, and fills missing slots using the same popularity ranking (28-day half-life). Exactly five distinct positive candidate IDs are required. `no_high_coverage_history_users` describes lack of >=80% training evidence; the partial-view variant may still have additional evidence for those users.

All model fitting uses pre-cutoff behaviour. Content feature vocabulary and IDF are learned from pre-cutoff programme metadata only. Future schedule attributes do not enter content features. Future schedule IDs supply candidate availability; an explicitly reported schedule-exposure experiment uses counts of future airings, never future viewers.

The holdout is used for model/parameter selection. Its best score is a **selection estimate**, not an independent final test and not a causal business impact estimate. The unseen Kaggle score provides the external check. All trials, including poor variants, remain in `results/validation_results.json`.

## Methods

- **Popularity:** total high-coverage viewing counts with a 28-day half-life.
- **Repeat history:** individual programme counts, compared with 0/14/28/56-day half-lives; later a partial-view coverage-weighted variant.
- **Content:** L2-normalized field-prefixed TF-IDF metadata and a normalized weighted user profile; cosine ranking. Fields: title, category, three genres, production year, director and actors.
- **Implicit ALS:** native NumPy implementation of sparse exact alternating normal equations for the Hu-Koren-Volinsky implicit objective. `p_ui=1[r_ui>0]`, `c_ui=1+alpha*log(1+r_ui)`. Unobserved pairs have preference 0 and confidence 1. Factors 32/64, alpha 10/40, regularization 1, 12 iterations, seed 42.
- **Association rules:** singleton rules `i -> j`; user support at least 3; smoothed confidence `n(i,j)/(n(i)+10)`. Histories are whole training-window baskets, not session sequences. Self-rules are removed. Aggregate using normalized decayed user history.
- **Hybrid:** convex combination after each component's nonnegative scores are divided by their per-user maximum. Weights are selected on validation, not interpreted as probabilities.

## Evidence and disclosure

`results/` contains full comparisons, per-user predictions, two examples selected by training activity, the overlap audit and a SHA-256 manifest of the exact submission. Test scripts check the interval join against a brute-force oracle, a hand-calculable AP example, deterministic ties/fallback, temporal filtering and sparse ALS against dense normal equations. `verify_join.py` independently reconstructs watched seconds for 64 sampled examples directly from the original CSV.

These checks were performed by the AI assistant and executed tools. They do not establish that the student personally performed them. The student should independently inspect at least one overlap calculation and one AP@5 example and explain the findings in their own words before submitting the report. No online A/B test, revenue effect or enjoyment measurement is claimed.

## References consulted

1. Kaggle, *sweet.tv - TV program recommender*, 2020, official description, target construction, data and evaluation. https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data
2. Y. Hu, Y. Koren and C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM 2008, pp. 263-272, DOI: 10.1109/ICDM.2008.22. Author copy, especially equations (3)-(6): https://yifanhu.net/PUB/cf.pdf
3. scikit-learn, *Feature extraction*, TF-IDF and normalized representations: https://scikit-learn.org/stable/modules/feature_extraction.html
4. R. Agrawal, T. Imielinski and A. Swami, "Mining association rules between sets of items in large databases," ACM SIGMOD Record, vol. 22, no. 2, pp. 207-216, 1993. https://doi.org/10.1145/170036.170072 ; original paper: https://www.cs.cmu.edu/~natassa/courses/15-721/papers/p207-agrawal.pdf

The implementation was generated and checked with Codex desktop. The native log records model `gpt-6.1-sol` and runner version `0.163.0-alpha.11`; the desktop app version was not separately measured. No external trained models, external datasets or third-party competition notebooks were used. Keep the original AI session export unmodified.
