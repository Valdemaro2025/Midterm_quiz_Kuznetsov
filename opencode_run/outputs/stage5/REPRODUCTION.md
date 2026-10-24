# Stage 5 reproduction

From the project root, run:

```sh
python3 scripts/run_stage5.py
python3 scripts/validate_stage5.py
python3 -m pytest -q
```

Inputs:
- `sweettv-tv-program-recommender/dataset11-30.csv`
- `sweettv-tv-program-recommender/export_arh_11-20-final.csv`
- `sweettv-tv-program-recommender/export_arh_21-30-final.csv`
- `sweettv-tv-program-recommender/export_arh_31-42-final.csv`
- `sweettv-tv-program-recommender/submission.csv` (read-only template)

Output:
- `/Users/vladimirkuznecov/Desktop/Midterm_quiz/outputs/stage5/kuznetsov_vladimir_midterm_submission.csv`

The pipeline reads no test-viewing behavior and does not modify the template or earlier outputs.
