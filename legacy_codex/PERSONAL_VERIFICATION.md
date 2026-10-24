# Personal verification worksheet

These checks were executed by the AI. Complete them yourself before claiming personal verification in the report.

1. Inspect `results/join_verification.json`: choose a raw user/channel interval and its airing. Recompute the intersection and coverage. Confirm the numerator is watched seconds and the denominator is programme duration, not the viewing-log duraton field.

2. For the heavy example user, inspect the Implicit ALS recommendation table in results/analysis.md. Relevant hits occur at ranks 1, 2, 5. Compute AP@5 as the sum of precision at each hit, divided by 5. Record your calculation in your own words.

3. Explain why a programme watched before may be recommended again in this competition.

4. Compare one failure: a familiar training favourite absent from the validation Top-5, or a plausible content recommendation with no validation hit. Distinguish “not in the target Top-5” from “never watched” or “disliked”.

5. Inspect all five IDs of the Kaggle CSV for one user, and verify the programme titles in the test schedule.

6. Add your actual Kaggle public/private screenshot, GitHub URL, and original native AI-session export before final submission. Do not edit or reconstruct the session log.