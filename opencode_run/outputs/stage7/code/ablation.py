#!/usr/bin/env python3
"""One feedback-motivated no-decay ablation; prior artifacts remain read-only."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("stage6_readonly", ROOT / "outputs/stage6/code/study.py")
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)
OUT = ROOT / "outputs/stage7"
S6 = ROOT / "outputs/stage6"
s6.OUT = OUT  # Reused persistence helpers must write exclusively into Stage 7.
NAME = "hybrid_repeat_fraction20_nodecay_w85"
PARENT_NAME = "hybrid_repeat_fraction20_h28_w85"
STAGE5_SHA = "cc0f9b56dd56200adc2075b4977dd9dbdf2fcd0ba2682ffe10529d8241b0cdcb"
STAGE6_SHA = "7f23a5d3d41cbae99f839cca19c262d5e59a334365b64c538dfec8f9619613fe"
STAGE5 = ROOT / "outputs/stage5/kuznetsov_vladimir_midterm_submission.csv"
STAGE6 = S6 / "kuznetsov_vladimir_midterm_submission_challenger.csv"
SUBMISSION = OUT / "kuznetsov_vladimir_midterm_submission_ablation.csv"


def strip_names(value):
    if isinstance(value, dict):
        return {k: strip_names(v) for k, v in value.items() if k != "name"}
    return value


def params():
    parent = s6.read_json(S6 / "final/configuration.json")["parameters"]
    result = copy.deepcopy(parent)
    result["name"] = NAME
    result["repeat"]["name"] = "repeat_fraction20"
    result["repeat"]["half_life_days"] = None
    expected = copy.deepcopy(parent)
    expected["repeat"]["half_life_days"] = None
    assert strip_names(result) == strip_names(expected)
    assert result["repeat"]["threshold"] == .2
    assert result["repeat"]["weighting"] == "fraction"
    assert result["repeat_weight"] == .85
    assert math.isclose(result["association_weight"], .15, abs_tol=1e-15)
    return result


def prior_manifest():
    files = []
    for stage in range(2, 7):
        files += [p for p in (ROOT / f"outputs/stage{stage}").rglob("*") if p.is_file()]
    for directory in ("config", "src", "scripts", "tests"):
        files += [p for p in (ROOT / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    return {str(p.relative_to(ROOT)): s6.sha256_file(p) for p in sorted(files)}


def prepare():
    if (OUT / "plan.json").exists():
        return
    assert s6.sha256_file(STAGE5) == STAGE5_SHA
    assert s6.sha256_file(STAGE6) == STAGE6_SHA
    s6.save("preservation_before.json", prior_manifest())
    inspected, matches = [], []
    configurations = []
    for stage in (3, 4):
        configurations += list((ROOT / f"outputs/stage{stage}/methods").glob("*/config.json"))
    configurations += list((S6 / "primary/methods").glob("*/parameters.json"))
    for path in sorted(configurations):
        config = s6.read_json(path)
        inspected.append({"path": str(path.relative_to(ROOT)), "sha256": s6.sha256_file(path)})
        if strip_names(config) == strip_names(params()):
            matches.append(str(path.parent.relative_to(ROOT)))
    # Earlier hybrids without nested repeat settings cannot be matches: their
    # explicit repeat weights are .5 or .7, not .85. Stage 6 includes every .85 hybrid.
    s6.save("deduplication.json", {"exact_matches": matches, "inspected": inspected,
         "note": "Stage 6 already evaluated the no-decay individual repeat signal, but not its 85/15 hybrid. Earlier Stage 3/4 hybrids have different weights.",
         "reuse_if_exact_match": True})
    s6.save("plan.json", {"name": NAME, "parameters": params(), "new_unique_configurations": 0 if matches else 1,
         "max_new_unique_configurations": 1, "earlier_unique_configurations": 30,
         "changed_experimental_factor": {"repeat.half_life_days": {"from": 28, "to": None}},
         "motivation": "Follow-up hypothesis motivated by the returned Stage 6 Kaggle regression; not a preplanned experiment",
         "search": "No additional thresholds, weights, or strategies",
         "splits": "reuse the two existing Stage 6 splits, labels, candidates, cohorts and fallback rules",
         "refit": "fixed variant on full history regardless of local outcome",
         "designated_best": "Stage 5 remains best; this study cannot change that designation"})
    s6.save("kaggle_negative_result.json", {
         "source": "user-reported returned Kaggle scores", "column_labels": None,
         "column_order": "first and second returned columns only; Public/Private unconfirmed",
         "attempts": [
             {"stage": 5, "observed_values": [.40974, .42586], "path": str(STAGE5.relative_to(ROOT)), "sha256": STAGE5_SHA},
             {"stage": 6, "observed_values": [.39348, .40839], "path": str(STAGE6.relative_to(ROOT)), "sha256": STAGE6_SHA}],
         "stage6_minus_stage5": [-.01626, -.01747],
         "result": "Stage 6 worsened both returned score columns despite local improvements",
         "motivated_stage7_hypothesis": True, "stage7_is_preplanned": False})
    s6.save("current_best_submission.json", {"stage": 5, "path": str(STAGE5.relative_to(ROOT)),
         "sha256": STAGE5_SHA, "observed_values": [.40974, .42586], "column_labels": None,
         "designation": "current best, retained per user instruction", "stage7_must_not_replace": True})
    s6.checkpoint("fixed_ablation_recorded", configuration=NAME, unique_new=0 if matches else 1)


def association(data):
    from stage3.association import build_association_index
    from stage4.association import association_scores
    ar_data = {"cohort": data["cohort"], "user_to_train_idx": data["user_to_train_idx"],
               "user_items": data["count_variants"]["primary80"], "val_candidates": data["val_candidates"]}
    index = build_association_index(ar_data)
    settings = params()["association"]
    keys = ("min_count", "min_confidence", "min_lift", "max_rules_per_antecedent", "score_scheme", "aggregate", "max_antecedents_per_user")
    return association_scores(ar_data, index, **{k: settings[k] for k in keys})


def split_data(scope):
    if scope == "primary":
        data = s6.load_stage4_data(s6.load_config(ROOT / "config/stage3.json"))
        data["cutoff"] = s6.PRIMARY_END
        table, _ = s6.load_table(ROOT / "outputs/stage3/interactions/train_user_airing.npz",
                                 ROOT / "outputs/stage3/airings/airings_train.csv")
    else:
        cfg, full, _, ids = s6.full_data()
        table = s6.slice_table(full, s6.START, s6.LATER_END)
        variants = s6.variants_from_table(table, s6.LATER_END)
        # Preserve exact training-view eligibility from the existing later split.
        with s6.np.load(cfg.out("views", "views_history.npz")) as views:
            mask = (views["start"] < s6.LATER_END) & (views["stop"] > s6.START)
            eligible = set(int(x) for x in s6.np.unique(views["user_idx"][mask]))
        candidates = s6.read_json(S6 / "robustness/candidates.json")
        popularity = s6.read_json(S6 / "robustness/popularity.json")
        pop_ids, pop_stats, rebuilt = s6.build_popularity(variants, candidates)
        assert rebuilt == popularity
        data = {"cohort": s6.read_json(S6 / "robustness/cohort.json"),
                "labels": s6.read_json(S6 / "robustness/labels.json"),
                "count_variants": variants, "user_to_train_idx": {u:i for i,u in enumerate(ids) if i in eligible},
                "val_candidates": candidates, "popularity_val": pop_ids, "popularity": pop_stats,
                "cutoff": s6.LATER_END}
    assert data["labels"] == s6.read_json(S6 / f"{scope}/labels.json")
    assert data["cohort"] == s6.read_json(S6 / f"{scope}/cohort.json")
    assert data["val_candidates"] == s6.read_json(S6 / f"{scope}/candidates.json")
    return data, table


def evaluate(scope):
    completed = OUT / f"{scope}/result.json"
    if completed.exists():
        return
    def run():
        loaded = time.perf_counter()
        data, table = split_data(scope)
        s6.save(f"{scope}/labels.json", data["labels"])
        s6.save(f"{scope}/cohort.json", data["cohort"])
        s6.save(f"{scope}/candidates.json", data["val_candidates"])
        s6.save(f"{scope}/provenance.json", {"labels_candidates_cohort_source": f"outputs/stage6/{scope}",
             "coverage_source": "outputs/stage3/interactions/train_user_airing.npz" if scope == "primary" else "outputs/stage5/interactions/history_user_airing.npz",
             "training_cutoff": data["cutoff"], "max_training_airing_end": int(table["end"].max()),
             "label_rule": "unchanged >=80% actual coverage frequency top5 labels",
             "robustness_interpretation": "overlapping temporal robustness check, not independent untouched test",
             "actual_union_seconds_only": True, "future_airings": int(s6.np.sum(table["end"] > data["cutoff"]))})
        previous = s6.read_json(OUT / "deduplication.json")["exact_matches"]
        reuse = None
        for candidate in previous:
            candidate = ROOT / candidate
            if scope == "robustness":
                candidate = Path(str(candidate).replace("/primary/", "/robustness/"))
            if (candidate / "top5.json").exists():
                reuse = candidate
                break
        if reuse:
            predictions = s6.read_json(reuse / "top5.json")
            detail = s6.read_json(reuse / "fallback.json")
            runtime = {"reused_exact_configuration": str(reuse.relative_to(ROOT)), "new_fit": False}
        else:
            started = time.perf_counter()
            ar_scores, ar_detail = association(data)
            predictions, detail, repeat_seconds, _ = s6.predict_challenger(table, data, params(), ar_scores)
            runtime = {"direct_fit_and_predict_seconds": time.perf_counter()-started,
                       "repeat_aggregate_and_fusion_seconds": repeat_seconds,
                       "total_with_cache_load_seconds": time.perf_counter()-loaded}
            s6.save(f"{scope}/association_details.json", ar_detail)
        # Verify the reused no-decay repeat subcomponent against its already
        # evaluated Stage 6 individual on the primary split, without a new search.
        if scope == "primary":
            items = s6.aggregate(table, data["cutoff"], params()["repeat"])
            scores = s6.scores_for(data["cohort"], data["user_to_train_idx"], items, data["val_candidates"])
            repeat_predictions, _ = s6.rank_scores(scores, data)
            assert repeat_predictions == s6.read_json(S6 / "primary/methods/repeat_fraction20/top5.json")
            s6.save("primary/repeat_subcomponent_reproduction.json", {"equal": True,
                 "source": "outputs/stage6/primary/methods/repeat_fraction20/top5.json",
                 "new_configuration": False})
        result = s6.persist_result(scope, NAME, predictions, data, params(), detail, runtime,
                                  "exact_configuration_reuse" if reuse else "new_unique" if scope == "primary" else "fixed_temporal_robustness")
        s6.save(f"{scope}/result.json", result)
        s6.checkpoint(f"{scope}_complete", map_at_5=result["map_at_k"])
    s6.attempted(scope, NAME, "new_unique" if scope == "primary" and not s6.read_json(OUT / "deduplication.json")["exact_matches"]
                 else "fixed_configuration_evaluation_or_reuse", run)


def refit():
    if (OUT / "final/configuration.json").exists():
        return
    def run():
        started = time.perf_counter()
        cfg, table, _, ids = s6.full_data()
        variants = s6.read_count_variants(cfg.out("interactions", "history_user_program_counts.csv"))
        candidates = s6.read_json(cfg.out("candidates", "candidates_test.json"))["candidate_ids"]
        users = s6.read_json(S6 / "primary/cohort.json")
        pop_ids, pop_stats, _ = s6.build_popularity(variants, candidates)
        data = {"cohort": users, "count_variants": variants, "val_candidates": candidates,
                "user_to_train_idx": {u:i for i,u in enumerate(ids)}, "popularity_val": pop_ids,
                "popularity": pop_stats, "cutoff": s6.FINAL_END}
        model_started = time.perf_counter()
        ar, ar_detail = association(data)
        predictions, detail, _, items = s6.predict_challenger(table, data, params(), ar)
        model_seconds = time.perf_counter()-model_started
        s6.write_submission(SUBMISSION, users, predictions)
        checks = s6.validate_submission(SUBMISSION, users, candidates)
        assert checks["all_passed"]
        s6.save("final/submission_checks.json", checks)
        s6.save("final/predictions.json", predictions)
        s6.save("final/fallback.json", detail)
        s6.save("final/association_details.json", ar_detail)
        cold = s6.cold_start_stats(users, data["user_to_train_idx"], variants)
        cold["scope"] = "full historical weeks 11-30; not validation-training cold start"
        cold["zero_ablation_repeat_signal_users"] = sum(not items.get(data["user_to_train_idx"].get(u), {}) for u in users)
        s6.save("final/cold_start.json", cold)
        s6.save("final/configuration.json", {"parameters": params(), "submission": str(SUBMISSION.relative_to(ROOT)),
             "submission_sha256": s6.sha256_file(SUBMISSION), "training_window": [s6.START, s6.FINAL_END],
             "test_candidates": len(candidates), "model_refit_predict_seconds": model_seconds,
             "total_refit_seconds_with_cache_load": time.perf_counter()-started,
             "boundary_handling": "reuse verified Stage 5 full-history cache: 66 May18 crossings restored; 58 July27 crossings excluded",
             "source_manifest": "preservation_before.json", "stage7_kaggle_score": None,
             "designated_best": "Stage 5, unchanged"})
        s6.checkpoint("refit_complete", sha256=s6.sha256_file(SUBMISSION))
    s6.attempted("full_history", NAME, "same_configuration_refit", run)


def independent_ap(predictions, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.
    seen, hits, contributions = set(), 0, []
    for rank, item in enumerate(predictions[:5], 1):
        if item in relevant and item not in seen:
            hits += 1
            contributions.append(hits/rank)
        seen.add(item)
    return math.fsum(contributions)/min(5, len(relevant))


def verify():
    checks, comparisons, recomputed = {}, {}, {}
    snapshot = s6.read_json(OUT / "preservation_before.json")
    checks["every_prior_artifact_unchanged"] = all((ROOT/p).exists() and s6.sha256_file(ROOT/p)==h for p,h in snapshot.items())
    checks["stage5_submission_sha256"] = s6.sha256_file(STAGE5)==STAGE5_SHA
    checks["stage6_submission_sha256"] = s6.sha256_file(STAGE6)==STAGE6_SHA
    for scope in ("primary", "robustness"):
        users = s6.read_json(OUT / f"{scope}/cohort.json")
        labels = s6.read_json(OUT / f"{scope}/labels.json")
        candidates = s6.read_json(OUT / f"{scope}/candidates.json")
        checks[f"{scope}_unchanged_labels"] = labels == s6.read_json(S6 / f"{scope}/labels.json")
        checks[f"{scope}_unchanged_cohort"] = users == s6.read_json(S6 / f"{scope}/cohort.json")
        checks[f"{scope}_unchanged_candidates"] = candidates == s6.read_json(S6 / f"{scope}/candidates.json")
        directory = OUT / f"{scope}/methods/{NAME}"
        predictions = s6.read_json(directory / "top5.json")
        aps = {u: independent_ap(predictions[u], labels["primary_top5"][u]) for u in users}
        recorded_aps = s6.read_json(directory / "per_user_ap5.json")
        metrics = s6.read_json(directory / "metrics.json")
        groups = {"map_at_k": users, "map_with_positive": [u for u in users if labels["primary_top5"][u]],
                  "map_with_at_least_k_positive": [u for u in users if len(set(labels["primary_top5"][u]))>=5]}
        values = {key: math.fsum(aps[u] for u in group)/len(group) if group else 0. for key,group in groups.items()}
        checks[f"{scope}_independent_three_cohort_metrics"] = all(abs(values[k]-metrics[k])<1e-12 for k in values)
        checks[f"{scope}_independent_user_aps"] = all(abs(aps[u]-recorded_aps[u])<1e-12 for u in users)
        checks[f"{scope}_predictions_valid"] = set(predictions)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predictions.values())
        provenance = s6.read_json(OUT / f"{scope}/provenance.json")
        checks[f"{scope}_no_future_airings"] = provenance["future_airings"]==0 and provenance["max_training_airing_end"]<=provenance["training_cutoff"]
        parent = s6.read_json(S6 / f"{scope}/methods/{PARENT_NAME}/metrics.json")
        control = s6.read_json(S6 / f"{scope}/methods/hybrid_70_30/metrics.json")
        comparisons[scope] = {"no_decay": values["map_at_k"], "stage6_with_decay": parent["map_at_k"],
             "stage5_strategy_control": control["map_at_k"],
             "no_decay_minus_decay": values["map_at_k"]-parent["map_at_k"],
             "no_decay_minus_stage5_control": values["map_at_k"]-control["map_at_k"]}
        recomputed[scope] = values
    # Single-factor control: names differ, but only the decay parameter may change.
    expected = s6.read_json(S6 / "final/configuration.json")["parameters"]
    expected["repeat"]["half_life_days"] = None
    checks["only_decay_changed"] = strip_names(params())==strip_names(expected)
    plan = s6.read_json(OUT / "plan.json")
    records = [s6.json.loads(line) for line in (OUT / "search_attempts.jsonl").read_text().splitlines()]
    new = {r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    checks["exactly_one_new_fixed_configuration_or_prior_reuse"] = len(new)==plan["new_unique_configurations"] and len(new)<=1
    checks["no_failed_attempts"] = not any(r["event"]=="failed" for r in records)
    config = s6.read_json(OUT / "final/configuration.json")
    cfg = s6.load_config(ROOT / "config/stage5.json")
    checks["submission_all_established_checks"] = s6.validate_submission(SUBMISSION, s6.read_json(S6 / "primary/cohort.json"),
         s6.read_json(cfg.out("candidates", "candidates_test.json"))["candidate_ids"])["all_passed"]
    checks["submission_sha256"] = s6.sha256_file(SUBMISSION)==config["submission_sha256"]
    with SUBMISSION.open(newline="", encoding="utf-8") as handle:
        from_csv = {row["user_id"]: [int(x) for x in row["tv_show_id"].split()] for row in csv.DictReader(handle)}
    checks["submission_matches_saved_predictions"] = from_csv==s6.read_json(OUT / "final/predictions.json")
    best = s6.read_json(OUT / "current_best_submission.json")
    checks["stage5_remains_designated_best"] = best["stage"]==5 and best["sha256"]==STAGE5_SHA
    summary = {"all_passed": all(checks.values()), "checks": checks, "checks_passed": sum(checks.values()),
               "checks_total": len(checks), "preserved_files": len(snapshot), "new_unique_configurations": len(new),
               "combined_unique_configurations": 30+len(new), "independently_recomputed_metrics": recomputed}
    s6.save("audit/validation.json", summary)
    s6.save("audit/source_code.json", {str(p.relative_to(ROOT)): s6.sha256_file(p) for p in sorted((OUT / "code").glob("*.py"))})
    s6.save("outcome.json", {"comparisons": comparisons, "current_best": best,
         "motivation": "Kaggle-feedback-motivated follow-up; not preplanned",
         "temporal_robustness": "later split overlaps original holdout and is not independent untouched evidence",
         "negative_result": "Stage 6 worsened both Kaggle score columns; local gains did not transfer",
         "no_decay_helps_both_local_splits": all(v["no_decay_minus_decay"]>0 for v in comparisons.values()),
         "stage7_uploaded": False, "stage7_kaggle_score": None,
         "recommendation": "Keep Stage 5 designated best regardless of this local ablation; no further search or upload"})
    assert summary["all_passed"], summary
    s6.checkpoint("verified_complete", checks=f"{sum(checks.values())}/{len(checks)}", current_best="Stage 5")
    print(s6.json.dumps({"comparisons": comparisons, "submission_sha256": config["submission_sha256"],
                        "checks": f"{sum(checks.values())}/{len(checks)}", "preserved_files": len(snapshot)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "refit", "verify", "all"))
    phase = parser.parse_args().phase
    if phase in ("prepare", "all"):
        prepare()
    if phase in ("evaluate", "all"):
        evaluate("primary")
        evaluate("robustness")
    if phase in ("refit", "all"):
        refit()
    if phase in ("verify", "all"):
        verify()


if __name__ == "__main__":
    main()
