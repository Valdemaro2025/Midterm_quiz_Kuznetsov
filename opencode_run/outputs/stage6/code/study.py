#!/usr/bin/env python3
"""Bounded Stage 6 study. All earlier artifacts are read-only inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from stage2.config import load_config, parse_epoch
from stage2.evaluation import map_at_k
from stage2.io_utils import read_json, sha256_file, write_json
from stage4.data import load_stage4_data
from stage4.hybrid import fused_predictions
from stage4.ranking import top_k
from stage5.frozen import AR_PARAMS, HYBRID_PARAMS
from stage5.history import build_popularity, cold_start_stats, read_count_variants
from stage5.predict import fit_predict_hybrid
from stage5.submission import validate_submission, write_submission

OUT = ROOT / "outputs/stage6"
ORIGINAL = ROOT / "outputs/stage5/kuznetsov_vladimir_midterm_submission.csv"
EXPECTED_SHA = "cc0f9b56dd56200adc2075b4977dd9dbdf2fcd0ba2682ffe10529d8241b0cdcb"
START = parse_epoch("2020-03-09T00:00:00")
PRIMARY_END = parse_epoch("2020-05-18T00:00:00")
LATER_END = parse_epoch("2020-06-15T00:00:00")
FINAL_END = parse_epoch("2020-07-27T00:00:00")
VARIANTS = [
    {"name": "repeat_count20", "threshold": .20, "weighting": "count", "half_life_days": None},
    {"name": "repeat_count35", "threshold": .35, "weighting": "count", "half_life_days": None},
    {"name": "repeat_fraction20", "threshold": .20, "weighting": "fraction", "half_life_days": None},
    {"name": "repeat_fraction50", "threshold": .50, "weighting": "fraction", "half_life_days": None},
    *[{"name": f"repeat_count50_h{days}", "threshold": .50, "weighting": "count", "half_life_days": days}
      for days in (14, 28, 56)],
    {"name": "repeat_fraction20_h28", "threshold": .20, "weighting": "fraction", "half_life_days": 28},
]


def save(path, value):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def checkpoint(stage, **details):
    payload = {"stage": stage, "time_epoch": time.time(), **details}
    save("checkpoints/progress.json", payload)
    save(f"checkpoints/{time.time_ns()}_{stage}.json", payload)


def append_attempt(record):
    path = OUT / "search_attempts.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_epoch": time.time(), **record}, sort_keys=True) + "\n")


def preservation_manifest():
    # Include every pre-existing output, not just the submission and search logs.
    paths = []
    for stage in (2, 3, 4, 5):
        paths.extend(p for p in (ROOT / f"outputs/stage{stage}").rglob("*") if p.is_file())
    for directory in ("config", "src", "scripts", "tests"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def prepare():
    if (OUT / "plan.json").exists():
        raise RuntimeError("Plan already exists; resume instead of replacing the predeclaration")
    assert sha256_file(ORIGINAL) == EXPECTED_SHA
    plan = {
        "predeclared_before_evaluation": True,
        "individual_repeat_variants": VARIANTS,
        "hybrids": {"repeat_source": "argmax primary MAP@5 of the eight declared repeat variants; declaration order resolves ties",
                    "repeat_weights": [.70, .85, .95], "association": AR_PARAMS,
                    "normalization_and_ranking": HYBRID_PARAMS},
        "planned_new_unique_configurations": 11, "max_budget": 12,
        "remaining_slot": "unused; no adaptive search",
        "previous_unique_configurations": 19, "expected_combined_unique_configurations": 30,
        "controls": ["repeat_50", "hybrid_70_30"],
        "coverage": "actual unioned covered_sec / original airing_duration; bridged_covered_sec never used",
        "recency": "2 ** (-(training_cutoff - airing_end_epoch) / (86400 * half_life_days)); reject future airing ends",
        "primary_protocol": "frozen Stage 3 labels, candidate pool, cohorts, fallback and repeats allowed",
        "later_split": {"train": ["2020-03-09", "2020-06-15"], "validation": ["2020-06-15", "2020-07-27"],
                        "strategies": ["repeat_50", "hybrid_70_30", "argmax primary MAP@5 of the eleven challengers"],
                        "description": "temporal robustness check, overlaps original holdout; not an independent untouched test",
                        "selection": "freeze challenger before robustness; never select or tune using robustness"},
        "refit_rule": "refit only the strongest challenger if its primary MAP@5 exceeds unchanged hybrid_70_30 control",
        "selection_ties": "declaration order", "no_external_implementations_or_results": True,
    }
    save("plan.json", plan)
    save("observed_kaggle_values.json", {"values": [0.40974, 0.42586], "column_labels": None,
         "source": "user reports successfully scored Stage 5 submission screenshot with absent column headers",
         "public_private_assignment": "unconfirmed", "used_for_tuning": False, "submission_sha256": EXPECTED_SHA})
    save("preservation_before.json", preservation_manifest())
    checkpoint("predeclared", planned_new=11, previous_unique=19)


def read_airings(path):
    fields = {key: [] for key in ("start_epoch", "end_epoch", "duration", "tv_show_id")}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in fields:
                fields[key].append(int(row[key]))
    return {key: np.asarray(values, dtype=np.int64) for key, values in fields.items()}


def load_table(npz_path, airings_path):
    with np.load(npz_path) as archive:
        table = {key: archive[key] for key in ("user_idx", "airing_idx", "covered_sec", "airing_duration", "show_id")}
    airings = read_airings(airings_path)
    indices = table["airing_idx"]
    assert np.array_equal(table["airing_duration"], airings["duration"][indices])
    assert np.array_equal(table["show_id"], airings["tv_show_id"][indices])
    assert np.all(table["covered_sec"] > 0)
    assert np.all(table["covered_sec"] <= table["airing_duration"])
    table["end"] = airings["end_epoch"][indices]
    table["start"] = airings["start_epoch"][indices]
    return table, airings


def slice_table(table, start, end):
    mask = (table["start"] >= start) & (table["end"] <= end)
    return {key: value[mask] for key, value in table.items()}


def airing_weights(covered, duration, ends, cutoff, threshold, weighting, half_life_days=None):
    covered, duration, ends = (np.asarray(x) for x in (covered, duration, ends))
    if np.any(duration <= 0) or np.any(covered < 0) or np.any(covered > duration):
        raise ValueError("Invalid actual observed coverage/original duration")
    if np.any(ends > cutoff):
        raise ValueError("Future airing end leaks beyond training cutoff")
    fractions = covered / duration
    weights = fractions.copy() if weighting == "fraction" else np.ones(len(fractions), dtype=float)
    if weighting not in ("count", "fraction"):
        raise ValueError("Unknown weighting")
    weights[fractions < threshold] = 0
    if half_life_days is not None:
        if half_life_days <= 0:
            raise ValueError("Half-life must be positive")
        weights *= np.exp2(-(cutoff - ends) / (86400.0 * half_life_days))
    return weights


def aggregate(table, cutoff, params):
    weights = airing_weights(table["covered_sec"], table["airing_duration"], table["end"], cutoff,
                             params["threshold"], params["weighting"], params.get("half_life_days"))
    valid = weights > 0
    # Stable airing-index order within each user/program; sum only observed seconds-derived weights.
    u, s, w = table["user_idx"][valid], table["show_id"][valid], weights[valid]
    order = np.lexsort((s, u))
    u, s, w = u[order], s[order], w[order]
    if not len(u):
        return {}
    boundaries = np.r_[0, np.flatnonzero((u[1:] != u[:-1]) | (s[1:] != s[:-1])) + 1]
    totals = np.add.reduceat(w, boundaries)
    result = defaultdict(dict)
    for user, show, value in zip(u[boundaries], s[boundaries], totals):
        result[int(user)][int(show)] = float(value)
    return dict(result)


def variants_from_table(table, cutoff):
    return {name: aggregate(table, cutoff, {"threshold": threshold, "weighting": "count"})
            for name, threshold in (("positive", 0), ("sens50", .5), ("primary80", .8))}


def scores_for(cohort, mapping, items, candidates):
    allowed = set(candidates)
    return {u: {s: v for s, v in items.get(mapping.get(u), {}).items() if s in allowed and v > 0}
            for u in cohort}


def rank_scores(scores, data):
    predictions, info = {}, []
    for uid in data["cohort"]:
        predictions[uid], detail = top_k(scores[uid], data["val_candidates"], data["popularity_val"], data["popularity"])
        info.append(detail)
    return predictions, {"users_needing_fallback": sum(int(x["used_fallback"]) for x in info),
                         "fallback_slots": sum(x["fallback_filled"] for x in info),
                         "mean_scored_candidates_per_user": sum(x["scored_candidates"] for x in info) / len(info)}


def fuse(scores, association, weight, data):
    return fused_predictions(data["cohort"], scores, association, data["val_candidates"],
                             data["popularity_val"], data["popularity"], weight, 1 - weight)


def fusion_rules():
    rules = {k: HYBRID_PARAMS[k] for k in
             ("normalization", "fusion", "tie_break", "zero_score_handling")}
    rules["fallback"] = "cutoff-specific >=80% training popularity restricted to the applicable candidate pool"
    return rules


def clarify_metadata():
    """Make inherited fusion rules explicit without retaining reference-model weights."""
    changed = []
    for scope in ("primary", "robustness"):
        directory = OUT / scope / "methods"
        if not directory.exists():
            continue
        for method in directory.iterdir():
            path = method / "parameters.json"
            params = read_json(path)
            if "repeat_weight" in params:
                params["fusion"] = fusion_rules()
                if params.get("frozen"):
                    params["ranking_and_fallback"] = fusion_rules()
            elif params.get("frozen"):
                params["repeat"] = {"threshold": .5, "weighting": "count", "half_life_days": None}
                params["ranking_and_fallback"] = fusion_rules()
                if params["name"] == "hybrid_70_30":
                    params.update(repeat_weight=.7, association_weight=.3)
            if params != read_json(path):
                write_json(path, params)
                provenance = read_json(method / "provenance.json")
                provenance["parameters_sha256"] = sha256_file(path)
                write_json(method / "provenance.json", provenance)
                changed.append(str(path.relative_to(OUT)))
    path = OUT / "primary/selection.json"
    selection = read_json(path)
    selection["parameters"] = read_json(OUT / f"primary/methods/{selection['strongest']['name']}/parameters.json")
    write_json(path, selection)
    final_path = OUT / "final/configuration.json"
    if final_path.exists():
        final = read_json(final_path)
        final["parameters"] = selection["parameters"]
        write_json(final_path, final)
    if changed:
        save("audit/metadata_clarification.json", {"files": changed,
             "description": "Reporting-only clarification: inherited fusion rules omit reference model name/weights; actual evaluated top-level weights unchanged. Controls spell out frozen thresholds and weights.",
             "experiments_rerun": False, "predictions_changed": False})


def persist_result(scope, name, predictions, data, params, detail, runtime, category):
    evaluation = map_at_k(predictions, data["labels"]["primary_top5"], data["cohort"])
    aps = evaluation.pop("aps")
    base = f"{scope}/methods/{name}"
    save(f"{base}/top5.json", predictions)
    save(f"{base}/per_user_ap5.json", aps)
    save(f"{base}/metrics.json", evaluation)
    save(f"{base}/parameters.json", params)
    save(f"{base}/fallback.json", detail)
    save(f"{base}/runtime.json", runtime)
    save(f"{base}/provenance.json", {"scope": scope, "category": category,
         "source_manifest": "preservation_before.json", "split_provenance": f"{scope}/provenance.json",
         "coverage": "actual unioned seconds; original duration", "training_cutoff": data["cutoff"],
         "repeats_allowed": True, "parameters_sha256": sha256_file(OUT / base / "parameters.json")})
    return {"name": name, **evaluation, "category": category}


def attempted(scope, name, category, function):
    append_attempt({"event": "started", "scope": scope, "name": name, "category": category})
    try:
        result = function()
    except Exception:
        append_attempt({"event": "failed", "scope": scope, "name": name, "category": category,
                        "traceback": traceback.format_exc()})
        checkpoint("failed", scope=scope, name=name)
        raise
    append_attempt({"event": "completed", "scope": scope, "name": name, "category": category})
    return result


def run_controls(data, scope):
    started = time.perf_counter()
    fitted = fit_predict_hybrid(data["cohort"], data["count_variants"], data["user_to_train_idx"],
                                data["val_candidates"], data["popularity_val"], data["popularity"])
    elapsed = time.perf_counter() - started
    results = []
    for name in ("repeat_50", "hybrid_70_30"):
        def run(name=name):
            pred, details = (rank_scores(fitted["repeat_scores"], data) if name == "repeat_50"
                             else (fitted["predictions"], fitted["details"]["fusion"]))
            if scope == "primary":
                target = ROOT / f"outputs/stage4/methods/{name}/top5.json"
                assert pred == read_json(target), f"Control changed: {name}"
            return persist_result(scope, name, pred, data, {"name": name, "frozen": True, "association": AR_PARAMS if name.startswith("hybrid") else None},
                                  details, {"shared_control_fit_predict_seconds": elapsed,
                                  "scope": "shared direct timing for both unchanged controls; not additive"}, "control_reproduction" if scope == "primary" else "robustness")
        results.append(attempted(scope, name, "control_reproduction" if scope == "primary" else "robustness", run))
    return fitted, results, elapsed


def primary():
    if (OUT / "primary/selection.json").exists():
        return
    assert (OUT / "plan.json").exists()
    data = load_stage4_data(load_config(ROOT / "config/stage3.json"))
    data["cutoff"] = PRIMARY_END
    save("primary/labels.json", data["labels"])
    save("primary/cohort.json", data["cohort"])
    save("primary/candidates.json", data["val_candidates"])
    save("primary/provenance.json", {"source": "unchanged frozen Stage 3 holdout",
         "coverage_cache": "outputs/stage3/interactions/train_user_airing.npz",
         "airings": "outputs/stage3/airings/airings_train.csv", "train_cutoff": PRIMARY_END,
         "labels_sha256": sha256_file(ROOT / "outputs/stage3/labels/val_labels_frozen.json"),
         "candidate_sha256": sha256_file(ROOT / "outputs/stage3/candidates/candidates_val.json")})
    table, _ = load_table(ROOT / "outputs/stage3/interactions/train_user_airing.npz",
                          ROOT / "outputs/stage3/airings/airings_train.csv")
    # Reproduction proves actual coverage count semantics match unchanged repeat_50.
    counted50 = aggregate(table, PRIMARY_END, {"threshold": .5, "weighting": "count"})
    assert counted50 == data["count_variants"]["sens50"]
    fitted, results, ar_seconds = run_controls(data, "primary")
    maps, times = {}, {}
    for params in VARIANTS:
        name = params["name"]
        def run(params=params, name=name):
            started = time.perf_counter()
            items = aggregate(table, PRIMARY_END, params)
            scores = scores_for(data["cohort"], data["user_to_train_idx"], items, data["val_candidates"])
            predictions, detail = rank_scores(scores, data)
            elapsed = time.perf_counter() - started
            maps[name], times[name] = scores, elapsed
            return persist_result("primary", name, predictions, data, params, detail,
                                  {"direct_aggregate_and_predict_seconds": elapsed}, "new_unique")
        results.append(attempted("primary", name, "new_unique", run))
        checkpoint("individuals", completed_new=len(results)-2)
    individual = max(results[2:], key=lambda r: r["map_at_k"])
    best_params = next(p for p in VARIANTS if p["name"] == individual["name"])
    save("primary/individual_selection.json", {"selected": individual, "parameters": best_params,
                                               "rule": "primary MAP@5; declaration order breaks ties"})
    for weight in (.70, .85, .95):
        name = f"hybrid_{individual['name']}_w{round(weight*100)}"
        params = {"name": name, "repeat": best_params, "repeat_weight": weight,
                  "association_weight": 1-weight, "association": AR_PARAMS, "fusion": fusion_rules()}
        def run(params=params, name=name, weight=weight):
            started = time.perf_counter()
            predictions, detail = fuse(maps[individual["name"]], fitted["association_scores"], weight, data)
            elapsed = time.perf_counter() - started
            return persist_result("primary", name, predictions, data, params, detail,
                                  {"direct_fusion_seconds": elapsed,
                                   "reused_repeat_aggregate_predict_seconds": times[individual["name"]],
                                   "shared_controls_and_association_seconds": ar_seconds,
                                   "scope": "fusion directly timed; components reused, not standalone end-to-end timing"}, "new_unique")
        results.append(attempted("primary", name, "new_unique", run))
        checkpoint("hybrids", completed_new=len(results)-2)
    strongest = max(results[2:], key=lambda r: r["map_at_k"])
    baseline = next(r for r in results if r["name"] == "hybrid_70_30")
    selection = {"strongest": strongest, "baseline": baseline,
                 "improved": strongest["map_at_k"] > baseline["map_at_k"],
                 "primary_gain": strongest["map_at_k"] - baseline["map_at_k"],
                 "new_unique_configurations": 11, "previous_unique": 19, "combined_unique": 30,
                 "parameters": read_json(OUT / f"primary/methods/{strongest['name']}/parameters.json"),
                 "frozen_before_robustness": True}
    save("primary/results.json", results)
    save("primary/selection.json", selection)
    checkpoint("primary_complete", selection=strongest["name"], gain=selection["primary_gain"])


def full_data():
    cfg = load_config(ROOT / "config/stage5.json")
    table, airings = load_table(cfg.out("interactions", "history_user_airing.npz"),
                                cfg.out("airings", "airings_history.csv"))
    ids = read_json(cfg.out("views", "id_maps_history.json"))["history"]["user_ids"]
    return cfg, table, airings, ids


def predict_challenger(table, data, params, association):
    repeat_params = params.get("repeat", params)
    started = time.perf_counter()
    items = aggregate(table, data["cutoff"], repeat_params)
    scores = scores_for(data["cohort"], data["user_to_train_idx"], items, data["val_candidates"])
    if "repeat_weight" in params:
        predictions, detail = fuse(scores, association, params["repeat_weight"], data)
    else:
        predictions, detail = rank_scores(scores, data)
    return predictions, detail, time.perf_counter() - started, items


def robustness():
    if (OUT / "robustness/results.json").exists():
        return
    selection = read_json(OUT / "primary/selection.json")
    cfg, table, airings, ids = full_data()
    train = slice_table(table, START, LATER_END)
    val = slice_table(table, LATER_END, FINAL_END)
    variants = variants_from_table(train, LATER_END)
    val80 = aggregate(val, FINAL_END, {"threshold": .8, "weighting": "count"})
    cohort = read_json(OUT / "primary/cohort.json")
    # Training user eligibility comes exclusively from views available before this cutoff.
    with np.load(cfg.out("views", "views_history.npz")) as views:
        eligible = set(int(x) for x in np.unique(views["user_idx"][(views["start"] < LATER_END) & (views["stop"] > START)]))
    mapping = {u: i for i, u in enumerate(ids) if i in eligible}
    labels = {}
    for uid in cohort:
        counts = val80.get(ids.index(uid), {}) if uid in ids else {}
        labels[uid] = sorted(counts, key=lambda s: (-counts[s], s))[:5]
    # The original candidate-airing cache includes Jul27-crossing airings excluded from interaction labels.
    candidate_airings = read_airings(ROOT / "outputs/stage3/airings/candidate_airings_val.csv")
    mask = (candidate_airings["start_epoch"] >= LATER_END) & (candidate_airings["start_epoch"] < FINAL_END)
    candidates = sorted(set(int(x) for x in candidate_airings["tv_show_id"][mask]))
    pop_ids, pop_stats, pop_rows = build_popularity(variants, candidates)
    data = {"cohort": cohort, "labels": {"primary_top5": labels}, "count_variants": variants,
            "user_to_train_idx": mapping, "val_candidates": candidates, "popularity_val": pop_ids,
            "popularity": pop_stats, "cutoff": LATER_END}
    save("robustness/labels.json", data["labels"])
    save("robustness/cohort.json", cohort)
    save("robustness/candidates.json", candidates)
    save("robustness/popularity.json", pop_rows)
    save("robustness/cold_start.json", cold_start_stats(cohort, mapping, variants))
    save("robustness/provenance.json", {
        "description": "temporal robustness check; overlaps original holdout; not an independent untouched test",
        "train_window": [START, LATER_END], "validation_window": [LATER_END, FINAL_END],
        "source": "Stage 5 full-history unioned coverage cache; filter fully contained airings before aggregating",
        "cache_validity": "For a fully contained airing, unioned observations intersecting that airing are identical with or without clipping views to the split. No crossing-cutoff airing is used.",
        "label_rule": "count >=80% actual coverage, top5 by descending qualifying count then ascending ID",
        "candidate_rule": "positive labelled airing starts in validation, including crossing end",
        "train_rows": len(train["user_idx"]), "val_rows": len(val["user_idx"]),
        "max_training_airing_end": int(train["end"].max()), "min_validation_airing_start": int(val["start"].min()),
        "training_users_with_views": len(eligible), "candidate_count": len(candidates),
        "excluded_cross_cutoff_interaction_rows": int(np.sum((table["start"] < LATER_END) & (table["end"] > LATER_END))),
        "selected_before_this_split": selection["strongest"]["name"],
    })
    fitted, results, elapsed = run_controls(data, "robustness")
    name, params = selection["strongest"]["name"], selection["parameters"]
    def run():
        predictions, detail, seconds, _ = predict_challenger(train, data, params, fitted["association_scores"])
        return persist_result("robustness", name, predictions, data, params, detail,
                              {"aggregate_predict_seconds": seconds, "shared_controls_association_seconds": elapsed}, "robustness")
    results.append(attempted("robustness", name, "robustness", run))
    save("robustness/results.json", results)
    checkpoint("robustness_complete", results=results)


def final_refit():
    clarify_metadata()
    if (OUT / "final/configuration.json").exists():
        return
    selection = read_json(OUT / "primary/selection.json")
    if not selection["improved"]:
        save("final/decision.json", {"refitted": False, "reason": "No primary improvement; retain Stage 5 submission"})
        return
    name, params = selection["strongest"]["name"], selection["parameters"]
    def run():
        started = time.perf_counter()
        cfg, table, airings, ids = full_data()
        variants = read_count_variants(cfg.out("interactions", "history_user_program_counts.csv"))
        candidates = read_json(cfg.out("candidates", "candidates_test.json"))["candidate_ids"]
        cohort = read_json(OUT / "primary/cohort.json")
        pop_ids, pop_stats, _ = build_popularity(variants, candidates)
        data = {"cohort": cohort, "count_variants": variants, "user_to_train_idx": {u:i for i,u in enumerate(ids)},
                "val_candidates": candidates, "popularity_val": pop_ids, "popularity": pop_stats, "cutoff": FINAL_END}
        model_started = time.perf_counter()
        # Only the selected strategy is refitted; association is fitted only if it is a hybrid.
        association = {}
        if "repeat_weight" in params:
            from stage3.association import build_association_index
            from stage4.association import association_scores
            ar_data = {"cohort": cohort, "user_to_train_idx": data["user_to_train_idx"],
                       "user_items": variants["primary80"], "val_candidates": candidates}
            index = build_association_index(ar_data)
            keys = ("min_count", "min_confidence", "min_lift", "max_rules_per_antecedent", "score_scheme", "aggregate", "max_antecedents_per_user")
            association, _ = association_scores(ar_data, index, **{k: AR_PARAMS[k] for k in keys})
        predictions, detail, _, items = predict_challenger(table, data, params, association)
        model_seconds = time.perf_counter() - model_started
        path = OUT / "kuznetsov_vladimir_midterm_submission_challenger.csv"
        write_submission(path, cohort, predictions)
        checks = validate_submission(path, cohort, candidates)
        assert checks["all_passed"], checks
        save("final/submission_checks.json", checks)
        save("final/predictions.json", predictions)
        cold = cold_start_stats(cohort, data["user_to_train_idx"], variants)
        cold["scope"] = "final full-history weeks 11-30, not primary-validation training"
        cold["challenger_distinct_qualifying_items_zero"] = sum(not items.get(data["user_to_train_idx"].get(u), {}) for u in cohort)
        save("final/cold_start.json", cold)
        save("final/fallback.json", detail)
        save("final/configuration.json", {"selected_strategy": name, "parameters": params,
             "training_window": [START, FINAL_END], "candidate_count": len(candidates),
             "submission": str(path.relative_to(ROOT)), "submission_sha256": sha256_file(path),
             "original_sha256": sha256_file(ORIGINAL), "model_refit_predict_seconds": model_seconds,
             "total_refit_seconds_including_cache_load": time.perf_counter()-started,
             "boundary_rules": "verified Stage 5 full-history reconstruction: restore 66 May18 crossings, exclude 58 Jul27 crossings",
             "source_manifest": "preservation_before.json", "kaggle_score": None})
        checkpoint("final_refit_complete", sha256=sha256_file(path))
    attempted("final", name, "final_refit_not_new_configuration", run)


def independent_ap(predicted, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.
    precisions = []
    encountered = set()
    for position, item in enumerate(predicted[:5]):
        if item in relevant and item not in encountered:
            precisions.append((len(precisions) + 1) / (position + 1))
        encountered.add(item)
    return math.fsum(precisions) / min(5, len(relevant))


def verify():
    clarify_metadata()
    checks = {}
    before = read_json(OUT / "preservation_before.json")
    checks["all_previous_artifacts_unchanged"] = all(sha256_file(ROOT / p) == h for p,h in before.items())
    checks["original_submission_sha256"] = sha256_file(ORIGINAL) == EXPECTED_SHA
    metrics_audit = []
    for scope in ("primary", "robustness"):
        users = read_json(OUT / f"{scope}/cohort.json")
        labels = read_json(OUT / f"{scope}/labels.json")["primary_top5"]
        candidates = set(read_json(OUT / f"{scope}/candidates.json"))
        for directory in sorted((OUT / scope / "methods").iterdir()):
            predictions = read_json(directory / "top5.json")
            aps = {u: independent_ap(predictions[u], labels[u]) for u in users}
            stored_aps = read_json(directory / "per_user_ap5.json")
            metrics = read_json(directory / "metrics.json")
            groups = {"map_at_k": users,
                      "map_with_positive": [u for u in users if labels[u]],
                      "map_with_at_least_k_positive": [u for u in users if len(set(labels[u])) >= 5]}
            recomputed = {k: math.fsum(aps[u] for u in group)/len(group) if group else 0.
                          for k,group in groups.items()}
            ok = all(abs(recomputed[k]-metrics[k]) < 1e-12 for k in recomputed)
            ok &= all(abs(aps[u]-stored_aps[u]) < 1e-12 for u in users)
            ok &= set(predictions) == set(users)
            ok &= all(len(p)==5 and len(set(p))==5 and set(p)<=candidates for p in predictions.values())
            checks[f"{scope}/{directory.name}/independent_metrics_and_predictions"] = bool(ok)
            metrics_audit.append({"scope": scope, "name": directory.name, "ok": bool(ok), **recomputed})
    attempts = [json.loads(line) for line in (OUT / "search_attempts.jsonl").read_text().splitlines()]
    new = {r["name"] for r in attempts if r["category"]=="new_unique" and r["event"]=="started"}
    completed = {r["name"] for r in attempts if r["category"]=="new_unique" and r["event"]=="completed"}
    checks["bounded_11_unique_configs"] = len(new)==len(completed)==11
    checks["three_fixed_robustness_strategies"] = len(read_json(OUT / "robustness/results.json"))==3
    provenance = read_json(OUT / "robustness/provenance.json")
    checks["robustness_no_training_future"] = provenance["max_training_airing_end"] <= LATER_END
    checks["robustness_no_validation_before_cutoff"] = provenance["min_validation_airing_start"] >= LATER_END
    # Reconstruct later labels independently using integer coverage comparisons,
    # without the signal aggregation or label generation code.
    _, table, _, ids = full_data()
    label_counts = defaultdict(lambda: defaultdict(int))
    mask = ((table["start"] >= LATER_END) & (table["end"] <= FINAL_END)
            & (table["covered_sec"] * 5 >= table["airing_duration"] * 4))
    for user, show in zip(table["user_idx"][mask], table["show_id"][mask]):
        label_counts[ids[int(user)]][int(show)] += 1
    robust_labels = read_json(OUT / "robustness/labels.json")["primary_top5"]
    independent_labels = {u: sorted(label_counts[u], key=lambda s: (-label_counts[u][s], s))[:5]
                          for u in robust_labels}
    checks["independent_later_labels_match_documented_rules"] = independent_labels == robust_labels
    selection = read_json(OUT / "primary/selection.json")
    if selection["improved"]:
        config = read_json(OUT / "final/configuration.json")
        path = ROOT / config["submission"]
        cfg = load_config(ROOT / "config/stage5.json")
        submission_checks = validate_submission(path, read_json(OUT / "primary/cohort.json"),
                                               read_json(cfg.out("candidates", "candidates_test.json"))["candidate_ids"])
        checks["challenger_submission_all_stage5_checks"] = submission_checks["all_passed"]
        checks["challenger_submission_sha256"] = sha256_file(path)==config["submission_sha256"]
        checks["final_cache_no_future_airings"] = bool(np.all(table["end"] <= FINAL_END))
        checks["final_boundary_handling_verified"] = (
            read_json(ROOT / "outputs/stage5/audit/history_schedule_audit.json")["restored_may18_crossing_airings"] == 66
            and read_json(ROOT / "outputs/stage5/audit/history_schedule_audit.json")["excluded_full_history_crossing_end"] == 58)
    summary = {"all_passed": all(checks.values()), "checks": checks, "checks_passed": sum(checks.values()),
               "checks_total": len(checks), "preserved_files": len(before), "new_unique_configs": len(new),
               "combined_unique_configs": 19+len(new), "attempt_event_count": len(attempts),
               "failed_attempts": [r for r in attempts if r["event"]=="failed"], "independent_metrics": metrics_audit}
    save("audit/validation.json", summary)
    save("audit/source_code.json", {str(p.relative_to(ROOT)): sha256_file(p)
         for p in sorted((OUT / "code").glob("*.py"))})
    robust = read_json(OUT / "robustness/results.json")
    robust_by_name = {r["name"]:r for r in robust}
    challenger = robust_by_name[selection["strongest"]["name"]]
    robustness_gain = challenger["map_at_k"] - robust_by_name["hybrid_70_30"]["map_at_k"]
    save("outcome.json", {"primary": selection, "robustness": robust,
         "robustness_gain_vs_hybrid_control": robustness_gain,
         "split_disagreement": selection["primary_gain"]>0 and robustness_gain<=0,
         "interpretation": "temporal robustness check, not an independent untouched test; same users/candidates/labels across strategies within each split",
         "recommendation": ("Consider one manual challenger upload; gains observed on both chronological splits, not independently established"
                            if selection["improved"] and robustness_gain>0 else
                            "Retain existing submission as preferred: split disagreement" if selection["improved"] else
                            "Retain existing submission: no primary improvement"),
         "kaggle_values": "observed 0.40974 and 0.42586; column labels unconfirmed; not used in model selection"})
    assert summary["all_passed"], summary
    checkpoint("verified_complete", checks=f"{sum(checks.values())}/{len(checks)}")
    print(json.dumps({"primary_best": selection["strongest"]["name"], "primary_map": selection["strongest"]["map_at_k"],
                      "primary_gain": selection["primary_gain"], "robustness_gain": robustness_gain,
                      "checks": f"{sum(checks.values())}/{len(checks)}"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "primary", "robustness", "final", "verify", "all"))
    args = parser.parse_args()
    phases = {"prepare": prepare, "primary": primary, "robustness": robustness, "final": final_refit, "verify": verify}
    if args.phase == "all":
        for phase in (primary, robustness, final_refit, verify):
            phase()
    else:
        phases[args.phase]()


if __name__ == "__main__":
    main()
