#!/usr/bin/env python3
"""Fixed Stage 8 B refit, equivalence checks and one Stage 9 submission."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("stage8_readonly", ROOT / "outputs/stage8/code/study.py")
s8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s8)
s6 = s8.s6
np = s8.np
OUT = ROOT / "outputs/stage9"
# Only read-only computation helpers are reused; redirect their artifact writers.
s8.OUT = OUT / "reproduction"
s6.OUT = s8.OUT
NAME = "B_schedule_adjusted_hybrid70"
DESTINATION = OUT / "kuznetsov_vladimir_midterm_submission_schedule.csv"
BEST = ROOT / "outputs/stage5/kuznetsov_vladimir_midterm_submission.csv"
BEST_SHA = "cc0f9b56dd56200adc2075b4977dd9dbdf2fcd0ba2682ffe10529d8241b0cdcb"
TRAIN = (s6.parse_epoch("2020-03-09T00:00:00"), s6.parse_epoch("2020-07-27T00:00:00"))
TARGET = (TRAIN[1], s6.parse_epoch("2020-10-19T00:00:00"))


def save(path, value):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    s6.write_json(path, value)


def checkpoint(stage, **details):
    value = {"stage": stage, "time_epoch": time.time(), **details}
    save("checkpoints/progress.json", value)
    save(f"checkpoints/{time.time_ns()}_{stage}.json", value)


def attempt(name, operation):
    path = OUT / "attempts.jsonl"
    def log(event, **details):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": name, "event": event, "time_epoch": time.time(),
                 "category": "fixed_refit_or_reproduction_not_new_configuration", **details}, sort_keys=True)+"\n")
    log("started")
    try:
        result = operation()
    except Exception:
        log("failed", traceback=traceback.format_exc())
        checkpoint("failed", operation=name)
        raise
    log("completed")
    return result


def workspace_manifest():
    # Include raw inputs, root-level submission copies, prior caches and sources.
    # New Stage 9 files alone are excluded; bytecode/cache writes are disabled.
    return {str(p.relative_to(ROOT)): {"sha256": s6.sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_relative_to(OUT)}


def prepare():
    if (OUT / "configuration.json").exists():
        return
    assert not DESTINATION.exists(), "Refuse to overwrite an existing Stage 9 CSV"
    assert s6.sha256_file(BEST) == BEST_SHA
    original = ROOT / f"outputs/stage8/horizon12/methods/{NAME}/parameters.json"
    settings = s6.read_json(original)
    assert settings == s6.read_json(ROOT / f"outputs/stage8/original10/methods/{NAME}/parameters.json")
    assert settings["association"] == s6.AR_PARAMS
    assert settings["repeat_weight"] == .7 and settings["association_weight"] == .3
    assert s6.read_json(ROOT / "outputs/stage8/search_history.json")["unique_configurations"] == 35
    started = time.perf_counter()
    save("preservation_before.json", workspace_manifest())
    save("configuration.json", {"name": NAME, "operation": "fixed final refit; no improvement study or search",
         "new_unique_configurations": 0, "unique_configurations_before": 35, "unique_configurations_after": 35,
         "stage8_parameters": settings, "stage8_parameter_file": str(original.relative_to(ROOT)),
         "stage8_parameter_sha256": s6.sha256_file(original),
         "repeat": {"coverage": ">=50% actual unioned observed seconds / original airing duration",
                    "weighting": "qualifying-airing count", "time_decay": None},
         "opportunity_adjustment": {"formula": "sqrt(clip((target_count/target_weeks)/(historical_count/training_weeks),0.25,4.0))",
                    "order": "multiply eligible personal repeat scores before per-user candidate-only max normalization",
                    "no_personal_history": "zero remains zero; no invented repeat evidence",
                    "zero_historical_opportunities": "no factor; any positive personal evidence without historical opportunity is an inconsistency"},
         "association": settings["association"], "repeat_weight": .7, "association_weight": .3,
         "normalization": "independent per-user max normalization of each component over eligible candidates",
         "ranking": "exact score descending, primary80 training-user popularity descending, primary80 airing popularity descending, program ID ascending",
         "fallback": "full-history >=80% training popularity restricted to target candidates", "repeats_allowed": True,
         "train_window": {"iso": ["2020-03-09T00:00:00", "2020-07-27T00:00:00"], "epochs": list(TRAIN), "weeks": weeks(TRAIN)},
         "target_window": {"iso": ["2020-07-27T00:00:00", "2020-10-19T00:00:00"], "epochs": list(TARGET), "weeks": weeks(TARGET)},
         "training_schedule_rule": "owned labelled positive airings, deterministic identity dedupe, starts in training window and ends <= cutoff",
         "target_schedule_rule": "owned labelled positive airings, deterministic identity dedupe, starts in target window including end-crossing airings",
         "expected_candidates": 6632, "best_scored_submission": str(BEST.relative_to(ROOT)), "best_sha256": BEST_SHA,
         "output": str(DESTINATION.relative_to(ROOT)), "column_labels": "cropped score-column identities remain unconfirmed"})
    save("audit/prepare_runtime.json", {"workspace_hash_and_configuration_seconds": time.perf_counter()-started})
    checkpoint("fixed_configuration_frozen", unique_configurations=35)


def weeks(window):
    if window[1] <= window[0]:
        raise ValueError("Window must have positive timestamp-derived duration")
    return (window[1]-window[0])/(7*24*60*60)


def factors_for(past_counts, target_counts, train_window, target_window, candidates):
    past_weeks, target_weeks = weeks(train_window), weeks(target_window)
    if train_window[1] > target_window[0]:
        raise ValueError("Training/target overlap would violate forecast cutoff")
    result = {}
    for program in candidates:
        h, t = past_counts.get(program, 0), target_counts.get(program, 0)
        if h < 0 or t < 0:
            raise ValueError("Negative opportunity count")
        if h > 0 and t > 0:
            ratio = (t/target_weeks)/(h/past_weeks)
            result[int(program)] = math.sqrt(min(4., max(.25, ratio)))
    return result


def adjust_repeat_scores(repeat_scores, factors, candidates):
    allowed = set(candidates)
    adjusted = {}
    for user, scores in repeat_scores.items():
        rows = {}
        for program, count in scores.items():
            if program not in allowed or count <= 0:
                continue
            if program not in factors:
                raise ValueError(f"Positive personal repeat evidence without historical/target schedule opportunity: {program}")
            rows[int(program)] = float(count)*factors[program]
        adjusted[user] = rows
    return adjusted


def fit_predict(data, past_counts, target_counts, train_window, target_window):
    """The identical prediction implementation for both reproductions and final refit."""
    started = time.perf_counter()
    factors = factors_for(past_counts, target_counts, train_window, target_window, data["val_candidates"])
    repeat = s6.scores_for(data["cohort"], data["user_to_train_idx"], data["count_variants"]["sens50"], data["val_candidates"])
    adjusted = adjust_repeat_scores(repeat, factors, data["val_candidates"])
    repeat_seconds = time.perf_counter()-started
    ar_start = time.perf_counter()
    association, ar_details = s8.ar_scores(data)
    association_seconds = time.perf_counter()-ar_start
    fusion_start = time.perf_counter()
    from stage4.hybrid import fused_predictions
    predictions, detail = fused_predictions(data["cohort"], adjusted, association, data["val_candidates"],
                data["popularity_val"], data["popularity"], repeat_weight=.7, association_weight=.3)
    runtime = {"factor_repeat_adjust_seconds": repeat_seconds, "association_fit_score_seconds": association_seconds,
               "fusion_rank_seconds": time.perf_counter()-fusion_start, "direct_fit_predict_seconds": time.perf_counter()-started,
               "scope": "directly measured fit/predict using already-loaded counts and opportunity counts; cache loading and schedule parsing excluded"}
    detail["users_without_personal_repeat_candidates"] = sum(not x for x in repeat.values())
    detail["adjustment_preserves_personal_support"] = all(set(repeat[u])==set(adjusted[u]) for u in repeat)
    return predictions, factors, detail, ar_details, runtime


def independent_metrics(predictions, labels, users):
    aps = {}
    for u in users:
        relevant = set(labels.get(u, []))
        seen, hits, terms = set(), 0, []
        for rank, item in enumerate(predictions[u][:5], 1):
            if item in relevant and item not in seen:
                hits += 1
                terms.append(hits/rank)
            seen.add(item)
        aps[u] = math.fsum(terms)/min(5, len(relevant)) if relevant else 0.
    groups = {"map_at_k": users, "map_with_positive": [u for u in users if labels.get(u)],
              "map_with_at_least_k_positive": [u for u in users if len(set(labels.get(u,[])))>=5]}
    metrics = {key: math.fsum(aps[u] for u in group)/len(group) if group else 0. for key,group in groups.items()}
    metrics["cohort_sizes"] = {key: len(group) for key,group in groups.items()}
    return metrics, aps


def identity_set(schedule):
    return set(zip(schedule["channel"].tolist(), schedule["start"].tolist(), schedule["end"].tolist(), schedule["show"].tolist()))


def reproduce():
    if (OUT / "reproduction/equivalence.json").exists():
        assert s6.read_json(OUT / "reproduction/equivalence.json")["all_passed"]
        return
    history = s8.raw_schedule()
    outcomes = {}
    for scope in ("horizon12", "original10"):
        def run(scope=scope):
            started = time.perf_counter()
            data, table, hist, target, _ = s8.build_split(scope, history)
            predicted, factors, detail, ar, runtime = fit_predict(data, Counter(int(x) for x in hist["show"]),
                 Counter(int(x) for x in target["show"]), (s6.START,data["cutoff"]), (data["cutoff"],s6.FINAL_END))
            directory = ROOT / f"outputs/stage8/{scope}/methods/{NAME}"
            old = s6.read_json(directory/"top5.json")
            saved_metrics, old_aps = independent_metrics(old,data["labels"]["primary_top5"],data["cohort"])
            metrics, aps = independent_metrics(predicted,data["labels"]["primary_top5"],data["cohort"])
            expected_metrics = s6.read_json(directory/"metrics.json")
            original_factors = {int(k):v for k,v in s6.read_json(ROOT/f"outputs/stage8/{scope}/schedule_adjustment_factors.json").items()}
            checks = {"top5_exact": predicted==old, "all_user_ap_exact": aps==old_aps,
                      "independent_metrics_match_stage8": all(abs(saved_metrics[k]-expected_metrics[k])<1e-12 for k in
                          ("map_at_k","map_with_positive","map_with_at_least_k_positive")),
                      "factors_exact": factors==original_factors, "training_ends_before_cutoff": bool(np.all(table["end"]<=data["cutoff"]))}
            save(f"reproduction/{scope}/final_implementation_top5.json",predicted)
            save(f"reproduction/{scope}/independent_metrics.json",metrics)
            save(f"reproduction/{scope}/independent_per_user_ap5.json",aps)
            save(f"reproduction/{scope}/runtime.json", {**runtime,"total_scope_seconds": time.perf_counter()-started})
            save(f"reproduction/{scope}/fallback.json",detail)
            save(f"reproduction/{scope}/association_details.json",ar)
            save(f"reproduction/{scope}/factors.json",factors)
            save(f"reproduction/{scope}/checks.json",checks)
            assert all(checks.values()), checks
            outcomes[scope] = {"checks": checks,"map_at_5": metrics["map_at_k"],"users_compared":len(data["cohort"])}
            checkpoint(f"{scope}_exact_reproduction",map_at_5=metrics["map_at_k"])
        attempt(f"reproduce_{scope}",run)
    save("reproduction/equivalence.json", {"all_passed":True,"splits":outcomes,"new_unique_configurations":0})


def read_target_schedule(path, window):
    """Stage 8 ownership/identity/eligibility rules applied to the target file."""
    kept = {}; audit = Counter()
    with path.open(newline="",encoding="utf-8") as handle:
        for number,row in enumerate(csv.DictReader(handle),1):
            audit["raw_rows"] += 1
            show = (row.get("tv_show_id") or "").strip()
            if not show or show=="0":
                audit["unlabelled"] += 1
                continue
            try:
                start=s6.parse_epoch(row["start_time"]); duration=int(float(row["duration"])); program=int(show)
            except (ValueError,KeyError):
                audit["invalid"] += 1
                continue
            if duration<=0 or program<=0:
                audit["nonpositive"] += 1
                continue
            if not window[0]<=start<window[1]:
                audit["spillover_dropped"] += 1
                continue
            identity=(row["channel_id"].strip(),start,start+duration,program)
            if identity in kept:
                audit["identity_duplicates_dropped"] += 1
                continue
            kept[identity]=(identity[0],start,start+duration,duration,program,number)
    rows=sorted(kept.values(),key=lambda r:(r[0],r[1],r[4],r[5]))
    names=("channel","start","end","duration","show","source_row")
    schedule={key:np.asarray([r[i] for r in rows],dtype=str if key=="channel" else np.int64) for i,key in enumerate(names)}
    audit["kept_airings"]=len(rows)
    audit["crossing_target_end_included"]=int(np.sum(schedule["end"]>window[1]))
    return schedule,dict(audit)


def opportunity_summary(past, target, factors):
    records=[]
    train_weeks,target_weeks=weeks(TRAIN),weeks(TARGET)
    for program in sorted(target):
        h,t=past.get(program,0),target[program]
        ratio=(t/target_weeks)/(h/train_weeks) if h>0 else None
        records.append({"tv_show_id":program,"historical_scheduled_airings":h,"target_scheduled_airings":t,
             "historical_rate":h/train_weeks,"target_rate":t/target_weeks,"rate_ratio":ratio,
             "clipped_rate_ratio":min(4.,max(.25,ratio)) if ratio is not None else None,"factor":factors.get(program)})
    save("final/opportunity_counts_and_factors.json",records)
    values=np.array(list(factors.values()))
    summary={"training_weeks":train_weeks,"target_weeks":target_weeks,"historical_airings":sum(past.values()),
             "target_airings":sum(target.values()),"historical_programs":len(past),"target_programs":len(target),
             "target_programs_without_historical_opportunities":sum(past.get(s,0)==0 for s in target),
             "factor_programs":len(factors),"factor_quantile_probabilities":[0,.1,.25,.5,.75,.9,1],
             "factor_quantiles":np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist(),
             "raw_ratio_below_lower_clip":sum(r["rate_ratio"] is not None and r["rate_ratio"]<.25 for r in records),
             "raw_ratio_above_upper_clip":sum(r["rate_ratio"] is not None and r["rate_ratio"]>4 for r in records),
             "source":"recomputed full-history and test opportunities, not validation factors"}
    save("final/opportunity_summary.json",summary)


def full_refit():
    if (OUT/"final/model_provenance.json").exists():
        assert DESTINATION.exists()
        return
    assert s6.read_json(OUT/"reproduction/equivalence.json")["all_passed"]
    assert not DESTINATION.exists(), "Refuse to overwrite a previously generated CSV"
    def run():
        started=time.perf_counter()
        schedule_started=time.perf_counter()
        history=s8.raw_schedule()
        hist=s8.subset(history,TRAIN[1])
        cfg=s6.load_config(ROOT/"config/stage5.json")
        target,target_audit=read_target_schedule(cfg.schedule_files["test"],TARGET)
        cached_hist=s8.cache_schedule(cfg.out("airings","airings_history.csv"))
        cached_target=s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv"))
        candidates=sorted(set(int(x) for x in target["show"]))
        previous_candidates=s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"]
        consistency={"historical_airing_identities_match_stage5":identity_set(hist)==identity_set(cached_hist) and len(hist["show"])==len(cached_hist["show"]),
                     "target_airing_identities_match_stage5":identity_set(target)==identity_set(cached_target) and len(target["show"])==len(cached_target["show"]),
                     "target_candidates_exactly_stage5":candidates==previous_candidates,"target_candidate_count_6632":len(candidates)==6632}
        save("audit/schedule_consistency.json",consistency)
        save("audit/target_schedule_reconstruction.json",target_audit)
        if not all(consistency.values()):
            raise RuntimeError("Schedule implementation inconsistency: stop before writing CSV; report audit/schedule_consistency.json before protocol changes")
        schedule_seconds=time.perf_counter()-schedule_started
        load_started=time.perf_counter()
        variants=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
            template=list(csv.DictReader(handle))
        users=[row["user_id"] for row in template]
        assert len(users)==len(set(users))==1260
        pop_ids,pop_stats,pop_rows=s6.build_popularity(variants,candidates)
        save("final/popularity.json",pop_rows)
        data={"cohort":users,"count_variants":variants,"user_to_train_idx":{u:i for i,u in enumerate(ids)},
              "val_candidates":candidates,"popularity_val":pop_ids,"popularity":pop_stats,"cutoff":TRAIN[1]}
        load_seconds=time.perf_counter()-load_started
        past_counts=Counter(int(x) for x in hist["show"])
        target_counts=Counter(int(x) for x in target["show"])
        predictions,factors,detail,ar,runtime=fit_predict(data,past_counts,target_counts,TRAIN,TARGET)
        opportunity_summary(past_counts,target_counts,factors)
        save("final/factors.json",factors)
        save("final/predictions.json",predictions)
        save("final/fallback.json",detail)
        save("final/association_details.json",ar)
        cold=s6.cold_start_stats(users,data["user_to_train_idx"],variants)
        cold["scope"]="full history, weeks 11-30; not validation training"
        save("final/cold_start.json",cold)
        from stage5.submission import write_submission,validate_submission
        write_submission(DESTINATION,users,predictions)
        submission_checks=validate_submission(DESTINATION,users,candidates)
        assert submission_checks["all_passed"], submission_checks
        save("audit/submission_checks.json",submission_checks)
        with BEST.open(newline="",encoding="utf-8") as handle:
            baseline={r["user_id"]:[int(x) for x in r["tv_show_id"].split()] for r in csv.DictReader(handle)}
        comparison=s8.prediction_changes(predictions,baseline,users)
        comparison.update(distinct_recommended_programs_stage9=len({s for p in predictions.values() for s in p}),
                          distinct_recommended_programs_stage5=len({s for p in baseline.values() for s in p}),
                          stage5_sha256=s6.sha256_file(BEST),stage9_sha256=s6.sha256_file(DESTINATION))
        save("comparison_with_stage5.json",comparison)
        runtime.update(raw_schedule_reconstruction_and_cache_consistency_seconds=schedule_seconds,
                       interaction_cache_loading_and_popularity_refit_seconds=load_seconds,
                       total_full_refit_seconds_including_schedule_cache_io_and_csv=time.perf_counter()-started)
        save("final/runtime.json",runtime)
        save("final/model_provenance.json",{"name":NAME,"new_unique_configurations":0,"unique_configurations":35,
             "submission":str(DESTINATION.relative_to(ROOT)),"submission_sha256":s6.sha256_file(DESTINATION),
             "training_window":list(TRAIN),"target_window":list(TARGET),"training_weeks":weeks(TRAIN),"target_weeks":weeks(TARGET),
             "candidates":len(candidates),"training_airings":len(hist["show"]),"target_airings":len(target["show"]),
             "full_history_counts":str(cfg.out("interactions","history_user_program_counts.csv").relative_to(ROOT)),
             "opportunity_sources":[str(p.relative_to(ROOT)) for p in cfg.schedule_files.values()],
             "validation_factors_reused":False,"future_viewing_behavior_used":False,
             "official_rendered_page_verification":"external Codex review; not personal verification by the human user or a rendered-browser execution in this OpenCode session",
             "score_column_labels":"unconfirmed; official 50/50 user split does not resolve cropped column identities",
             "designated_best_scored_submission":str(BEST.relative_to(ROOT)),"stage9_kaggle_score":None})
        checkpoint("fixed_full_refit_complete",sha256=s6.sha256_file(DESTINATION))
    attempt("full_history_fixed_refit",run)


def verify():
    started=time.perf_counter()
    checks={}
    before=s6.read_json(OUT/"preservation_before.json")
    after=workspace_manifest()
    save("preservation_after.json",after)
    checks["all_prior_workspace_files_and_raw_inputs_byte_preserved"]=before==after
    checks["stage5_best_sha256_preserved"]=s6.sha256_file(BEST)==BEST_SHA
    raw=s6.read_json(ROOT/"outputs/stage5/audit/raw_manifest_after.json")
    checks["raw_inputs_match_verified_stage5"]=all(s6.sha256_file(Path(v["path"]))==v["sha256"] for v in raw.values())
    equivalence=s6.read_json(OUT/"reproduction/equivalence.json")
    for scope, result in equivalence["splits"].items():
        for key,value in result["checks"].items():
            checks[f"{scope}/{key}"]=value
        saved=s6.read_json(OUT/f"reproduction/{scope}/final_implementation_top5.json")
        original=s6.read_json(ROOT/f"outputs/stage8/{scope}/methods/{NAME}/top5.json")
        users=s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json")
        labels=s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json")["primary_top5"]
        metrics,aps=independent_metrics(saved,labels,users)
        old_metrics,_=independent_metrics(original,labels,users)
        checks[f"{scope}/saved_predictions_still_exact"]=saved==original
        checks[f"{scope}/independently_recomputed_metrics"]=metrics==old_metrics
    checks.update(s6.read_json(OUT/"audit/schedule_consistency.json"))
    cfg=s6.load_config(ROOT/"config/stage5.json")
    table,_=s6.load_table(cfg.out("interactions","history_user_airing.npz"),cfg.out("airings","airings_history.csv"))
    counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
    checks["full_history_repeat50_cache_matches_actual_coverage"]=s6.aggregate(table,TRAIN[1],{"threshold":.5,"weighting":"count"})==counts["sens50"]
    checks["full_history_primary80_cache_matches_actual_coverage"]=s6.aggregate(table,TRAIN[1],{"threshold":.8,"weighting":"count"})==counts["primary80"]
    checks["no_future_airing_coverage"]=bool(np.all(table["end"]<=TRAIN[1]))
    view_audit=s6.read_json(cfg.out("audit","history_view_audit.json"))
    checks["no_future_view_observations"]=view_audit["no_future_observations"] and view_audit["max_observed_stop"]<=TRAIN[1]
    boundary=s6.read_json(cfg.out("audit","history_schedule_audit.json"))
    checks["verified_internal_boundary_restoration"]=boundary["restored_may18_crossing_airings"]==66
    checks["verified_final_cutoff_exclusions"]=boundary["excluded_full_history_crossing_end"]==58
    rows=s6.read_json(OUT/"final/opportunity_counts_and_factors.json")
    factors={int(k):v for k,v in s6.read_json(OUT/"final/factors.json").items()}
    # Separate formula check, using timestamp durations rather than cached factors.
    checks["independent_factor_formula"]=all((row["factor"] is None and row["historical_scheduled_airings"]==0) or
         abs(row["factor"]-math.sqrt(max(.25,min(4.,row["target_scheduled_airings"]*(TRAIN[1]-TRAIN[0]) /
             (row["historical_scheduled_airings"]*(TARGET[1]-TARGET[0]))))))<1e-14 for row in rows)
    checks["factor_clip_bounds"]=all(.5<=v<=2 for v in factors.values())
    checks["window_lengths_timestamp_derived_20_and_12"]=weeks(TRAIN)==20 and weeks(TARGET)==12
    with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle); template_headers=reader.fieldnames; users=[r["user_id"] for r in reader]
    from stage5.submission import validate_submission
    submission=validate_submission(DESTINATION,users,s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"])
    for item in submission["checks"]:
        checks[f"submission/{item['check']}"]=item["ok"]
    with DESTINATION.open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle); headers=reader.fieldnames
        csv_predictions={r["user_id"]:[int(x) for x in r["tv_show_id"].split()] for r in reader}
    checks["header_matches_actual_template"]=headers==template_headers
    checks["csv_exactly_matches_saved_predictions"]=csv_predictions==s6.read_json(OUT/"final/predictions.json")
    provenance=s6.read_json(OUT/"final/model_provenance.json")
    checks["submission_sha256_matches_provenance"]=s6.sha256_file(DESTINATION)==provenance["submission_sha256"]
    history=s6.read_json(ROOT/"outputs/stage8/search_history.json")
    checks["unique_configuration_count_unchanged_35"]=history["unique_configurations"]==35 and provenance["new_unique_configurations"]==0
    checks["only_one_new_submission_at_exact_path"]=list(OUT.glob("*.csv"))==[DESTINATION]
    checks["stage5_remains_designated_best"]=provenance["designated_best_scored_submission"]==str(BEST.relative_to(ROOT)) and provenance["stage9_kaggle_score"] is None
    checks["personal_repeat_support_preserved"]=s6.read_json(OUT/"final/fallback.json")["adjustment_preserves_personal_support"]
    attempts=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    checks["no_failed_model_attempts"]=not any(r["event"]=="failed" for r in attempts)
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),
             "checks_total":len(checks),"preserved_workspace_files":len(before),"unique_configurations":35,
             "new_unique_configurations":0,"submission":str(DESTINATION.relative_to(ROOT)),"sha256":s6.sha256_file(DESTINATION),
             "verification_seconds":time.perf_counter()-started}
    save("audit/validation.json",summary)
    source_paths={Path(__file__),ROOT/"outputs/stage8/code/study.py",ROOT/"outputs/stage6/code/study.py"}
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": source_paths.add(path)
    save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(source_paths)})
    save("search_history.json",{"unique_configurations":35,"new_configurations":0,
         "source":"outputs/stage8/search_history.json","source_sha256":s6.sha256_file(ROOT/"outputs/stage8/search_history.json"),
         "operation":"existing B fixed refit and two equivalence reproductions"})
    assert summary["all_passed"],summary
    checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",sha256=summary["sha256"])
    print(json.dumps({"submission":summary["submission"],"sha256":summary["sha256"],"checks":f"{sum(checks.values())}/{len(checks)}",
          "preserved_workspace_files":len(before),"comparison":s6.read_json(OUT/"comparison_with_stage5.json")},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","reproduce","refit","verify","all"))
    phase=parser.parse_args().phase
    if phase in ("prepare","all"): prepare()
    if phase in ("reproduce","all"): reproduce()
    if phase in ("refit","all"): full_refit()
    if phase in ("verify","all"): verify()


if __name__=="__main__": main()
