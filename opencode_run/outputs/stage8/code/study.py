#!/usr/bin/env python3
"""Stage 8: four fixed configurations, two backtests, no submission generation."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("stage6_helpers", ROOT / "outputs/stage6/code/study.py")
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)
OUT = ROOT / "outputs/stage8"
s6.OUT = OUT
np = s6.np
REVIEW = Path("/Users/vladimirkuznecov/Documents/Codex/2026-10-21/new-chat-5/outputs/sweettv_independent_review_20261023.json")
HORIZON_CUTOFF = s6.parse_epoch("2020-05-04T00:00:00")
SEED, RESAMPLES = 137, 10000
CONTROLS = ("repeat_50", "stage5_hybrid_70_30", "submitted_stage6", "submitted_stage7")
NEW = ("A_repeat50_hybrid85", "B_schedule_adjusted_hybrid70", "C_channel_time", "D_hybrid90_schedule10")
EPOCH = datetime(1970, 1, 1)


def save(path, value):
    s6.save(path, value)


def snapshot():
    paths = []
    for stage in range(2, 8):
        paths += [p for p in (ROOT / f"outputs/stage{stage}").rglob("*") if p.is_file()]
    for directory in ("src", "scripts", "tests", "config"):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return {str(p.relative_to(ROOT)): s6.sha256_file(p) for p in sorted(paths)}


def declarations():
    return {
        NEW[0]: {"repeat": "original repeat_50", "association": s6.AR_PARAMS, "repeat_weight": .85, "association_weight": .15},
        NEW[1]: {"repeat": "original repeat_50 times sqrt(clip(future_weekly_airings / past_weekly_airings, .25, 4))",
                 "repeat_weight": .7, "association_weight": .3, "association": s6.AR_PARAMS,
                 "zero_personal_repeat_remains_zero": True},
        NEW[2]: {"training_interactions": ">=80% actual coverage", "time_bins": "weekday/weekend x six 4-hour start bins",
                 "smoothing": 20, "p_uc": "watched_airings_uc / scheduled_airings_c, zero if unavailable",
                 "p_ucb": "(watched_airings_ucb + 20*p_uc) / (scheduled_airings_cb + 20)",
                 "target_score": "sum p_ucb over program target airings", "interpretation": "forecasting heuristic, not calibrated probability"},
        NEW[3]: {"original_stage5_hybrid_weight": .9, "scheduling_model_C_weight": .1,
                 "normalization": "per-user max-normalize original hybrid score and C score before fusion",
                 "prerequisite": "C individually evaluated first"},
    }


def prepare():
    if (OUT / "plan.json").exists():
        return
    save("preservation_before.json", snapshot())
    save("plan.json", {"predeclared_before_stage8_evaluation": True, "new_configurations": declarations(),
         "planned_new_unique": 4, "previous_unique": 31, "combined_unique": 35,
         "controls": list(CONTROLS), "stage5_designated_best": True,
         "splits": {"horizon12": {"train": ["2020-03-09", "2020-05-04"], "validation": ["2020-05-04", "2020-07-27"]},
                    "original10": {"train": ["2020-03-09", "2020-05-18"], "validation": ["2020-05-18", "2020-07-27"]}},
         "coverage": "actual unioned seconds/original duration, no gap bridging",
         "schedule_rules": "nominal start ownership, identity dedupe by channel/start/end/program, first source then source row wins; labelled positive duration only",
         "boundaries": "interaction airings must end by cutoff; target schedule includes crossing-validation-end airings",
         "C_time_convention": "naive schedule timestamps as supplied, no invented timezone conversion; weekday Mon-Fri, weekend Sat-Sun; floor(start_hour/4)",
         "fallback_and_ranking": s6.fusion_rules(), "repeats_allowed": True,
         "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "interval": "paired-user percentile 95%, all 3 cohorts",
                       "caveat": "conditional descriptive, not selection-adjusted; overlapping historical backtests"},
         "no_additional_configurations": True, "no_final_refit_or_submission_generation": True,
         "recommendation_rule": "tiny gains alone do not warrant refitting; compare both splits, effect size, uncertainty and model selection"})
    save("external_review_attribution.json", {"source": "external Codex diagnostic review", "path": str(REVIEW),
         "sha256": s6.sha256_file(REVIEW), "external_computations_are_not_this_session_executions": True,
         "official_page_access_in_this_session": "direct fetch returned title/shell only; rendered facts attributed to user and external review"})
    save("current_best_submission.json", s6.read_json(ROOT / "outputs/stage7/current_best_submission.json"))
    s6.checkpoint("predeclared_four", previous_unique=31, new_unique=4)


def bins_for(epochs):
    return np.array([(int((EPOCH + timedelta(seconds=int(t))).weekday() >= 5) * 6
                      + (EPOCH + timedelta(seconds=int(t))).hour // 4) for t in epochs], dtype=np.int16)


def raw_schedule():
    """Read both raw files independently; do not inherit the weeks21-30 pool."""
    cfg = s6.load_config(ROOT / "config/stage5.json")
    kept, audit = {}, {}
    for priority, name in enumerate(("history_11_20", "history_21_30")):
        bounds = cfg.raw["schedule_windows"][name]
        lo, hi = map(s6.parse_epoch, bounds)
        stats = Counter()
        with cfg.schedule_files[name].open(newline="", encoding="utf-8") as handle:
            for number, raw in enumerate(csv.DictReader(handle), 1):
                stats["raw_rows"] += 1
                show = (raw.get("tv_show_id") or "").strip()
                if not show or show == "0":
                    continue
                stats["labelled_rows"] += 1
                try:
                    start = s6.parse_epoch(raw["start_time"])
                    duration = int(float(raw["duration"]))
                    program = int(show)
                except (ValueError, KeyError):
                    stats["invalid"] += 1
                    continue
                if duration <= 0 or program <= 0:
                    stats["nonpositive"] += 1
                    continue
                if not lo <= start < hi:
                    stats["spillover_dropped"] += 1
                    continue
                channel = raw["channel_id"].strip()
                identity = (channel, start, start+duration, program)
                if identity in kept:
                    stats["identity_duplicate_dropped"] += 1
                    continue
                kept[identity] = (channel, start, start+duration, duration, program, priority, number)
                stats["owned_labelled_positive_rows"] += 1
        audit[name] = dict(stats)
    rows = sorted(kept.values(), key=lambda r: (r[0], r[1], r[4], r[5], r[6]))
    names = ("channel", "start", "end", "duration", "show", "source", "source_row")
    schedule = {name: np.asarray([row[i] for row in rows], dtype=str if name == "channel" else np.int64)
                for i, name in enumerate(names)}
    schedule["bin"] = bins_for(schedule["start"])
    save("audit/raw_schedule_reconstruction.json", {"sources": audit, "labelled_kept": len(rows),
         "identity": "channel/start/end/program", "source_priority": "history_11_20 before history_21_30 then row"})
    return schedule


def subset(schedule, cutoff, target=False):
    mask = ((schedule["start"] >= cutoff) & (schedule["start"] < s6.FINAL_END) if target else
            (schedule["start"] >= s6.START) & (schedule["start"] < cutoff) & (schedule["end"] <= cutoff))
    return {key: value[mask] for key, value in schedule.items()}


def cache_schedule(path):
    columns = {k: [] for k in ("channel", "start", "end", "duration", "show")}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, field in (("channel", "channel_id"), ("start", "start_epoch"), ("end", "end_epoch"), ("duration", "duration"), ("show", "tv_show_id")):
                columns[key].append(row[field] if key == "channel" else int(row[field]))
    result = {k: np.asarray(v) for k,v in columns.items()}
    result["bin"] = bins_for(result["start"])
    return result


def build_split(scope, schedule):
    cutoff = HORIZON_CUTOFF if scope == "horizon12" else s6.PRIMARY_END
    target = subset(schedule, cutoff, target=True)
    hist = subset(schedule, cutoff)
    cfg, full, _, ids = s6.full_data()
    if scope == "original10":
        data = s6.load_stage4_data(s6.load_config(ROOT / "config/stage3.json"))
        table, _ = s6.load_table(ROOT / "outputs/stage3/interactions/train_user_airing.npz", ROOT / "outputs/stage3/airings/airings_train.csv")
        interaction_schedule = cache_schedule(ROOT / "outputs/stage3/airings/airings_train.csv")
        assert set(target["show"]) == set(data["val_candidates"])
    else:
        table = s6.slice_table(full, s6.START, cutoff)
        interaction_schedule = cache_schedule(cfg.out("airings", "airings_history.csv"))
        variants = s6.variants_from_table(table, cutoff)
        val = s6.slice_table(full, cutoff, s6.FINAL_END)
        val80 = s6.aggregate(val, s6.FINAL_END, {"threshold": .8, "weighting": "count"})
        users = s6.read_json(ROOT / "outputs/stage6/primary/cohort.json")
        index = {u:i for i,u in enumerate(ids)}
        labels = {u: sorted(val80.get(index.get(u), {}), key=lambda s: (-val80[index[u]][s], s))[:5] for u in users}
        with np.load(cfg.out("views", "views_history.npz")) as views:
            eligible = set(int(x) for x in np.unique(views["user_idx"][(views["start"] < cutoff) & (views["stop"] > s6.START)]))
        mapping = {u:i for i,u in enumerate(ids) if i in eligible}
        candidates = sorted(set(int(x) for x in target["show"]))
        pop_ids, pop_stats, pop_rows = s6.build_popularity(variants, candidates)
        data = {"cohort": users, "count_variants": variants, "labels": {"primary_top5": labels},
                "user_to_train_idx": mapping, "val_candidates": candidates, "popularity_val": pop_ids, "popularity": pop_stats}
        save(f"{scope}/popularity.json", pop_rows)
        save(f"{scope}/cold_start.json", s6.cold_start_stats(users, mapping, variants))
        # Separate integer comparison reconstruction verifies newly built labels.
        counts = defaultdict(Counter)
        valid = ((full["start"] >= cutoff) & (full["end"] <= s6.FINAL_END)
                 & (full["covered_sec"] * 5 >= full["airing_duration"] * 4))
        for ui, si in zip(full["user_idx"][valid], full["show_id"][valid]):
            counts[ids[int(ui)]][int(si)] += 1
        independent = {u: [s for s,_ in sorted(counts[u].items(), key=lambda x: (-x[1],x[0]))[:5]] for u in users}
        assert labels == independent
    data["cutoff"] = cutoff
    assert np.all(table["end"] <= cutoff)
    assert len(data["cohort"]) == 1260
    assert set(data["val_candidates"]) == set(target["show"])
    assert len(hist["show"]) == int(np.sum((interaction_schedule["start"]>=s6.START) & (interaction_schedule["end"]<=cutoff)))
    save(f"{scope}/labels.json", data["labels"])
    save(f"{scope}/cohort.json", data["cohort"])
    save(f"{scope}/candidates.json", data["val_candidates"])
    extra = set(target["show"][target["source"] == 0]) - set(target["show"][target["source"] == 1])
    save(f"{scope}/provenance.json", {"cutoff": cutoff, "training_window": [s6.START, cutoff],
         "validation_window": [cutoff, s6.FINAL_END], "training_weeks": (cutoff-s6.START)/604800,
         "validation_weeks": (s6.FINAL_END-cutoff)/604800,
         "train_airings": len(hist["show"]), "target_airings": len(target["show"]), "candidate_count": len(data["val_candidates"]),
         "target_airings_by_source": {"history_11_20": int(np.sum(target["source"]==0)), "history_21_30": int(np.sum(target["source"]==1))},
         "candidate_ids_only_in_weeks19_20": sorted(int(x) for x in extra),
         "max_training_observation_airing_end": int(table["end"].max()), "interaction_rows": len(table["user_idx"]),
         "cache_validity": "unioned coverage for fully contained airings is invariant to clipping views at split boundaries",
         "label_rule": ">=80% actual coverage; top5 by qualifying-airing frequency descending then ID ascending; fewer if fewer qualify",
         "description": "new overlapping retrospective backtest, not independent untouched test" if scope=="horizon12" else "unchanged original frozen 10-week validation"})
    return data, table, hist, target, interaction_schedule


def normalize(scores):
    maximum = max(scores.values(), default=0)
    return {s: v/maximum for s,v in scores.items() if v>0} if maximum>0 else {}


def blend(left, right, weight, right_weight=None):
    if right_weight is None:
        right_weight = 1-weight
    result = {s: weight*v for s,v in normalize(left).items()}
    for s,v in normalize(right).items():
        result[s] = result.get(s, 0) + right_weight*v
    return result


def schedule_factors(hist, target, past_weeks, future_weeks):
    past, future = Counter(int(x) for x in hist["show"]), Counter(int(x) for x in target["show"])
    return {s: math.sqrt(min(4., max(.25, (n/future_weeks)/(past[s]/past_weeks))))
            for s,n in future.items() if past[s]>0}


def channel_time(table, interaction_schedule, hist, target, data):
    from scipy.sparse import coo_matrix
    channels = sorted(set(hist["channel"]) | set(target["channel"]))
    channel_index = {c:i for i,c in enumerate(channels)}
    users = data["cohort"]
    user_index = {data["user_to_train_idx"][u]: i for i,u in enumerate(users) if u in data["user_to_train_idx"]}
    scheduled = np.zeros((len(channels), 12), dtype=float)
    for channel, bin_id in zip(hist["channel"], hist["bin"]):
        scheduled[channel_index[channel], int(bin_id)] += 1
    watched = np.zeros((len(users), len(channels), 12), dtype=float)
    qualifying = table["covered_sec"] * 5 >= table["airing_duration"] * 4
    for user, airing in zip(table["user_idx"][qualifying], table["airing_idx"][qualifying]):
        ui = user_index.get(int(user))
        if ui is not None:
            channel = interaction_schedule["channel"][int(airing)]
            bin_id = interaction_schedule["bin"][int(airing)]
            watched[ui, channel_index[channel], int(bin_id)] += 1
    denominators = scheduled.sum(axis=1)
    puc = np.divide(watched.sum(axis=2), denominators[None,:],
                    out=np.zeros((len(users),len(channels))), where=denominators[None,:]>0)
    probabilities = (watched + 20*puc[:,:,None])/(scheduled[None,:,:]+20)
    assert np.all(puc<=1+1e-12) and np.all(probabilities<=1+1e-12)
    candidate_index = {s:i for i,s in enumerate(data["val_candidates"])}
    rows = [channel_index[c]*12+int(b) for c,b in zip(target["channel"],target["bin"])]
    cols = [candidate_index[int(s)] for s in target["show"]]
    matrix = coo_matrix((np.ones(len(rows)), (rows,cols)), shape=(len(channels)*12,len(candidate_index))).tocsr()
    scores = matrix.T.dot(probabilities.reshape(len(users),-1).T).T
    diagnostics = {"channels": len(channels), "scheduled_training_airings": int(scheduled.sum()),
         "watched_qualifying_cohort_airings": int(watched.sum()), "target_airings": len(rows), "smoothing": 20,
         "future_viewing_used": False, "maximum_p_uc": float(puc.max()), "maximum_p_ucb": float(probabilities.max()),
         "interpretation": "forecasting heuristic; program sums are not calibrated probabilities"}
    return scores, diagnostics


def ar_scores(data):
    from stage3.association import build_association_index
    from stage4.association import association_scores
    ar = {"cohort": data["cohort"], "user_to_train_idx": data["user_to_train_idx"],
          "user_items": data["count_variants"]["primary80"], "val_candidates": data["val_candidates"]}
    keys = ("min_count", "min_confidence", "min_lift", "max_rules_per_antecedent", "score_scheme", "aggregate", "max_antecedents_per_user")
    return association_scores(ar, build_association_index(ar), **{k:s6.AR_PARAMS[k] for k in keys})


def paired_bootstrap(deltas, seed=SEED, resamples=RESAMPLES):
    if deltas.ndim==1:
        deltas = deltas[:,None]
    if len(deltas)==0:
        return [[0.,0.] for _ in range(deltas.shape[1])]
    rng = np.random.default_rng(seed)
    sampled = np.empty((resamples,deltas.shape[1]))
    for i in range(0,resamples,250):
        indices = rng.integers(0,len(deltas),size=(min(250,resamples-i),len(deltas)))
        sampled[i:i+len(indices)] = deltas[indices].mean(axis=1)
    return np.quantile(sampled,[.025,.975],axis=0).T.tolist()


def prediction_changes(predictions, baseline, users):
    return {"users_changed_list": sum(predictions[u]!=baseline[u] for u in users),
            "users_changed_top1": sum(predictions[u][0]!=baseline[u][0] for u in users),
            "mean_common_top5_ids": sum(len(set(predictions[u])&set(baseline[u])) for u in users)/len(users)}


def evaluate_split(scope, schedule):
    if (OUT / f"{scope}/results.json").exists():
        return
    started = time.perf_counter()
    data, table, hist, target, interaction_schedule = build_split(scope,schedule)
    t = time.perf_counter()
    association, ar_details = ar_scores(data)
    ar_seconds = time.perf_counter()-t
    save(f"{scope}/association_details.json", ar_details)
    repeat = s6.scores_for(data["cohort"],data["user_to_train_idx"],data["count_variants"]["sens50"],data["val_candidates"])
    hybrid = {u: blend(repeat[u],association[u],.7,.3) for u in data["cohort"]}
    saved_predictions, results, runtime_components = {}, [], {"association_fit_seconds": ar_seconds}

    def record(name, scores, category, parameters, start_time, component_seconds=0.):
        predictions, details = s6.rank_scores(scores,data)
        elapsed = time.perf_counter()-start_time
        if scope=="original10" and name in CONTROLS:
            old = {"repeat_50": ROOT/"outputs/stage4/methods/repeat_50/top5.json",
                   "stage5_hybrid_70_30": ROOT/"outputs/stage4/methods/hybrid_70_30/top5.json",
                   "submitted_stage6": ROOT/"outputs/stage6/primary/methods/hybrid_repeat_fraction20_h28_w85/top5.json",
                   "submitted_stage7": ROOT/"outputs/stage7/primary/methods/hybrid_repeat_fraction20_nodecay_w85/top5.json"}[name]
            assert predictions==s6.read_json(old), f"Control reproduction mismatch: {name}"
        result = s6.persist_result(scope,name,predictions,data,parameters,details,
                     {"direct_score_and_rank_seconds": elapsed, "reused_component_fit_seconds": component_seconds,
                      "timing_scope": "component reuse explicitly reported, no standalone end-to-end claim"},category)
        saved_predictions[name] = predictions
        results.append(result)
        return predictions

    for name in CONTROLS:
        def run(name=name):
            t = time.perf_counter()
            if name=="repeat_50":
                return record(name,repeat,"fixed_control_refit",{"repeat_threshold":.5,"weighting":"count","half_life_days":None},t)
            if name=="stage5_hybrid_70_30":
                return record(name,hybrid,"fixed_control_refit",s6.HYBRID_PARAMS,t,ar_seconds)
            params = s6.read_json(ROOT / ("outputs/stage6/final/configuration.json" if name=="submitted_stage6" else "outputs/stage7/final/configuration.json"))["parameters"]
            items = s6.aggregate(table,data["cutoff"],params["repeat"])
            scores = s6.scores_for(data["cohort"],data["user_to_train_idx"],items,data["val_candidates"])
            fused = {u:blend(scores[u],association[u],.85) for u in data["cohort"]}
            return record(name,fused,"fixed_control_refit",params,t,ar_seconds)
        s6.attempted(scope,name,"fixed_control_refit_not_new",run)

    def run_a():
        t=time.perf_counter()
        scores={u:blend(repeat[u],association[u],.85,.15) for u in data["cohort"]}
        return record(NEW[0],scores,"new_unique" if scope=="horizon12" else "repeated_backtest_fit",declarations()[NEW[0]],t,ar_seconds)
    s6.attempted(scope,NEW[0],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run_a)

    def run_b():
        t=time.perf_counter()
        past_weeks=(data["cutoff"]-s6.START)/604800
        future_weeks=(s6.FINAL_END-data["cutoff"])/604800
        factors=schedule_factors(hist,target,past_weeks,future_weeks)
        assert all(s in factors for user in repeat.values() for s in user)
        adjusted={u:{s:v*factors[s] for s,v in repeat[u].items()} for u in data["cohort"]}
        scores={u:blend(adjusted[u],association[u],.7,.3) for u in data["cohort"]}
        save(f"{scope}/schedule_adjustment_factors.json",factors)
        return record(NEW[1],scores,"new_unique" if scope=="horizon12" else "repeated_backtest_fit",declarations()[NEW[1]],t,ar_seconds)
    s6.attempted(scope,NEW[1],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run_b)

    c_matrix=None
    def run_c():
        nonlocal c_matrix
        t=time.perf_counter()
        c_matrix, details=channel_time(table,interaction_schedule,hist,target,data)
        fit_time=time.perf_counter()-t
        runtime_components["channel_time_fit_predict_seconds"]=fit_time
        save(f"{scope}/channel_time_details.json",details)
        predictions, fallback, slots, scored = {},0,0,0
        for i,u in enumerate(data["cohort"]):
            scores={s:float(v) for s,v in zip(data["val_candidates"],c_matrix[i]) if v>0}
            top,info=s6.top_k(scores,data["val_candidates"],data["popularity_val"],data["popularity"])
            predictions[u]=top
            fallback+=int(info["used_fallback"]); slots+=info["fallback_filled"]; scored+=info["scored_candidates"]
        result=s6.persist_result(scope,NEW[2],predictions,data,declarations()[NEW[2]],
             {"users_needing_fallback":fallback,"fallback_slots":slots,"mean_scored_candidates_per_user":scored/len(predictions)},
             {"direct_fit_predict_rank_seconds":time.perf_counter()-t},"new_unique" if scope=="horizon12" else "repeated_backtest_fit")
        saved_predictions[NEW[2]]=predictions; results.append(result)
    s6.attempted(scope,NEW[2],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run_c)

    def run_d():
        assert NEW[2] in saved_predictions
        t=time.perf_counter()
        # Stream each user's scores to avoid retaining a large duplicate score map.
        predictions, fallback, slots, scored = {},0,0,0
        for i,u in enumerate(data["cohort"]):
            c={s:float(v) for s,v in zip(data["val_candidates"],c_matrix[i]) if v>0}
            fused=blend(hybrid[u],c,.9,.1)
            top,info=s6.top_k(fused,data["val_candidates"],data["popularity_val"],data["popularity"])
            predictions[u]=top
            fallback+=int(info["used_fallback"]); slots+=info["fallback_filled"]; scored+=info["scored_candidates"]
        result=s6.persist_result(scope,NEW[3],predictions,data,declarations()[NEW[3]],
             {"users_needing_fallback":fallback,"fallback_slots":slots,"mean_scored_candidates_per_user":scored/len(predictions)},
             {"direct_fusion_rank_seconds":time.perf_counter()-t,"reused_components":runtime_components},
             "new_unique" if scope=="horizon12" else "repeated_backtest_fit")
        saved_predictions[NEW[3]]=predictions; results.append(result)
    s6.attempted(scope,NEW[3],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run_d)

    labels=data["labels"]["primary_top5"]
    # Use the evaluator's already saved user APs here; verify() recomputes them independently.
    aps={}
    for name in saved_predictions:
        per_user=s6.read_json(OUT/f"{scope}/methods/{name}/per_user_ap5.json")
        aps[name]=np.array([per_user[u] for u in data["cohort"]])
    names=list(saved_predictions)
    groups={"all_users":np.ones(len(data["cohort"]),dtype=bool),
            "positive":np.array([len(labels[u])>0 for u in data["cohort"]]),
            "at_least_5":np.array([len(labels[u])>=5 for u in data["cohort"]])}
    baseline=aps["stage5_hybrid_70_30"]
    bootstrap={}
    for group,mask in groups.items():
        deltas=np.column_stack([aps[name]-baseline for name in names])[mask]
        intervals=paired_bootstrap(deltas)
        bootstrap[group]={name:{"delta_map":float(deltas[:,i].mean()),"paired_95_interval":intervals[i],
                   "cohort_size":int(mask.sum()),"users_better":int(np.sum(deltas[:,i]>1e-14)),
                   "users_worse":int(np.sum(deltas[:,i]<-1e-14)),"users_tied":int(np.sum(abs(deltas[:,i])<=1e-14))}
             for i,name in enumerate(names)}
    save(f"{scope}/paired_bootstrap.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":bootstrap,
         "caveat":"conditional descriptive paired-user uncertainty, not model-selection-adjusted or independent evidence"})
    save(f"{scope}/prediction_changes.json",{name:prediction_changes(pred,saved_predictions["stage5_hybrid_70_30"],data["cohort"])
         for name,pred in saved_predictions.items()})
    save(f"{scope}/runtime.json",{"total_split_seconds":time.perf_counter()-started, "reused_components":runtime_components})
    save(f"{scope}/results.json",results)
    s6.checkpoint(f"{scope}_complete",new_configs=4,controls=4)


def independent_ap(predictions,relevant):
    relevant=set(relevant)
    if not relevant: return 0.
    hits=0; values=[]; seen=set()
    for rank,item in enumerate(predictions[:5],1):
        if item in relevant and item not in seen:
            hits+=1; values.append(hits/rank)
        seen.add(item)
    return math.fsum(values)/min(5,len(relevant))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","evaluate","all"))
    args=parser.parse_args()
    if args.phase in ("prepare","all"): prepare()
    if args.phase in ("evaluate","all"):
        assert (OUT/"plan.json").exists()
        schedule=raw_schedule()
        evaluate_split("horizon12",schedule)
        evaluate_split("original10",schedule)


if __name__=="__main__": main()
