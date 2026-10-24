#!/usr/bin/env python3
"""Six fixed refinements of Stage 9; decision frozen before evaluation."""
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

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("stage9_readonly",ROOT/"outputs/stage9/code/refit.py")
s9=importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)
s8=s9.s8; s6=s9.s6; np=s9.np
OUT=ROOT/"outputs/stage10"
s9.OUT=OUT/"cache"; s8.OUT=OUT/"cache"; s6.OUT=OUT/"cache"
BEST=ROOT/"outputs/stage9/kuznetsov_vladimir_midterm_submission_schedule.csv"
BEST_SHA="f88ac045cd9c4347e72fd7a3a90f4c97748d44a055af8faf5c93a684fa0f9ea9"
DEST=OUT/"kuznetsov_vladimir_midterm_submission_refined.csv"
CONTROL={"name":"stage9_control","alpha":.5,"repeat_weight":.7,"association_weight":.3}
CANDIDATES=[
    {"name":"alpha025_w70","alpha":.25,"repeat_weight":.7,"association_weight":.3},
    {"name":"alpha075_w70","alpha":.75,"repeat_weight":.7,"association_weight":.3},
    {"name":"alpha100_w70","alpha":1.,"repeat_weight":.7,"association_weight":.3},
    {"name":"alpha050_w50","alpha":.5,"repeat_weight":.5,"association_weight":.5},
    {"name":"alpha050_w85","alpha":.5,"repeat_weight":.85,"association_weight":.15},
    {"name":"alpha050_w100","alpha":.5,"repeat_weight":1.,"association_weight":0.},
]
SEED,RESAMPLES=137,10000


def save(path,value):
    path=OUT/path; path.parent.mkdir(parents=True,exist_ok=True); s6.write_json(path,value)


def checkpoint(phase,**details):
    value={"phase":phase,"time_epoch":time.time(),**details}
    save("checkpoints/progress.json",value); save(f"checkpoints/{time.time_ns()}_{phase}.json",value)


def manifest():
    return {str(p.relative_to(ROOT)):{"sha256":s6.sha256_file(p),"bytes":p.stat().st_size}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_relative_to(OUT)}


def attempt(scope,name,category,function):
    def log(event,**details):
        with (OUT/"attempts.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps({"scope":scope,"name":name,"category":category,"event":event,"time_epoch":time.time(),**details},sort_keys=True)+"\n")
    log("started")
    try: result=function()
    except Exception:
        log("failed",traceback=traceback.format_exc()); checkpoint("failed",scope=scope,name=name); raise
    log("completed"); return result


def signature(params):
    return {"architecture":"schedule_adjusted_repeat50_ar80_max","alpha":params["alpha"],
            "repeat_weight":params["repeat_weight"],"association_weight":params["association_weight"],
            "clip_ratio":[.25,4.],"repeat_coverage":.5,"decay":None,"association":s6.AR_PARAMS}


def prepare():
    if (OUT/"declaration.json").exists(): return
    assert s6.sha256_file(BEST)==BEST_SHA
    history=s6.read_json(ROOT/"outputs/stage8/search_history.json")
    assert history["unique_configurations"]==35
    save("preservation_before.json",manifest())
    # Inspect all preserved parameter records plus complete later declarations.
    # Only B has this adjusted-repeat architecture in the 35-entry catalogue.
    inspected=[]; previous=[]
    for row in history["configurations"]:
        path=ROOT/row["parameters_path"]; params=s6.read_json(path)
        adjusted=(row["name"]=="B_schedule_adjusted_hybrid70")
        inferred=signature(CONTROL) if adjusted else None
        inspected.append({"name":row["name"],"path":row["parameters_path"],"sha256":s6.sha256_file(path),
             "same_adjusted_repeat_architecture":adjusted,"canonical_signature":inferred})
        if adjusted: previous.append({"signature":inferred,"original_stage":8,"name":row["name"]})
    declaration_paths=[ROOT/f"outputs/stage{s}/{f}" for s,f in ((6,"plan.json"),(7,"plan.json"),(8,"plan.json"),(9,"configuration.json"))]
    prior_declarations={str(p.relative_to(ROOT)):{"sha256":s6.sha256_file(p),"declaration":s6.read_json(p)} for p in declaration_paths}
    duplicates={p["name"]:[r for r in previous if r["signature"]==signature(p)] for p in CANDIDATES}
    save("duplicate_audit.json",{"inspected_configurations":inspected,"prior_declarations":prior_declarations,"exact_duplicates":duplicates,
         "architecture_note":"Earlier unadjusted repeats/hybrids, coverage-fraction/decay variants and channel-time fusions differ. The only previous schedule-adjusted repeat50+AR configuration is B at alpha=.5, 70/30."
         ,"replace_duplicate_slot":False})
    save("declaration.json",{"declared_before_stage10_evaluation":True,"feedback_informed_followup":True,
         "not_originally_preplanned_or_independently_confirmed":True,"kaggle_scores_are_not_optimization_targets":True,
         "control":CONTROL,"six_configurations":CANDIDATES,"previous_unique":35,"maximum_total_unique":41,
         "new_unique":sum(not duplicates[p["name"]] for p in CANDIDATES),
         "formula":"clip((target_count/target_weeks)/(historical_count/training_weeks),.25,4) ** alpha",
         "factor_bounds":"ratio clipped before exponent; no post-exponent clipping to [.5,2]",
         "fixed":{"repeat_coverage":.5,"repeat_weighting":"qualifying-airing counts","time_decay":None,
                  "association":s6.AR_PARAMS,"component_normalization":"independent per-user max over eligible candidates after opportunity adjustment",
                  "zero_history":"zero remains zero; pure repeat uses established popularity fallback",
                  "fallback_ranking":s6.fusion_rules(),"repeats_allowed":True},
         "splits":{"horizon12":"unchanged Stage 8 8-week train / 12-week forecast",
                   "original10":"unchanged frozen Stage 8 original 10-week validation"},
         "reuse":"one shared AR fit and repeat50 map per split; reuse alpha=.5 factors across weight variants; exact prior duplicates reuse predictions when available",
         "uncertainty":{"method":"paired-user percentile descriptive 95% intervals, all three cohorts","seed":SEED,"resamples":RESAMPLES,
                        "user_order":"unchanged template order","caveat":"overlapping periods, repeated selection, selection-unadjusted; not independent confirmation"},
         "selection_rule":{"minimum_gain_each_split":.002,"horizon12_interval_lower_strictly_above":0.,
                 "objective":"largest minimum MAP@5 gain over the two splits",
                 "tie_break_1":"fewer independent parameters changed from Stage 9; fusion weights are one independent parameter",
                 "tie_break_2":"lower summed actual end-to-end computational cost across both splits, measured by fresh exact reproduction fits only for exact-tied finalists",
                 "remaining_exact_cost_tie":"retain Stage 9 rather than invent an additional selection rule",
                 "interpretation":"practical conservative screening rule, not independent statistical confirmation"},
         "refit":"only one eligible winner; no CSV if none qualifies","output":str(DEST.relative_to(ROOT))})
    save("score_evidence.json",{"source":"user-reported latest screenshot confirms column identities",
         "stage9":{"private":.41656,"public":.43185,"submission":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA},
         "stage5":{"private":.40974,"public":.42586,"submission":"outputs/stage5/kuznetsov_vladimir_midterm_submission.csv"},
         "historical_artifacts_modified":False,"used_as_optimization_targets":False,
         "interpretation":"returned feedback motivates this follow-up; not preplanned or independent confirmation"})
    save("current_best_submission.json",{"stage":9,"path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA,
         "private":.41656,"public":.43185,"designation":"best scored, retained until a later returned score establishes otherwise"})
    checkpoint("declaration_frozen",new_unique=sum(not duplicates[p["name"]] for p in CANDIDATES))


def factors_for(past,target,train_window,target_window,candidates,alpha):
    hw,tw=s9.weeks(train_window),s9.weeks(target_window)
    if train_window[1]>target_window[0]: raise ValueError("Training/target overlap")
    factors={}
    for s in candidates:
        h,t=past.get(s,0),target.get(s,0)
        if h>0 and t>0:
            clipped=min(4.,max(.25,(t/tw)/(h/hw)))
            # Keep control arithmetic identical to the verified Stage 9 sqrt.
            factors[int(s)]=math.sqrt(clipped) if alpha==.5 else clipped**alpha
    return factors


def predict(data,repeat,association,factors,params):
    started=time.perf_counter()
    adjusted=s9.adjust_repeat_scores(repeat,factors,data["val_candidates"])
    rw,aw=params["repeat_weight"],params["association_weight"]
    predictions={}; fallback_users=fallback_slots=scored=0
    for u in data["cohort"]:
        fused={s:rw*v for s,v in s8.normalize(adjusted[u]).items()} if rw>0 else {}
        if aw>0:
            for s,v in s8.normalize(association[u]).items(): fused[s]=fused.get(s,0)+aw*v
        top,info=s6.top_k(fused,data["val_candidates"],data["popularity_val"],data["popularity"])
        predictions[u]=top; fallback_users+=int(info["used_fallback"]); fallback_slots+=info["fallback_filled"]; scored+=info["scored_candidates"]
    return predictions,{"users_needing_fallback":fallback_users,"fallback_slots":fallback_slots,
                       "mean_scored_candidates_per_user":scored/len(predictions),
                       "users_without_personal_repeat_candidates":sum(not repeat[u] for u in data["cohort"]),
                       "zero_weight_association_contributes_no_items":aw==0},time.perf_counter()-started


def load_split(scope,history=None,need_ar=True):
    started=time.perf_counter()
    if history is None: history=s8.raw_schedule()
    data,table,hist,target,_=s8.build_split(scope,history)
    frozen=ROOT/f"outputs/stage8/{scope}"
    assert data["labels"]==s6.read_json(frozen/"labels.json")
    assert data["cohort"]==s6.read_json(frozen/"cohort.json")
    assert data["val_candidates"]==s6.read_json(frozen/"candidates.json")
    assert np.all(table["end"]<=data["cutoff"])
    repeat=s6.scores_for(data["cohort"],data["user_to_train_idx"],data["count_variants"]["sens50"],data["val_candidates"])
    ar_start=time.perf_counter()
    if need_ar: ar,details=s8.ar_scores(data); ar_seconds=time.perf_counter()-ar_start
    else: ar={u:{} for u in data["cohort"]}; details={"not_fitted":"zero association weight"}; ar_seconds=0.
    return data,repeat,ar,Counter(int(x) for x in hist["show"]),Counter(int(x) for x in target["show"]),{
         "training_cache_schedule_statistics_and_AR_seconds":time.perf_counter()-started,"shared_AR_fit_score_seconds":ar_seconds,
         "AR_details":details,"max_training_airing_end":int(table["end"].max())}


def write_result(scope,params,predictions,data,fallback,runtime,category):
    base=f"{scope}/methods/{params['name']}"
    save(f"{base}/top5.json",predictions)
    # AP and all cohort metrics are computed from the saved top5 file, separately from ranking.
    metrics,aps=s9.independent_metrics(s6.read_json(OUT/f"{base}/top5.json"),data["labels"]["primary_top5"],data["cohort"])
    save(f"{base}/per_user_ap5.json",aps); save(f"{base}/metrics.json",metrics)
    save(f"{base}/parameters.json",{**params,"canonical_signature":signature(params)})
    save(f"{base}/fallback.json",fallback); save(f"{base}/runtime.json",runtime)
    save(f"{base}/provenance.json",{"category":category,"training_cutoff":data["cutoff"],
         "declaration":"declaration.json","source_manifest":"preservation_before.json","future_viewing_used":False})
    return {"name":params["name"],"map_at_5":metrics["map_at_k"],"category":category}


def evaluate():
    assert (OUT/"declaration.json").exists()
    if (OUT/"results.json").exists(): return
    history=s8.raw_schedule(); all_results={}
    duplicates=s6.read_json(OUT/"duplicate_audit.json")["exact_duplicates"]
    for scope in ("horizon12","original10"):
        if (OUT/f"{scope}/results.json").exists():
            all_results[scope]=s6.read_json(OUT/f"{scope}/results.json"); continue
        data,repeat,ar,past,target,shared=load_split(scope,history)
        save(f"{scope}/labels.json",data["labels"]); save(f"{scope}/cohort.json",data["cohort"])
        save(f"{scope}/candidates.json",data["val_candidates"]); save(f"{scope}/shared_runtime.json",shared)
        save(f"cache/{scope}/repeat50_scores.json",repeat); save(f"cache/{scope}/association_scores.json",ar)
        factors_cache={}; factor_times={}; results=[]
        for params in (CONTROL,*CANDIDATES):
            name=params["name"]
            category="control_reproduction" if params is CONTROL else "prior_configuration_reuse" if duplicates[name] else "new_unique" if scope=="horizon12" else "repeated_backtest_fit"
            def run(params=params,name=name,category=category):
                alpha=params["alpha"]
                if alpha not in factors_cache:
                    t=time.perf_counter(); factors_cache[alpha]=factors_for(past,target,(s6.START,data["cutoff"]),(data["cutoff"],s6.FINAL_END),data["val_candidates"],alpha)
                    factor_times[alpha]=time.perf_counter()-t
                    save(f"{scope}/factors_alpha{alpha}.json",factors_cache[alpha])
                reused=None
                if params is not CONTROL and duplicates[name]:
                    duplicate=duplicates[name][0]
                    path=ROOT/f"outputs/stage{duplicate['original_stage']}/{scope}/methods/{duplicate['name']}"
                    if (path/"top5.json").exists(): reused=path
                if reused:
                    predictions=s6.read_json(reused/"top5.json"); fallback=s6.read_json(reused/"fallback.json"); elapsed=0.
                else: predictions,fallback,elapsed=predict(data,repeat,ar,factors_cache[alpha],params)
                if params is CONTROL:
                    expected=s6.read_json(ROOT/f"outputs/stage8/{scope}/methods/B_schedule_adjusted_hybrid70/top5.json")
                    assert predictions==expected,"Stage 9 control did not reproduce exactly"
                runtime={"direct_adjust_normalize_fuse_rank_seconds":elapsed,"factor_calculation_seconds":factor_times[alpha],
                         "shared_input_and_AR_seconds":shared["training_cache_schedule_statistics_and_AR_seconds"],
                         "AR_needed_for_strategy":params["association_weight"]>0,
                         "scope":"shared components reused; these are not standalone end-to-end measurements",
                         "exact_duplicate_reused":str(reused.relative_to(ROOT)) if reused else None}
                return write_result(scope,params,predictions,data,fallback,runtime,category)
            results.append(attempt(scope,name,category,run))
            checkpoint("evaluating",scope=scope,last_completed=name)
        users=data["cohort"]; labels=data["labels"]["primary_top5"]
        names=[r["name"] for r in results]
        aps={}
        for name in names:
            values=s6.read_json(OUT/f"{scope}/methods/{name}/per_user_ap5.json")
            aps[name]=np.array([values[u] for u in users])
        deltas=np.column_stack([aps[name]-aps[CONTROL["name"]] for name in names])
        groups={"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),
                "at_least_5":np.array([len(labels[u])>=5 for u in users])}
        bootstrap={}
        for group,mask in groups.items():
            intervals=s8.paired_bootstrap(deltas[mask],SEED,RESAMPLES)
            bootstrap[group]={name:{"delta_map":float(deltas[mask,i].mean()),"paired_95_interval":intervals[i],
                        "users":int(mask.sum()),"users_better":int(np.sum(deltas[mask,i]>1e-14)),
                        "users_worse":int(np.sum(deltas[mask,i]<-1e-14)),"users_tied":int(np.sum(abs(deltas[mask,i])<=1e-14))} for i,name in enumerate(names)}
        save(f"{scope}/paired_bootstrap.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":bootstrap,
             "caveat":"descriptive paired-user, selection-unadjusted; overlapping periods and repeated model selection"})
        baseline=s6.read_json(OUT/f"{scope}/methods/{CONTROL['name']}/top5.json")
        save(f"{scope}/recommendation_changes.json",{name:s8.prediction_changes(s6.read_json(OUT/f"{scope}/methods/{name}/top5.json"),baseline,users) for name in names})
        save(f"{scope}/results.json",results); all_results[scope]=results
        checkpoint(f"{scope}_complete",configurations=7)
    save("results.json",all_results)


def parameter_changes(params):
    return int(params["alpha"]!=CONTROL["alpha"])+int(params["repeat_weight"]!=CONTROL["repeat_weight"])


def eligible(row):
    return row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0


def reproduce_candidate(params,scope,prefix):
    started=time.perf_counter()
    data,repeat,ar,past,target,shared=load_split(scope,need_ar=params["association_weight"]>0)
    factors=factors_for(past,target,(s6.START,data["cutoff"]),(data["cutoff"],s6.FINAL_END),data["val_candidates"],params["alpha"])
    predicted,detail,seconds=predict(data,repeat,ar,factors,params)
    elapsed=time.perf_counter()-started
    expected=s6.read_json(OUT/f"{scope}/methods/{params['name']}/top5.json")
    assert predicted==expected,"Final/shared implementation does not reproduce selected historical predictions"
    save(f"{prefix}/{scope}.json",{"exact_top5_match":True,"users":len(predicted),"end_to_end_seconds":elapsed,
         "scope":"fresh schedule parsing, cache loading, split-audit writes, training statistics/AR, factors and ranking; final reproduction-record write excluded",
          "direct_prediction_seconds":seconds,"shared_components":shared})
    return elapsed


def select():
    if (OUT/"selection.json").exists(): return
    comparison=[]
    for params in CANDIDATES:
        metrics={scope:s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json") for scope in ("horizon12","original10")}
        delta={scope:s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]["all_users"][params["name"]] for scope in metrics}
        row={**params,"horizon12_map":metrics["horizon12"]["map_at_k"],"original10_map":metrics["original10"]["map_at_k"],
             "horizon12_gain":delta["horizon12"]["delta_map"],"original10_gain":delta["original10"]["delta_map"],
             "horizon12_interval":delta["horizon12"]["paired_95_interval"],"original10_interval":delta["original10"]["paired_95_interval"],
             "changed_independent_parameters":parameter_changes(params)}
        row["eligible"]=eligible(row); row["minimum_gain"]=min(row["horizon12_gain"],row["original10_gain"]); comparison.append(row)
    candidates=[r for r in comparison if r["eligible"]]
    winner=None; tie_costs={}
    if candidates:
        best=max(r["minimum_gain"] for r in candidates); tied=[r for r in candidates if r["minimum_gain"]==best]
        fewest=min(r["changed_independent_parameters"] for r in tied); tied=[r for r in tied if r["changed_independent_parameters"]==fewest]
        if len(tied)==1: winner=next(p for p in CANDIDATES if p["name"]==tied[0]["name"])
        else:
            for row in tied:
                params=next(p for p in CANDIDATES if p["name"]==row["name"])
                tie_costs[row["name"]]=sum(attempt(scope,params["name"],"exact_tie_cost_reproduction",lambda scope=scope:reproduce_candidate(params,scope,f"tie_cost/{params['name']}")) for scope in ("horizon12","original10"))
            lowest=min(tie_costs.values()); names=[n for n,v in tie_costs.items() if v==lowest]
            if len(names)==1: winner=next(p for p in CANDIDATES if p["name"]==names[0])
    save("comparison.json",comparison)
    save("selection.json",{"selected":winner,"comparison":comparison,"exact_tie_costs":tie_costs,
         "rule_source":"declaration.json","screening_not_independent_confirmation":True,
         "decision":"freeze and refit only selected candidate" if winner else "retain Stage 9; no candidate satisfies the predeclared screen" if not candidates else "retain Stage 9; unresolved exact cost tie",
         "best_scored_submission":"Stage 9, unchanged"})
    checkpoint("selection_complete",selected=winner["name"] if winner else None)


def final_refit():
    selection=s6.read_json(OUT/"selection.json"); params=selection["selected"]
    if params is None:
        assert not DEST.exists()
        if not any(r["eligible"] for r in selection["comparison"]):
            selection["decision"]="retain Stage 9; no candidate satisfies the predeclared screen"
            save("selection.json",selection)
        save("final_decision.json",{"refitted":False,"submission_generated":False,"reason":"no candidate passed predeclared screen","retain":"Stage 9"}); return
    if (OUT/"final/provenance.json").exists(): return
    for scope in ("horizon12","original10"):
        attempt(scope,params["name"],"selected_final_implementation_reproduction",lambda scope=scope:reproduce_candidate(params,scope,"selected_reproduction"))
    def run():
        assert not DEST.exists(),"Refuse to overwrite existing refined CSV"
        started=time.perf_counter()
        history=s8.raw_schedule(); hist=s8.subset(history,s9.TRAIN[1])
        cfg=s6.load_config(ROOT/"config/stage5.json")
        target,audit=s9.read_target_schedule(cfg.schedule_files["test"],s9.TARGET)
        candidates=sorted(set(int(x) for x in target["show"]))
        assert candidates==s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"] and len(candidates)==6632
        assert s9.identity_set(hist)==s9.identity_set(s8.cache_schedule(cfg.out("airings","airings_history.csv")))
        assert s9.identity_set(target)==s9.identity_set(s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv")))
        variants=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle: users=[r["user_id"] for r in csv.DictReader(handle)]
        pop_ids,pop_stats,pop_rows=s6.build_popularity(variants,candidates)
        data={"cohort":users,"count_variants":variants,"user_to_train_idx":{u:i for i,u in enumerate(ids)},
              "val_candidates":candidates,"popularity_val":pop_ids,"popularity":pop_stats,"cutoff":s9.TRAIN[1]}
        model_start=time.perf_counter()
        repeat=s6.scores_for(users,data["user_to_train_idx"],variants["sens50"],candidates)
        if params["association_weight"]>0: ar,ar_details=s8.ar_scores(data)
        else: ar={u:{} for u in users}; ar_details={"not_fitted":"zero association weight"}
        past_counts=Counter(int(x) for x in hist["show"]); target_counts=Counter(int(x) for x in target["show"])
        factors=factors_for(past_counts,target_counts,s9.TRAIN,s9.TARGET,candidates,params["alpha"])
        predictions,fallback,direct_seconds=predict(data,repeat,ar,factors,params)
        model_seconds=time.perf_counter()-model_start
        from stage5.submission import write_submission,validate_submission
        write_submission(DEST,users,predictions)
        submission_checks=validate_submission(DEST,users,candidates); assert submission_checks["all_passed"]
        save("final/submission_checks.json",submission_checks); save("final/predictions.json",predictions)
        save("final/factors.json",factors); save("final/fallback.json",fallback); save("final/association_details.json",ar_details)
        save("final/popularity.json",pop_rows); save("final/cold_start.json",s6.cold_start_stats(users,data["user_to_train_idx"],variants))
        save("final/opportunity_counts.json",{"training_weeks":s9.weeks(s9.TRAIN),"target_weeks":s9.weeks(s9.TARGET),
             "past":past_counts,"target":target_counts,"alpha":params["alpha"],"factor_min":min(factors.values()),"factor_max":max(factors.values())})
        save("final/configuration.json",{**params,"canonical_signature":signature(params),"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET)})
        save("final/provenance.json",{"submission":str(DEST.relative_to(ROOT)),"sha256":s6.sha256_file(DEST),"selected":params,
             "test_candidates":6632,"template_users":len(users),"future_viewing_used":False,"best_scored_submission":str(BEST.relative_to(ROOT)),
             "stage10_returned_score":None,"source_manifest":"preservation_before.json","feedback_informed_followup":True,
             "runtime":{"direct_adjust_fuse_rank_seconds":direct_seconds,"fit_predict_loaded_inputs_seconds":model_seconds,
                         "full_refit_with_schedule_and_io_seconds":time.perf_counter()-started}})
        save("final/target_schedule_audit.json",audit)
        checkpoint("selected_refit_complete",sha256=s6.sha256_file(DEST))
    attempt("full_history",params["name"],"selected_refit_not_new_configuration",run)


def verify():
    checks={}; before=s6.read_json(OUT/"preservation_before.json"); after=manifest()
    selection=s6.read_json(OUT/"selection.json")
    save("preservation_after.json",after)
    checks["all_prior_files_and_raw_inputs_byte_preserved"]=before==after
    checks["stage9_best_sha_preserved"]=s6.sha256_file(BEST)==BEST_SHA
    for scope in ("horizon12","original10"):
        users=s6.read_json(OUT/f"cache/{scope}/cohort.json"); labels=s6.read_json(OUT/f"cache/{scope}/labels.json")
        candidates=s6.read_json(OUT/f"cache/{scope}/candidates.json")
        checks[f"{scope}/unchanged_users_labels_candidates"]=(users==s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json") and
             labels==s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json") and candidates==s6.read_json(ROOT/f"outputs/stage8/{scope}/candidates.json"))
        aps={}
        for params in (CONTROL,*CANDIDATES):
            name=params["name"]; base=OUT/f"{scope}/methods/{name}"
            predictions=s6.read_json(base/"top5.json"); metrics=s6.read_json(base/"metrics.json"); recorded=s6.read_json(base/"per_user_ap5.json")
            independent={}
            for u in users:
                actual=set(labels["primary_top5"][u]); seen=set(); hit=0; terms=[]
                for i,s in enumerate(predictions[u][:5],1):
                    if s in actual and s not in seen: hit+=1; terms.append(hit/i)
                    seen.add(s)
                independent[u]=math.fsum(terms)/min(5,len(actual)) if actual else 0.
            checks[f"{scope}/{name}/independent_AP"]=all(abs(independent[u]-recorded[u])<1e-12 for u in users)
            groups={"map_at_k":users,"map_with_positive":[u for u in users if labels["primary_top5"][u]],
                    "map_with_at_least_k_positive":[u for u in users if len(labels["primary_top5"][u])>=5]}
            checks[f"{scope}/{name}/three_cohort_metrics"]=all(abs((math.fsum(independent[u] for u in group)/len(group) if group else 0)-metrics[k])<1e-12 for k,group in groups.items())
            checks[f"{scope}/{name}/valid_predictions"]=set(predictions)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predictions.values())
            aps[name]=np.array([independent[u] for u in users])
        checks[f"{scope}/control_exact"]=s6.read_json(OUT/f"{scope}/methods/{CONTROL['name']}/top5.json")==s6.read_json(ROOT/f"outputs/stage8/{scope}/methods/B_schedule_adjusted_hybrid70/top5.json")
        names=[p["name"] for p in (CONTROL,*CANDIDATES)]
        deltas=np.column_stack([aps[name]-aps[CONTROL["name"]] for name in names])
        intervals=s8.paired_bootstrap(deltas,SEED,RESAMPLES); saved=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]["all_users"]
        checks[f"{scope}/paired_intervals_reproduced"]=all(np.allclose(intervals[i],saved[name]["paired_95_interval"],rtol=0,atol=1e-12) for i,name in enumerate(names))
        for i,name in enumerate(names[1:],1):
            row=next(r for r in selection["comparison"] if r["name"]==name)
            checks[f"{scope}/{name}/independent_selection_gain"]=abs(float(deltas[:,i].mean())-row[f"{scope}_gain"])<1e-12
        base_factors={int(k):v for k,v in s6.read_json(OUT/f"{scope}/factors_alpha0.5.json").items()}
        reference={int(k):v for k,v in s6.read_json(ROOT/f"outputs/stage8/{scope}/schedule_adjustment_factors.json").items()}
        checks[f"{scope}/control_factors_exact"]=base_factors==reference
        factor_summary={}
        for alpha in (.25,.5,.75,1.):
            values={int(k):v for k,v in s6.read_json(OUT/f"{scope}/factors_alpha{alpha}.json").items()}
            checks[f"{scope}/alpha{alpha}_independent_exponent_and_bounds"]=(set(values)==set(base_factors) and
                 all(abs(v-base_factors[s]**(2*alpha))<1e-14 and .25**alpha-1e-14<=v<=4.**alpha+1e-14 for s,v in values.items()))
            factor_summary[str(alpha)]={"programs":len(values),"minimum":min(values.values()),"maximum":max(values.values()),
                                       "ratio_clip":[.25,4.],"post_exponent_old_bounds_clipping":False}
        save(f"{scope}/factor_summary.json",factor_summary)
        shared=s6.read_json(OUT/f"{scope}/shared_runtime.json"); cutoff=s8.HORIZON_CUTOFF if scope=="horizon12" else s6.PRIMARY_END
        checks[f"{scope}/no_training_future"]=shared["max_training_airing_end"]<=cutoff
    comparison=selection["comparison"]
    checks["eligibility_exact_predeclared_rule"]=all(r["eligible"]==eligible(r) for r in comparison)
    qualified=[r for r in comparison if r["eligible"]]
    if selection["selected"]:
        chosen=next(r for r in comparison if r["name"]==selection["selected"]["name"])
        checks["maximin_selection"]=chosen["eligible"] and chosen["minimum_gain"]==max(r["minimum_gain"] for r in qualified)
        for scope in ("horizon12","original10"):
            checks[f"{scope}/selected_final_implementation_exact"]=s6.read_json(OUT/f"selected_reproduction/{scope}.json")["exact_top5_match"]
        provenance=s6.read_json(OUT/"final/provenance.json")
        cfg=s6.load_config(ROOT/"config/stage5.json")
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle: users=[r["user_id"] for r in csv.DictReader(handle)]
        from stage5.submission import validate_submission
        audit=validate_submission(DEST,users,s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"])
        for item in audit["checks"]: checks[f"submission/{item['check']}"]=item["ok"]
        checks["final_sha256"]=s6.sha256_file(DEST)==provenance["sha256"]
        checks["at_most_one_csv"]=list(OUT.glob("*.csv"))==[DEST]
    else:
        checks["no_eligible_candidate_and_no_csv"]=not qualified and not DEST.exists() and not list(OUT.glob("*.csv"))
    declaration=s6.read_json(OUT/"declaration.json")
    checks["six_authorized_candidates_only"]=declaration["six_configurations"]==CANDIDATES and len(comparison)==6
    records=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    new={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    checks["unique_configurations_within_41"]=len(new)==declaration["new_unique"] and 35+len(new)<=41
    checks["no_failed_model_attempts"]=not any(r["event"]=="failed" for r in records)
    checks["stage9_remains_best_scored"]=s6.read_json(OUT/"current_best_submission.json")["stage"]==9
    checks["declaration_before_first_evaluation"]=(OUT/"declaration.json").stat().st_mtime<min(r["time_epoch"] for r in records)
    save("search_history.json",{"previous_unique":35,"new_unique":len(new),"total_unique":35+len(new),
         "previous_catalog":"outputs/stage8/search_history.json","previous_catalog_sha256":s6.sha256_file(ROOT/"outputs/stage8/search_history.json"),
         "new_configurations":[{**p,"signature":signature(p)} for p in CANDIDATES if p["name"] in new],
         "refits_reproductions_and_duplicate_reuse_not_counted_as_new":True})
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
             "preserved_workspace_files":len(before),"total_unique_configurations":35+len(new),"selected":selection["selected"]}
    save("audit/validation.json",summary)
    sources={p.resolve() for p in (OUT/"code").glob("*.py")}
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": sources.add(path)
    # importlib helpers are intentionally not registered as sys.modules entries.
    sources.update(ROOT/f"outputs/stage{s}/code/{f}" for s,f in ((6,"study.py"),(8,"study.py"),(9,"refit.py")))
    save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(sources)})
    assert summary["all_passed"],summary
    checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",selected=selection["selected"])
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","comparison":comparison,"decision":selection["decision"],
          "selected":selection["selected"],"preserved_files":len(before)},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("prepare","evaluate","select","final","verify","all")); phase=parser.parse_args().phase
    if phase in ("prepare","all"): prepare()
    if phase in ("evaluate","all"): evaluate()
    if phase in ("select","all"): select()
    if phase in ("final","all"): final_refit()
    if phase in ("verify","all"): verify()


if __name__=="__main__":main()
