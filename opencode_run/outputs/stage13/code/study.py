#!/usr/bin/env python3
"""Five fixed refinements of Stage 12, using its verified weighted-EASE fits."""
from __future__ import annotations

import argparse
import csv
import gc
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
spec=importlib.util.spec_from_file_location("stage12_readonly",ROOT/"outputs/stage12/code/study.py")
p12=importlib.util.module_from_spec(spec); spec.loader.exec_module(p12)
r11,s6,s8,s9,e,np=p12.r11,p12.s6,p12.s8,p12.s9,p12.e,p12.np
OUT=ROOT/"outputs/stage13"
s8.OUT=OUT/"cache"; s6.OUT=OUT/"cache"; s9.OUT=OUT/"cache"
BEST=ROOT/"outputs/stage12/kuznetsov_vladimir_midterm_submission_frequency_ease.csv"
BEST_SHA="5128cb80f1fa160b08a26a67539cd3a8059640dac6bb8038b55e43ccbd639aee"
DEST=OUT/"kuznetsov_vladimir_midterm_submission_refined_ease.csv"
CONTROL={"name":"stage12_control","lambda":1000,"beta":.1,"EASE_opportunity_adjustment":False}
VARIANTS=[
    {"name":"A_beta005","lambda":1000,"beta":.05,"EASE_opportunity_adjustment":False},
    {"name":"B_beta015","lambda":1000,"beta":.15,"EASE_opportunity_adjustment":False},
    {"name":"C_beta020","lambda":1000,"beta":.2,"EASE_opportunity_adjustment":False},
    {"name":"D_beta030","lambda":1000,"beta":.3,"EASE_opportunity_adjustment":False},
    {"name":"E_EASE_opportunity","lambda":1000,"beta":.1,"EASE_opportunity_adjustment":True},
]
SCOPES=("horizon12","original10")
SEED,RESAMPLES=137,10000


def save(path,value):
    path=OUT/path; path.parent.mkdir(parents=True,exist_ok=True); s6.write_json(path,value)


def checkpoint(phase,**details):
    item={"phase":phase,"time_epoch":time.time(),**details}
    save("checkpoints/progress.json",item); save(f"checkpoints/{time.time_ns()}_{phase}.json",item)


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
    return {"method":"global-training-scaled log1p primary80 EASE plus exact S9", "lambda":params["lambda"],
            "beta":params["beta"],"EASE_opportunity_adjustment":params["EASE_opportunity_adjustment"]}


def prepare():
    if (OUT/"declaration.json").exists(): return
    assert s6.sha256_file(BEST)==BEST_SHA
    assert s6.read_json(ROOT/"outputs/stage12/search_history.json")["total_unique"]==53
    save("preservation_before.json",manifest())
    catalog=s6.read_json(ROOT/"outputs/stage8/search_history.json")["configurations"]
    inspected=[]
    for row in catalog:
        path=ROOT/row["parameters_path"]
        inspected.append({"name":row["name"],"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path),"canonical_signature":None})
    prior_count_models=[]
    for stage in (10,11,12):
        for row in s6.read_json(ROOT/f"outputs/stage{stage}/search_history.json")["new_configurations"]:
            path=ROOT/f"outputs/stage{stage}/horizon12/methods/{row['name']}/parameters.json"
            canonical=signature({**row,"EASE_opportunity_adjustment":False}) if stage==12 else None
            inspected.append({"name":row["name"],"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path),"canonical_signature":canonical})
            if canonical is not None: prior_count_models.append({"name":row["name"],"stage":12,"signature":canonical})
    assert len(inspected)==53
    duplicates={v["name"]:[row for row in prior_count_models if row["signature"]==signature(v)] for v in VARIANTS}
    declarations={}
    for stage,file in ((6,"plan.json"),(7,"plan.json"),(8,"plan.json"),(9,"configuration.json"),(10,"declaration.json"),(11,"declaration.json"),(12,"declaration.json")):
        path=ROOT/f"outputs/stage{stage}/{file}"; declarations[str(path.relative_to(ROOT))]={"sha256":s6.sha256_file(path),"declaration":s6.read_json(path)}
    assert declarations["outputs/stage12/declaration.json"]["declaration"]["fixed"]["EASE_schedule_adjusted"] is False
    save("duplicate_audit.json",{"prior_53_parameter_records":inspected,"prior_declarations":declarations,"exact_duplicates":duplicates,
         "note":"Binary EASE beta .20/.30 are not duplicates of weighted-input EASE. Earlier schedule correction on repeat is not EASE opportunity adjustment.","replace_duplicate_slots":False})
    save("declaration.json",{"declared_before_evaluation":True,"feedback_informed_followup":True,"not_independent_confirmation":True,
         "suggestion_source":"external AI assistant Codex","control":CONTROL,"five_variants":VARIANTS,
         "previous_unique":53,"new_unique_budget":5,"maximum_total_unique":58,
         "fixed":{"EASE_input":"training-only globally scaled log1p >=80% qualifying-airing counts, exact Stage 12 scalar",
              "lambda":1000,"negative_coefficients":"retained through complete X_user@B", "seen_items_excluded":False,
              "S9":"exact .70 normalized adjusted repeat50 + .30 normalized AR80; no additional S9 normalization",
              "fusion":"(1-beta)*S9 + beta*normalized_positive_EASE", "fallback_ranking":s6.fusion_rules()},
         "variant_E":{"only_change":"EASE opportunity adjustment; beta .10 and S9 unchanged",
              "factor":"sqrt(clip((target_count/target_weeks)/(historical_count/training_weeks),.25,4))",
              "order":"complete weighted-EASE prediction, clip scores to zero, multiply eligible positive scores by factor, then per-user candidate-only max normalization",
              "zero_evidence":"remains zero", "positive_score_missing_historical_opportunity":"raise inconsistency, do not invent a denominator",
              "counts":"Stage 8 ownership, identity dedupe, labelled positive training airings fully before cutoff; target starts inside window including end crossings",
              "interpretation":"forecasting heuristic, not calibrated probability", "combined_with_other_betas":False},
         "data":"unchanged two Stage 8 splits, labels, 1260 users, cohorts, pools and temporal boundaries",
         "reuse":"Stage 12 lambda1000 raw completed EASE score caches and verified S9 scores; no historical EASE refits needed",
         "controls":"reproduce exact Stage 12 lists and independently recompute saved metric values, not rounded inputs",
         "diagnostics":"existing repeat50 relevant-pair and eligible-history <20/>=20 partitions, not model or selection variants",
         "bootstrap":{"resamples":RESAMPLES,"seed":SEED,"order":"template users","interval":"paired-user percentile descriptive 95%, all three cohorts"},
         "selection":{"gain_each_split_at_least":.002,"horizon12_interval_lower_strictly_above":0.,
              "objective":"maximize minimum gain across two splits","tie_break":"lower measured computational cost summed across splits",
              "cost_scope":"direct EASE component adjustment/normalization and fusion/ranking plus reused measured EASE fit/query cost; shared component-attributed, not standalone end-to-end measurement",
              "remaining_exact_tie":"retain Stage 12"},
         "uncertainty":"overlapping and repeatedly used historical periods; selection-unadjusted intervals; no independent statistical confirmation",
         "kaggle_scores_are_not_targets":True,"conditional_refit":"one frozen winner only; no CSV if none qualifies",
         "output":str(DEST.relative_to(ROOT))})
    save("source_attribution.json",{"suggestion":"external AI assistant Codex suggested this bounded follow-up",
         "calculations_implementation_automated_verification":"performed in this OpenCode session using project data/code",
         "personal_user_analysis_or_verification_claimed":False,"personal_verification":"pending","external_diagnostic_files_read":False,
         "feedback_informed":True,"independent_confirmation":False})
    save("score_evidence.json",{"source":"user confirms returned Stage 12 scores","stage12":{"public":.44041,"private":.42267,
         "submission":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA},"historical_artifacts_updated":False,"used_for_optimization":False})
    save("current_best_submission.json",{"stage":12,"path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA,"public":.44041,"private":.42267,
         "designation":"best scored, retained unless a later returned score establishes otherwise"})
    checkpoint("five_variants_declared")


def component(raw,candidates,factors=None,adjust=False):
    """Adjustment follows completed-score clipping and precedes candidate max."""
    positive=np.maximum(raw,0.)
    if adjust:
        factors=factors or {}
        for i,program in enumerate(candidates):
            if np.any(positive[:,i]>0) and (program not in factors or not math.isfinite(factors[program]) or factors[program]<=0):
                raise ValueError(f"Positive learned-item EASE score without valid historical opportunity: {program}")
        multiplier=np.array([factors.get(program,0.) for program in candidates],dtype=np.float64)
        positive*=multiplier[None,:]
    maximum=positive.max(axis=1,keepdims=True)
    return np.divide(positive,maximum,out=np.zeros_like(positive),where=maximum>0)


def predict(data,S9,raw,params,factors=None):
    started=time.perf_counter()
    normalized=component(raw,data["val_candidates"],factors,params["EASE_opportunity_adjustment"])
    normalize_seconds=time.perf_counter()-started
    predictions,fallback,fusion_seconds=r11.combine(data,S9,normalized,params["beta"])
    return predictions,fallback,{"component_clip_adjust_normalize_seconds":normalize_seconds,"fusion_rank_seconds":fusion_seconds,
          "direct_variant_seconds":time.perf_counter()-started}


def load_inputs(scope):
    data,S9,repeat,baseline_stage9,provenance=r11.inputs(scope)
    baseline=s6.read_json(ROOT/f"outputs/stage12/{scope}/methods/logcount_ease_l1000_b10/top5.json")
    with np.load(ROOT/f"outputs/stage12/{scope}/models/lambda1000/scores.npz") as saved:
        raw=saved["raw"]; normalized=saved["normalized"]
    assert np.array_equal(component(raw,data["val_candidates"]),normalized)
    stats=s6.read_json(ROOT/f"outputs/stage12/{scope}/models/input_statistics.json")
    source_counts=[n for items in data["count_variants"]["primary80"].values() for n in items.values() if n>0]
    scalar=math.sqrt(len(source_counts)/math.fsum(math.log1p(n)**2 for n in source_counts)) if source_counts else 0.
    assert abs(scalar-stats["normalization_scalar"])<1e-13
    assert np.all(np.isfinite(raw))
    return data,S9,repeat,baseline,raw,provenance


def opportunity_inputs(scope,data,history):
    started=time.perf_counter(); hist=s8.subset(history,data["cutoff"]); target=s8.subset(history,data["cutoff"],target=True)
    assert sorted(set(int(j) for j in target["show"]))==data["val_candidates"]
    past=Counter(int(j) for j in hist["show"]); future=Counter(int(j) for j in target["show"])
    train_window=(s6.START,data["cutoff"]); target_window=(data["cutoff"],s6.FINAL_END)
    factors=s9.factors_for(past,future,train_window,target_window,data["val_candidates"])
    previous={int(k):v for k,v in s6.read_json(ROOT/f"outputs/stage9/reproduction/{scope}/factors.json").items()}
    assert factors==previous
    summary={"past":past,"target":future,"training_window":list(train_window),"target_window":list(target_window),
             "training_weeks":s9.weeks(train_window),"target_weeks":s9.weeks(target_window),"factors":factors,
             "training_airings":len(hist["show"]),"target_airings":len(target["show"]),"matches_verified_Stage9_factors":True,
             "direct_opportunity_count_and_factor_seconds":time.perf_counter()-started}
    save(f"{scope}/opportunities.json",summary)
    return factors,summary


def persist(scope,params,predictions,data,baseline,repeat,fallback,runtime,category):
    base=f"{scope}/methods/{params['name']}"; save(f"{base}/top5.json",predictions)
    metrics,aps=s9.independent_metrics(s6.read_json(OUT/f"{base}/top5.json"),data["labels"]["primary_top5"],data["cohort"])
    save(f"{base}/metrics.json",metrics); save(f"{base}/per_user_ap5.json",aps); save(f"{base}/parameters.json",params)
    save(f"{base}/fallback.json",fallback); save(f"{base}/runtime.json",runtime)
    save(f"{base}/recommendation_changes.json",s8.prediction_changes(predictions,baseline,data["cohort"]))
    save(f"{base}/subgroups.json",r11.subgroup_results(data,repeat,baseline,predictions))
    save(f"{base}/provenance.json",{"category":category,"training_cutoff":data["cutoff"],"labels_used_as_training_features":False,
         "EASE_scores_source":f"outputs/stage12/{scope}/models/lambda1000/scores.npz",
         "EASE_scores_source_sha256":s6.sha256_file(ROOT/f"outputs/stage12/{scope}/models/lambda1000/scores.npz"),
         "S9_renormalized":False,"declaration":"declaration.json","source_manifest":"preservation_before.json"})
    save(f"{base}/complete.json",{"completed":True})
    return {"name":params["name"],"MAP":metrics["map_at_k"]}


def evaluate():
    assert s6.read_json(OUT/"declaration.json")["five_variants"]==VARIANTS
    if (OUT/"results.json").exists(): return
    history=s8.raw_schedule(); all_results={}
    duplicates=s6.read_json(OUT/"duplicate_audit.json")["exact_duplicates"]
    for scope in SCOPES:
        if (OUT/f"{scope}/results.json").exists(): all_results[scope]=s6.read_json(OUT/f"{scope}/results.json"); continue
        data,S9,repeat,baseline,raw,provenance=load_inputs(scope)
        factors,opportunities=opportunity_inputs(scope,data,history)
        save(f"{scope}/input_provenance.json",provenance); save(f"{scope}/labels.json",data["labels"])
        save(f"{scope}/cohort.json",data["cohort"]); save(f"{scope}/candidates.json",data["val_candidates"])
        control_pred,control_fb,control_time=predict(data,S9,raw,CONTROL)
        assert control_pred==baseline,"Stage 12 exact control reproduction failed"
        control_metrics,control_AP=s9.independent_metrics(baseline,data["labels"]["primary_top5"],data["cohort"])
        saved_control=s6.read_json(ROOT/f"outputs/stage12/{scope}/methods/logcount_ease_l1000_b10/metrics.json")
        assert all(abs(control_metrics[k]-saved_control[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive"))
        persist(scope,CONTROL,control_pred,data,baseline,repeat,control_fb,control_time,"fixed_control_reproduction")
        results=[]
        for params in VARIANTS:
            base=OUT/f"{scope}/methods/{params['name']}"
            if (base/"complete.json").exists():
                results.append({"name":params["name"],"MAP":s6.read_json(base/"metrics.json")["map_at_k"]}); continue
            category="prior_configuration_reuse" if duplicates[params["name"]] else "new_unique" if scope=="horizon12" else "repeated_backtest_evaluation"
            def run(params=params,category=category):
                if duplicates[params["name"]]:
                    item=duplicates[params["name"]][0]
                    previous=ROOT/f"outputs/stage{item['stage']}/{scope}/methods/{item['name']}"
                    predicted=s6.read_json(previous/"top5.json"); fb=s6.read_json(previous/"fallback.json"); timing={"reused_exact_configuration":str(previous.relative_to(ROOT))}
                else: predicted,fb,timing=predict(data,S9,raw,params,factors)
                prior_runtime=s6.read_json(ROOT/f"outputs/stage12/{scope}/models/lambda1000/runtime.json")
                timing["reused_EASE_fit_query_seconds"]=prior_runtime["direct_fit_and_score_seconds"]
                timing["measured_cost_for_selection_seconds"]=prior_runtime["direct_fit_and_score_seconds"]+timing.get("direct_variant_seconds",0.)
                timing["scope"]="reused EASE fit/query plus directly measured component/fusion work; component-attributed, not standalone end-to-end runtime"
                return persist(scope,params,predicted,data,baseline,repeat,fb,timing,category)
            results.append(attempt(scope,params["name"],category,run)); checkpoint("evaluating",scope=scope,last_completed=params["name"])
        users=data["cohort"]; labels=data["labels"]["primary_top5"]; arrays=[]
        for params in VARIANTS:
            aps=s6.read_json(OUT/f"{scope}/methods/{params['name']}/per_user_ap5.json")
            arrays.append(np.array([aps[u]-control_AP[u] for u in users]))
        delta=np.column_stack(arrays); intervals={}
        for group,mask in {"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}.items():
            bounds=s8.paired_bootstrap(delta[mask],SEED,RESAMPLES)
            intervals[group]={p["name"]:{"delta_MAP":float(delta[mask,i].mean()),"paired_95_interval":bounds[i],"users":int(mask.sum())} for i,p in enumerate(VARIANTS)}
        save(f"{scope}/paired_bootstrap.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":intervals,
             "uncertainty":"overlapping repeatedly used periods; descriptive selection-unadjusted intervals, not independent confirmation"})
        save(f"{scope}/results.json",results); all_results[scope]=results; checkpoint(f"{scope}_complete")
        del data,raw; gc.collect()
    save("results.json",all_results)


def select():
    if (OUT/"selection.json").exists(): return
    comparison=[]
    for params in VARIANTS:
        row={**params}; costs=[]
        for scope in SCOPES:
            metrics=s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")
            interval=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]["all_users"][params["name"]]
            row[f"{scope}_MAP"]=metrics["map_at_k"]; row[f"{scope}_gain"]=interval["delta_MAP"]; row[f"{scope}_interval"]=interval["paired_95_interval"]
            costs.append(s6.read_json(OUT/f"{scope}/methods/{params['name']}/runtime.json")["measured_cost_for_selection_seconds"])
        row["eligible"]=row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0
        row["minimum_gain"]=min(row["horizon12_gain"],row["original10_gain"]); row["measured_cost_sum_seconds"]=sum(costs); comparison.append(row)
    eligible=[r for r in comparison if r["eligible"]]; selected=None
    if eligible:
        best=max(r["minimum_gain"] for r in eligible); tied=[r for r in eligible if r["minimum_gain"]==best]
        cost=min(r["measured_cost_sum_seconds"] for r in tied); tied=[r for r in tied if r["measured_cost_sum_seconds"]==cost]
        if len(tied)==1: selected=next(p for p in VARIANTS if p["name"]==tied[0]["name"])
    save("comparison.json",comparison); save("selection.json",{"selected":selected,"comparison":comparison,"rule_source":"declaration.json",
         "decision":"freeze and refit only the selected fixed variant" if selected else "retain Stage 12; no eligible winner",
         "diagnostics_and_Kaggle_scores_not_selection_targets":True,"independent_confirmation":False})
    checkpoint("selection_complete",selected=selected)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("prepare","evaluate","select")); phase=parser.parse_args().phase
    if phase=="prepare": prepare()
    elif phase=="evaluate": evaluate()
    else: select()


if __name__=="__main__":main()
