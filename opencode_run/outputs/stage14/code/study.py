#!/usr/bin/env python3
"""Six fixed recommendation configurations testing three association signals."""
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
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("stage13_readonly",ROOT/"outputs/stage13/code/study.py")
p13=importlib.util.module_from_spec(spec); spec.loader.exec_module(p13)
r11,s6,s8,s9,e,np=p13.r11,p13.s6,p13.s8,p13.s9,p13.e,p13.np
OUT=ROOT/"outputs/stage14"
s8.OUT=OUT/"cache"; s6.OUT=OUT/"cache"; s9.OUT=OUT/"cache"
BEST=ROOT/"outputs/stage13/kuznetsov_vladimir_midterm_submission_refined_ease.csv"
BEST_SHA="628edd64a9c4ecc26df1f0c2174d7378db4c15ebc4f01615a51f841ff44779d4"
DEST=OUT/"kuznetsov_vladimir_midterm_submission_preference_ar.csv"
SCOPES=("horizon12","original10")
MODELS=[{"variant":"A","target":"original_any_primary80","K":None,"AR_opportunity_adjustment":True},
        {"variant":"B","target":"fractional_preferred_primary80","K":5,"AR_opportunity_adjustment":False},
        {"variant":"C","target":"fractional_preferred_primary80","K":10,"AR_opportunity_adjustment":False}]
CONFIGS=[{**model,"name":f"{model['variant']}_{role}","role":role} for model in MODELS for role in ("standalone","hybrid")]
SEED,RESAMPLES=137,10000


def save(path,value):
    path=OUT/path; path.parent.mkdir(parents=True,exist_ok=True); s6.write_json(path,value)


def checkpoint(phase,**details):
    value={"phase":phase,"time_epoch":time.time(),**details}
    save("checkpoints/progress.json",value); save(f"checkpoints/{time.time_ns()}_{phase}.json",value)


def manifest():
    return {str(p.relative_to(ROOT)):{"sha256":s6.sha256_file(p),"bytes":p.stat().st_size}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_relative_to(OUT)}


def attempt(scope,name,category,operation):
    def log(event,**details):
        with (OUT/"attempts.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps({"scope":scope,"name":name,"category":category,"event":event,"time_epoch":time.time(),**details},sort_keys=True)+"\n")
    log("started")
    try: result=operation()
    except Exception:
        log("failed",traceback=traceback.format_exc()); checkpoint("failed",scope=scope,name=name); raise
    log("completed"); return result


def prepare():
    if (OUT/"declaration.json").exists(): return
    assert s6.sha256_file(BEST)==BEST_SHA
    assert s6.read_json(ROOT/"outputs/stage13/search_history.json")["total_unique"]==58
    save("preservation_before.json",manifest())
    inspected=[]
    for row in s6.read_json(ROOT/"outputs/stage8/search_history.json")["configurations"]:
        path=ROOT/row["parameters_path"]; inspected.append({"name":row["name"],"path":row["parameters_path"],"sha256":s6.sha256_file(path)})
    for stage in (10,11,12,13):
        for row in s6.read_json(ROOT/f"outputs/stage{stage}/search_history.json")["new_configurations"]:
            path=ROOT/f"outputs/stage{stage}/horizon12/methods/{row['name']}/parameters.json"
            inspected.append({"name":row["name"],"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path)})
    assert len(inspected)==58
    prior={}
    for stage,file in ((6,"plan.json"),(7,"plan.json"),(8,"plan.json"),(9,"configuration.json"),(10,"declaration.json"),(11,"declaration.json"),(12,"declaration.json"),(13,"declaration.json")):
        path=ROOT/f"outputs/stage{stage}/{file}"; prior[str(path.relative_to(ROOT))]={"sha256":s6.sha256_file(path),"declaration":s6.read_json(path)}
    assert not any("preferred" in row["name"] or "preference_ar" in row["name"] for row in inspected)
    save("duplicate_audit.json",{"prior_58_parameter_records":inspected,"prior_declarations":prior,"exact_duplicates":{c["name"]:[] for c in CONFIGS},
         "note":"Earlier schedule adjustments changed repeat or EASE scores, not AR. No earlier method learns fractional Top-K consequent memberships. Original standalone AR is not adjusted AR A.","duplicate_slots_replaced":False})
    source_paths=[ROOT/"src/stage3/association.py",ROOT/"src/stage4/association.py"]
    save("association_inspection.json",{"source_sha256":{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in source_paths},
         "current_model":"binary any >=80% training history; float32 binary Gram converted to float64 rule arithmetic",
         "current_rule_ties":"confidence descending then sorted training-item column index",
         "current_antecedents":"qualifying training count descending then sorted training-item column, maximum 50",
         "current_aggregation":"maximum confidence; target candidates filtered before top-50 rule truncation",
         "new_membership_ties":"fractional full-catalogue boundary memberships, not ID-based preference",
         "new_C":"asymmetric float64 A.T @ Y; no symmetry assumption or rounding"})
    save("declaration.json",{"declared_before_evaluation":True,"suggestion_source":"external AI assistant Codex",
         "previous_unique":58,"new_recommendation_configurations":CONFIGS,"maximum_new_unique":6,"maximum_total_unique":64,
         "standalone_counts_toward_budget":True,"conditional_refit_candidates":"hybrids A/B/C only; standalone results are diagnostic",
         "control":{"source":"exact Stage 13 score","algebraic_weights":{"repeat":.63,"original_AR":.27,"adjusted_EASE":.1},
              "arithmetic":"preserve evaluated nested order .90*(.70*R_norm + .30*AR_norm) + .10*E_norm rather than rounding/reassociating scores",
              "EASE":"training-only globally normalized log1p primary80, lambda1000, opportunity adjustment before candidate-only max",
              "repeat":"unchanged adjusted repeat50 counts, no decay","combined_score_renormalized":False},
         "A":{"change":"multiply original completed max-confidence candidate AR scores by established factor, then candidate-only per-user maximum normalization",
              "factor":"sqrt(clip((target_count/target_weeks)/(historical_count/training_weeks),.25,4))","rule_learning_unchanged":True},
         "B_C":{"A_ui":"1 if any >=80% qualifying training airing, zero otherwise",
              "n_ui":"deduplicated qualifying training-airing count","catalogue":"all training programmes, before target filtering",
              "Y":"Top-K membership: all 1 when positives<=K; above Kth count 1, below 0; boundary tie (K-a)/b",
              "Y_row_sum":"min(K, number of positive training programmes)","K_values":[5,10],
              "C_ij":"sum_u A_ui*Y_uj = A.T@Y; asymmetric fractional float64 mass",
              "support_A_i":"sum_u A_ui","support_Y_j":"sum_u Y_uj",
              "confidence":"C_ij/support_A_i","lift":"confidence/(support_Y_j/number_of_usable_primary80_training_users)",
              "zero_denominators":"no rule","user_population":"exact original AR usable-primary80 population, all training users",
              "AR_opportunity_adjustment":False},
         "rule_settings":{"effective_mass_min":5,"confidence_min":.005,"lift_min":1.,"exclude_diagonal":True,
              "max_rules_per_antecedent":50,"max_personal_antecedents":50,"aggregation":"max confidence",
              "antecedent_order":"original training count descending then sorted catalogue column index",
              "rule_order":"confidence descending then sorted catalogue column index","candidate_filter":"rules after full-catalogue statistics, before top-50 pruning"},
         "normalization":"independent positive component maxima over eligible candidates; standalone uses the same normalized AR component",
         "splits":"unchanged Stage 8 horizon12 and original10 labels, users, pools, cutoffs, cohorts and boundaries",
         "fallback_ranking":s6.fusion_rules(),"seen_items_excluded":False,
         "reuse":"verified repeat and adjusted weighted-EASE components; each of A/B/C AR signals reused for standalone and hybrid",
         "bootstrap":{"resamples":RESAMPLES,"seed":SEED,"user_order":"template order","interval":"paired-user descriptive percentile 95%, all three cohorts"},
         "diagnostics":"familiar vs no personal >=50% repeat evidence, plus existing global/history-size partitions; never extra models or selection targets",
         "selection":{"hybrids_only":True,"gain_each_split_at_least":.002,"horizon12_interval_lower_strictly_above":0.,
              "objective":"maximize minimum gain across splits","tie_break":"lower measured component cost summed across both splits; retain Stage 13 on exact cost tie"},
         "interpretation":"retrospective preference associations, not calibrated future Top-5 probabilities",
         "uncertainty":"overlapping repeatedly used historical periods; selection-unadjusted evidence, not independent confirmation",
         "kaggle_scores_not_optimization_targets":True,"conditional_refit":"one frozen qualifying hybrid only; no CSV if none",
         "output":str(DEST.relative_to(ROOT))})
    save("source_attribution.json",{"suggestion":"external AI assistant Codex suggested this follow-up",
         "implementation_calculations_automated_verification":"performed in this OpenCode session with project data/code",
         "personal_verification":"pending; not claimed","external_diagnostic_files_read":False,"independent_confirmation_claimed":False})
    save("score_evidence.json",{"source":"user confirms returned Stage 13 scores","stage13":{"public":.44099,"private":.42315,
         "path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA},"observations_only":True,"optimization_targets":False,"historical_artifacts_modified":False})
    save("current_best_submission.json",{"stage":13,"path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA,"public":.44099,"private":.42315,
         "designation":"best scored, retained until a later returned score establishes otherwise"})
    checkpoint("six_configurations_declared")


def ar_module():
    spec=importlib.util.spec_from_file_location("stage14_preference_AR",OUT/"code/preference_ar.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def positive_normalize(scores,candidates):
    allowed=set(candidates)
    return {u:s8.normalize({j:float(v) for j,v in rows.items() if j in allowed and v>0}) for u,rows in scores.items()}


def opportunity_adjust(scores,factors,candidates):
    allowed=set(candidates); result={}
    for u,rows in scores.items():
        adjusted={}
        for j,value in rows.items():
            if j not in allowed or value<=0: continue
            if j not in factors: raise ValueError(f"Positive AR score without historical opportunity: {j}")
            adjusted[j]=float(value)*factors[j]
        result[u]=adjusted
    return result


def rank_hybrid(data,R,AR,E):
    # Preserve original nested arithmetic; no extra max on S9 or the final sum.
    modified_S9={}
    for u in data["cohort"]:
        row={j:.7*value for j,value in R[u].items()}
        for j,value in AR[u].items(): row[j]=row.get(j,0)+.3*value
        modified_S9[u]=row
    return r11.combine(data,modified_S9,E,.1)


def load_inputs(scope):
    data,S9,repeat,prior,raw,provenance=p13.load_inputs(scope)
    opportunities=s6.read_json(ROOT/f"outputs/stage13/{scope}/opportunities.json")
    past={int(k):v for k,v in opportunities["past"].items()}; target={int(k):v for k,v in opportunities["target"].items()}
    factors=s9.factors_for(past,target,tuple(opportunities["training_window"]),tuple(opportunities["target_window"]),data["val_candidates"])
    assert factors=={int(k):v for k,v in opportunities["factors"].items()}
    R=positive_normalize(s9.adjust_repeat_scores(repeat,factors,data["val_candidates"]),data["val_candidates"])
    original=r11.int_scores(ROOT/f"outputs/stage11/cache/{scope}/association_scores.json")
    original_norm=positive_normalize(original,data["val_candidates"])
    E=p13.component(raw,data["val_candidates"],factors,True)
    baseline=s6.read_json(ROOT/f"outputs/stage13/{scope}/methods/E_EASE_opportunity/top5.json")
    control=rank_hybrid(data,R,original_norm,E)[0]
    assert control==baseline,"Exact Stage 13 control reproduction failed"
    return data,repeat,R,original,original_norm,E,baseline,factors,provenance


def persist(scope,params,predictions,data,baseline,repeat,fallback,runtime,rule_details,category):
    base=f"{scope}/methods/{params['name']}"; save(f"{base}/top5.json",predictions)
    metrics,aps=s9.independent_metrics(s6.read_json(OUT/f"{base}/top5.json"),data["labels"]["primary_top5"],data["cohort"])
    save(f"{base}/metrics.json",metrics); save(f"{base}/per_user_ap5.json",aps); save(f"{base}/parameters.json",params)
    save(f"{base}/fallback.json",fallback); save(f"{base}/runtime.json",runtime); save(f"{base}/rule_details.json",rule_details)
    save(f"{base}/recommendation_changes.json",s8.prediction_changes(predictions,baseline,data["cohort"]))
    save(f"{base}/subgroups.json",r11.subgroup_results(data,repeat,baseline,predictions))
    save(f"{base}/provenance.json",{"category":category,"training_cutoff":data["cutoff"],"labels_as_features":False,
         "declaration":"declaration.json","source_manifest":"preservation_before.json","repeat_EASE_unchanged":True})
    save(f"{base}/complete.json",{"completed":True})
    return {"name":params["name"],"role":params["role"],"MAP":metrics["map_at_k"]}


def evaluate():
    assert s6.read_json(OUT/"declaration.json")["new_recommendation_configurations"]==CONFIGS
    if (OUT/"results.json").exists(): return
    model=ar_module(); all_results={}
    for scope in SCOPES:
        if (OUT/f"{scope}/results.json").exists(): all_results[scope]=s6.read_json(OUT/f"{scope}/results.json"); continue
        data,repeat,R,original,original_norm,E,baseline,factors,provenance=load_inputs(scope)
        save(f"{scope}/input_provenance.json",provenance); save(f"{scope}/labels.json",data["labels"])
        save(f"{scope}/cohort.json",data["cohort"]); save(f"{scope}/candidates.json",data["val_candidates"])
        control,fb,secs=rank_hybrid(data,R,original_norm,E)
        control_params={"name":"stage13_control","role":"control","variant":"control"}
        control_metrics,control_AP=s9.independent_metrics(baseline,data["labels"]["primary_top5"],data["cohort"])
        frozen=s6.read_json(ROOT/f"outputs/stage13/{scope}/methods/E_EASE_opportunity/metrics.json")
        assert all(abs(control_metrics[k]-frozen[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive"))
        persist(scope,control_params,control,data,baseline,repeat,fb,{"direct_rank_seconds":secs},{},"control_reproduction")
        results=[]
        for definition in MODELS:
            variant=definition["variant"]
            normalized_path=OUT/f"{scope}/AR_models/{variant}/normalized_scores.json"
            if normalized_path.exists() and (normalized_path.parent/"runtime.json").exists():
                normalized=r11.int_scores(normalized_path); rule_details=s6.read_json(normalized_path.parent/"details.json")
                model_seconds=s6.read_json(normalized_path.parent/"runtime.json")["direct_AR_fit_score_normalize_seconds"]
            else:
                def fit():
                    started=time.perf_counter()
                    if variant=="A":
                        raw_scores=opportunity_adjust(original,factors,data["val_candidates"])
                        original_details=s6.read_json(ROOT/f"outputs/stage8/{scope}/association_details.json")
                        details={**original_details,"rule_learning_unchanged":True,"adjust_completed_scores_only":True}
                    else:
                        fitted=model.fit(data["count_variants"]["primary80"],definition["K"],data["val_candidates"])
                        original_population=s6.read_json(ROOT/f"outputs/stage8/{scope}/association_details.json")["training_users"]
                        assert fitted["n_users"]==original_population==len(data["count_variants"]["primary80"])
                        raw_scores,details=model.user_scores(data,fitted)
                        model.save_model(OUT/f"{scope}/AR_models/{variant}",fitted)
                    normalized=positive_normalize(raw_scores,data["val_candidates"])
                    elapsed=time.perf_counter()-started
                    save(f"{scope}/AR_models/{variant}/raw_scores.json",raw_scores)
                    save(f"{scope}/AR_models/{variant}/normalized_scores.json",normalized)
                    save(f"{scope}/AR_models/{variant}/details.json",details)
                    save(f"{scope}/AR_models/{variant}/runtime.json",{"direct_AR_fit_score_normalize_seconds":elapsed})
                    return normalized,details,elapsed
                normalized,rule_details,model_seconds=attempt(scope,f"{variant}_AR_fit","shared_association_fit_not_new_recommendation",fit)
            for params in [p for p in CONFIGS if p["variant"]==variant]:
                base=OUT/f"{scope}/methods/{params['name']}"
                if (base/"complete.json").exists():
                    results.append({"name":params["name"],"role":params["role"],"MAP":s6.read_json(base/"metrics.json")["map_at_k"]}); continue
                def run(params=params):
                    started=time.perf_counter()
                    if params["role"]=="standalone": predicted,fallback=s6.rank_scores(normalized,data); rank_seconds=time.perf_counter()-started
                    else: predicted,fallback,rank_seconds=rank_hybrid(data,R,normalized,E)
                    return persist(scope,params,predicted,data,baseline,repeat,fallback,
                         {"direct_prediction_seconds":rank_seconds,"shared_AR_model_seconds":model_seconds,
                          "measured_cost_for_selection_seconds":model_seconds+rank_seconds,
                          "scope":"shared AR model reused by standalone and hybrid; component-attributed, not independently timed end-to-end"},rule_details,
                         "new_unique" if scope=="horizon12" else "repeated_backtest_evaluation")
                results.append(attempt(scope,params["name"],"new_unique" if scope=="horizon12" else "repeated_backtest_evaluation",run))
                checkpoint("evaluating",scope=scope,last_completed=params["name"])
        users=data["cohort"]; labels=data["labels"]["primary_top5"]; arrays=[]
        for params in CONFIGS:
            aps=s6.read_json(OUT/f"{scope}/methods/{params['name']}/per_user_ap5.json")
            arrays.append(np.array([aps[u]-control_AP[u] for u in users]))
        delta=np.column_stack(arrays); intervals={}
        for group,mask in {"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}.items():
            bounds=s8.paired_bootstrap(delta[mask],SEED,RESAMPLES)
            intervals[group]={p["name"]:{"delta_MAP":float(delta[mask,i].mean()),"paired_95_interval":bounds[i],"users":int(mask.sum())} for i,p in enumerate(CONFIGS)}
        save(f"{scope}/paired_bootstrap.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":intervals,"selection_adjusted":False})
        save(f"{scope}/results.json",results); all_results[scope]=results; checkpoint(f"{scope}_complete")
        del data,E; gc.collect()
    save("results.json",all_results)


def select():
    if (OUT/"selection.json").exists(): return
    comparison=[]
    for params in CONFIGS:
        row={**params}; costs=[]
        for scope in SCOPES:
            metric=s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")
            interval=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]["all_users"][params["name"]]
            row[f"{scope}_MAP"]=metric["map_at_k"]; row[f"{scope}_gain"]=interval["delta_MAP"]; row[f"{scope}_interval"]=interval["paired_95_interval"]
            costs.append(s6.read_json(OUT/f"{scope}/methods/{params['name']}/runtime.json")["measured_cost_for_selection_seconds"])
        row["eligible"]=params["role"]=="hybrid" and row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0
        row["minimum_gain"]=min(row["horizon12_gain"],row["original10_gain"]); row["measured_cost_sum_seconds"]=sum(costs); comparison.append(row)
    eligible=[x for x in comparison if x["eligible"]]; winner=None
    if eligible:
        best=max(x["minimum_gain"] for x in eligible); tied=[x for x in eligible if x["minimum_gain"]==best]
        cost=min(x["measured_cost_sum_seconds"] for x in tied); tied=[x for x in tied if x["measured_cost_sum_seconds"]==cost]
        if len(tied)==1: winner=next(p for p in CONFIGS if p["name"]==tied[0]["name"])
    save("comparison.json",comparison); save("selection.json",{"selected":winner,"comparison":comparison,
         "decision":"freeze and refit selected hybrid" if winner else "retain Stage 13; no qualifying hybrid",
         "standalone_candidates_for_refit":False,"rule_source":"declaration.json","independent_confirmation":False})
    checkpoint("selection_complete",selected=winner)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("prepare","evaluate","select")); phase=parser.parse_args().phase
    if phase=="prepare": prepare()
    elif phase=="evaluate": evaluate()
    else: select()


if __name__=="__main__":main()
