#!/usr/bin/env python3
"""Fixed log-count EASE study; Stage 11 controls are read-only reused results."""
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
spec=importlib.util.spec_from_file_location("stage11_readonly",ROOT/"outputs/stage11/code/run.py")
r11=importlib.util.module_from_spec(spec); spec.loader.exec_module(r11)
s6,s8,s9,np,e=r11.s6,r11.s8,r11.s9,r11.np,r11.e
OUT=ROOT/"outputs/stage12"
# r11.inputs is read-only and intentionally retains its Stage 11 cache root.
# Redirect older helper writers used only by a conditional full-history refit.
s8.OUT=OUT/"cache"; s6.OUT=OUT/"cache"; s9.OUT=OUT/"cache"
BEST=ROOT/"outputs/stage9/kuznetsov_vladimir_midterm_submission_schedule.csv"
BEST_SHA="f88ac045cd9c4347e72fd7a3a90f4c97748d44a055af8faf5c93a684fa0f9ea9"
DEST=OUT/"kuznetsov_vladimir_midterm_submission_frequency_ease.csv"
SCOPES=("horizon12","original10")
CONFIGS=[{"name":f"logcount_ease_l{reg}_b{round(beta*100)}","lambda":reg,"beta":beta}
         for reg in (100,1000) for beta in (.1,1.)]
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
    assert s6.read_json(ROOT/"outputs/stage11/search_history.json")["total_unique"]==49
    save("preservation_before.json",manifest())
    catalog=s6.read_json(ROOT/"outputs/stage8/search_history.json")["configurations"]
    inspected=[]
    for row in catalog:
        path=ROOT/row["parameters_path"]
        inspected.append({"name":row["name"],"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path)})
    for stage in (10,11):
        for row in s6.read_json(ROOT/f"outputs/stage{stage}/search_history.json")["new_configurations"]:
            path=ROOT/f"outputs/stage{stage}/horizon12/methods/{row['name']}/parameters.json"
            inspected.append({"name":row["name"],"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path)})
    assert len(inspected)==49
    prior={}
    for stage,file in ((6,"plan.json"),(7,"plan.json"),(8,"plan.json"),(9,"configuration.json"),(10,"declaration.json"),(11,"declaration.json")):
        path=ROOT/f"outputs/stage{stage}/{file}"
        prior[str(path.relative_to(ROOT))]={"sha256":s6.sha256_file(path),"declaration":s6.read_json(path)}
    assert "binary float64" in prior["outputs/stage11/declaration.json"]["declaration"]["implementation"]["X"]
    assert not any("logcount_ease" in row["name"] for row in inspected)
    save("duplicate_audit.json",{"previous_49_configurations":inspected,"prior_declarations":prior,
         "exact_duplicates":{p["name"]:[] for p in CONFIGS},
         "note":"Matching lambda/beta Stage 11 controls use binary inputs, not training-scaled log-count inputs; not exact duplicates.",
         "duplicate_or_failed_slots_replaced":False})
    save("declaration.json",{"declared_before_model_evaluation":True,"suggestion_source":"external AI assistant Codex",
         "personal_analysis_or_verification_claimed":False,"personal_verification":"pending",
         "hypothesis":"preserving >=80% qualifying-airing frequency in EASE input may complement Stage 9 more effectively than binary EASE",
         "configurations":CONFIGS,"previous_unique":49,"new_unique_budget":4,"maximum_total_unique":53,
         "only_change":"training input representation relative to matched Stage 11 models",
         "input":{"count":"number of distinct qualifying training user-airing interactions at >=80% actual observed coverage",
              "duplicate_policy":"unioned user-airing cache, deduplicate representations of (user_idx,airing_idx); distinct repeated airings each count once",
              "transform":"Z_ui=log1p(n_ui)","scale":"sqrt(nnz(Z)/sum(Z_ui**2))","X":"scale*Z",
              "scale_fit_scope":"all qualifying training users/programs only, before cutoff; no labels or target data",
              "axes":"same positive support and learned axes as Stage 11; noncohort users and noncandidate historical programs included",
              "degenerate":"no positive entries => scale=0, empty X, no solve, all EASE scores zero, established fallback; no division by zero",
              "convention":"global normalization is an explicitly declared experiment convention, not standard paper preprocessing or proven optimal"},
         "fixed":{"feature_coverage":.8,"dtype":"float64","solver":"Stage 11 Cholesky solve of G+lambda I against identity",
              "B":"column-wise -P_ij/P_jj, exact zero diagonal; negative coefficients retained",
              "prediction":"X_user @ B with weighted query rows; filter target candidates then clip completed scores and max-normalize positives",
              "seen_items_excluded":False,"EASE_schedule_adjusted":False,
              "fusion":"(1-beta)*S9 + beta*normalized_positive_EASE; S9 is not additionally normalized",
              "S9":"exact Stage 9 adjusted repeat50/AR80, 70/30, no decay; no setting changes",
              "fallback_and_ranking":s6.fusion_rules()},
         "splits":"unchanged Stage 8 horizon12 and original10 users, labels, pools, boundaries and evaluation rules",
         "controls":"reuse Stage 9 and matched Stage 11 binary lambda/beta predictions; independently recompute their metrics, do not refit controls",
         "comparison":"three-cohort metrics and per-user AP; paired deltas vs Stage 9; matched log-count minus binary differences; relevant-pair repeat50 partitions are diagnostic only",
         "bootstrap":{"seed":SEED,"resamples":RESAMPLES,"interval":"paired-user percentile descriptive 95%","order":"template order","cohorts":"all, positive, >=5 positives"},
         "selection":{"gain_each_split_at_least":.002,"horizon12_interval_lower_strictly_above":0.,
              "objective":"maximize smaller gain across the two splits","tie_break_1":"lower beta",
              "tie_break_2":"lower measured computational cost summed across splits: shared EASE fit/score plus direct fusion",
              "remaining_exact_tie":"retain Stage 9 rather than invent another rule"},
         "resources":"same sequential fits, matrix memory estimate and fixed cap min(4 GiB, physical_memory/4) as Stage 11; no replacement on failure",
         "uncertainty":"overlapping, repeatedly used historical periods; selection-unadjusted descriptive intervals, not independent confirmation",
         "kaggle_scores_not_targets":True,"conditional_refit":"only one qualifying winner; fresh full-history scalar and S9 factors; no CSV if none",
         "output":str(DEST.relative_to(ROOT))})
    save("source_attribution.json",{"suggestion":"external AI assistant Codex suggested testing frequency-preserving EASE",
         "implementation_diagnostics_verification":"performed in this OpenCode session using only project data/code",
         "personal_user_analysis_or_verification_claimed":False,"personal_verification":"pending",
         "external_diagnostic_files_read":False,"normalization":"user-declared experimental convention, not a standard/proven-optimal preprocessing claim",
         "paper_for_unchanged_EASE_solver":"Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data, WWW 2019, https://arxiv.org/abs/1905.03375"})
    save("current_best_submission.json",{"stage":9,"path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA,"private":.41656,"public":.43185,
         "designation":"best scored, unchanged until a later returned score establishes otherwise"})
    checkpoint("four_configurations_declared")


def count_module():
    spec=importlib.util.spec_from_file_location("stage12_count_input",OUT/"code/count_input.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def load_inputs(scope):
    data,S9,repeat,baseline,provenance=r11.inputs(scope)
    expected_stage9=s6.read_json(ROOT/f"outputs/stage11/{scope}/methods/stage9_control/top5.json")
    assert expected_stage9==baseline
    assert r11.combine(data,S9,None,0)[0]==baseline
    # Matching binary controls are reused, never refitted or re-ranked.
    controls={"stage9":baseline}
    for params in CONFIGS:
        name=f"ease_l{params['lambda']}_b{round(params['beta']*100)}"
        controls[params["name"]]=s6.read_json(ROOT/f"outputs/stage11/{scope}/methods/{name}/top5.json")
    return data,S9,repeat,baseline,controls,provenance


def audit_training_counts(scope,data,module):
    if scope=="original10":
        table,_=s6.load_table(ROOT/"outputs/stage3/interactions/train_user_airing.npz",ROOT/"outputs/stage3/airings/airings_train.csv")
    else:
        cfg=s6.load_config(ROOT/"config/stage5.json")
        table,_=s6.load_table(cfg.out("interactions","history_user_airing.npz"),cfg.out("airings","airings_history.csv"))
    reconstructed,audit=module.count_distinct_airings(table,s6.START,data["cutoff"])
    expected=data["count_variants"]["primary80"]
    audit["equals_verified_training_count_cache"]=reconstructed==expected
    assert audit["equals_verified_training_count_cache"]
    save(f"{scope}/training_count_audit.json",audit)
    return audit


def persist(scope,params,predictions,data,baseline,binary,repeat,fallback,runtime):
    name=params["name"]; base=f"{scope}/methods/{name}"
    save(f"{base}/top5.json",predictions)
    metrics,aps=s9.independent_metrics(s6.read_json(OUT/f"{base}/top5.json"),data["labels"]["primary_top5"],data["cohort"])
    binary_metrics,binary_aps=s9.independent_metrics(binary,data["labels"]["primary_top5"],data["cohort"])
    save(f"{base}/metrics.json",metrics); save(f"{base}/per_user_ap5.json",aps)
    save(f"{base}/parameters.json",{**params,"input":"training-scaled log1p primary80 qualifying-airing counts"})
    save(f"{base}/fallback.json",fallback); save(f"{base}/runtime.json",runtime)
    save(f"{base}/matched_binary_comparison.json",{"binary_source":f"outputs/stage11/{scope}/methods/ease_l{params['lambda']}_b{round(params['beta']*100)}",
         "binary_metrics":binary_metrics,"count_minus_binary_metrics":{k:metrics[k]-binary_metrics[k] for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive")},
         "prediction_changes":s8.prediction_changes(predictions,binary,data["cohort"])})
    save(f"{base}/count_minus_binary_per_user_ap5.json",{u:aps[u]-binary_aps[u] for u in aps})
    save(f"{base}/recommendation_changes_vs_stage9.json",s8.prediction_changes(predictions,baseline,data["cohort"]))
    save(f"{base}/subgroups_vs_stage9.json",r11.subgroup_results(data,repeat,baseline,predictions))
    save(f"{base}/subgroups_vs_binary.json",r11.subgroup_results(data,repeat,binary,predictions))
    save(f"{base}/provenance.json",{"training_cutoff":data["cutoff"],"labels_used_as_features":False,"scalar_source":f"{scope}/models/input_statistics.json",
         "S9_renormalized":False,"source_manifest":"preservation_before.json","declaration":"declaration.json"})
    save(f"{base}/complete.json",{"completed":True})
    return {"name":name,"status":"completed","MAP":metrics["map_at_k"]}


def evaluate():
    assert s6.read_json(OUT/"declaration.json")["configurations"]==CONFIGS
    module=count_module(); all_results={}
    for scope in SCOPES:
        if (OUT/f"{scope}/results.json").exists(): all_results[scope]=s6.read_json(OUT/f"{scope}/results.json"); continue
        data,S9,repeat,baseline,controls,provenance=load_inputs(scope)
        save(f"{scope}/input_provenance.json",provenance)
        audit_training_counts(scope,data,module)
        save(f"{scope}/labels.json",data["labels"]); save(f"{scope}/cohort.json",data["cohort"]); save(f"{scope}/candidates.json",data["val_candidates"])
        for name,predictions in controls.items():
            metrics,aps=s9.independent_metrics(predictions,data["labels"]["primary_top5"],data["cohort"])
            save(f"{scope}/controls/{name}/metrics.json",metrics); save(f"{scope}/controls/{name}/per_user_ap5.json",aps)
            if name=="stage9": old_name="stage9_control"
            else:
                matched_params=next(p for p in CONFIGS if p["name"]==name)
                old_name=f"ease_l{matched_params['lambda']}_b{round(matched_params['beta']*100)}"
            source=ROOT/f"outputs/stage11/{scope}/methods/{old_name}/top5.json"
            save(f"{scope}/controls/{name}/provenance.json",{"source":str(source.relative_to(ROOT)),"sha256":s6.sha256_file(source),"reused_without_model_refit":True})
        plan=r11.resource_plan(data); save(f"{scope}/memory_estimate.json",plan)
        if not plan["allowed"]: raise MemoryError("Declared weighted EASE does not fit budget; no replacement authorized")
        started=time.perf_counter()
        X,learned_users,learned_items,statistics=module.transformed_matrix(data["count_variants"]["primary80"])
        G=module.gram_matrix(X); matrix_seconds=time.perf_counter()-started
        previous=s6.read_json(ROOT/f"outputs/stage11/{scope}/models/mappings.json")
        assert learned_users==previous["learned_user_training_indices"] and learned_items==previous["learned_program_ids"]
        from scipy.sparse import save_npz
        (OUT/f"{scope}/models").mkdir(parents=True,exist_ok=True); save_npz(OUT/f"{scope}/models/training_X.npz",X)
        save(f"{scope}/models/input_statistics.json",statistics)
        save(f"{scope}/models/mappings.json",{"learned_user_training_indices":learned_users,"learned_program_ids":learned_items,
             "nnz":X.nnz,"dtype":str(X.dtype),"Gram_dtype":str(G.dtype),"matrix_transform_and_Gram_seconds":matrix_seconds})
        results=[]
        for reg in (100,1000):
            params_for_fit=[p for p in CONFIGS if p["lambda"]==reg]
            if all((OUT/f"{scope}/methods/{p['name']}/complete.json").exists() for p in params_for_fit):
                results.extend({"name":p["name"],"status":"completed","MAP":s6.read_json(OUT/f"{scope}/methods/{p['name']}/metrics.json")["map_at_k"]} for p in params_for_fit); continue
            directory=OUT/f"{scope}/models/lambda{reg}"
            if (directory/"scores.npz").exists() and (directory/"runtime.json").exists():
                with np.load(directory/"scores.npz") as saved: normalized=saved["normalized"]
                fit_seconds=s6.read_json(directory/"runtime.json")["direct_fit_and_score_seconds"]
            else:
                def fit():
                    t=time.perf_counter()
                    if X.nnz==0:
                        normalized=np.zeros((len(data["cohort"]),len(data["val_candidates"])),dtype=np.float64); raw=normalized.copy()
                        fit_audit={"degenerate":True,"no_solve":True}; score_audit={"all_scores_zero":True}
                    else:
                        B,fit_audit=e.fit_from_gram(G,reg)
                        normalized,raw,score_audit=e.candidate_scores(X,B,learned_users,learned_items,data["cohort"],data["user_to_train_idx"],data["val_candidates"])
                        del B
                    elapsed=time.perf_counter()-t
                    save(f"{scope}/models/lambda{reg}/fit_audit.json",fit_audit); save(f"{scope}/models/lambda{reg}/score_audit.json",score_audit)
                    np.savez_compressed(directory/"scores.npz",normalized=normalized,raw=raw)
                    save(f"{scope}/models/lambda{reg}/runtime.json",{"direct_fit_and_score_seconds":elapsed,"score_artifact_compression_excluded":True})
                    return normalized,elapsed
                normalized,fit_seconds=attempt(scope,f"weighted_EASE_lambda{reg}","shared_fit_not_new_configuration",fit)
            for params in params_for_fit:
                if (OUT/f"{scope}/methods/{params['name']}/complete.json").exists():
                    results.append({"name":params["name"],"status":"completed","MAP":s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")["map_at_k"]}); continue
                def run(params=params):
                    predicted,fallback,seconds=r11.combine(data,S9,normalized,params["beta"])
                    return persist(scope,params,predicted,data,baseline,controls[params["name"]],repeat,fallback,
                         {"direct_fusion_and_rank_seconds":seconds,"shared_fit_and_score_seconds":fit_seconds,"shared_matrix_transform_Gram_seconds":matrix_seconds,
                          "measured_cost_for_selection_seconds":fit_seconds+seconds,"scope":"shared fit reused across beta variants; not standalone end-to-end cost"})
                results.append(attempt(scope,params["name"],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run))
                checkpoint("evaluating",scope=scope,last_completed=params["name"])
            del normalized; gc.collect()
        intervals={}; binary_intervals={}
        users=data["cohort"]; labels=data["labels"]["primary_top5"]
        base_AP=s6.read_json(OUT/f"{scope}/controls/stage9/per_user_ap5.json")
        deltas=[]; matched=[]
        for params in CONFIGS:
            aps=s6.read_json(OUT/f"{scope}/methods/{params['name']}/per_user_ap5.json")
            binary=s6.read_json(OUT/f"{scope}/controls/{params['name']}/per_user_ap5.json")
            deltas.append(np.array([aps[u]-base_AP[u] for u in users])); matched.append(np.array([aps[u]-binary[u] for u in users]))
        delta=np.column_stack(deltas); match=np.column_stack(matched)
        for group,mask in {"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}.items():
            bounds=s8.paired_bootstrap(delta[mask],SEED,RESAMPLES); paired=s8.paired_bootstrap(match[mask],SEED,RESAMPLES)
            intervals[group]={p["name"]:{"delta_MAP":float(delta[mask,i].mean()),"paired_95_interval":bounds[i],"users":int(mask.sum())} for i,p in enumerate(CONFIGS)}
            binary_intervals[group]={p["name"]:{"count_minus_binary_MAP":float(match[mask,i].mean()),"paired_95_interval":paired[i],"users":int(mask.sum())} for i,p in enumerate(CONFIGS)}
        save(f"{scope}/paired_bootstrap_vs_stage9.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":intervals,"selection_adjusted":False})
        save(f"{scope}/paired_bootstrap_vs_binary.json",{"seed":SEED,"resamples":RESAMPLES,"cohorts":binary_intervals,"diagnostic_only":True})
        save(f"{scope}/results.json",results); all_results[scope]=results; checkpoint(f"{scope}_complete")
        del X,G,data; gc.collect()
    save("results.json",all_results)


def select():
    if (OUT/"selection.json").exists(): return
    comparison=[]
    for params in CONFIGS:
        row={**params}; cost=0.
        for scope in SCOPES:
            metrics=s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")
            interval=s6.read_json(OUT/f"{scope}/paired_bootstrap_vs_stage9.json")["cohorts"]["all_users"][params["name"]]
            binary=s6.read_json(OUT/f"{scope}/methods/{params['name']}/matched_binary_comparison.json")
            row[f"{scope}_MAP"]=metrics["map_at_k"]; row[f"{scope}_gain"]=interval["delta_MAP"]; row[f"{scope}_interval"]=interval["paired_95_interval"]
            row[f"{scope}_binary_MAP"]=binary["binary_metrics"]["map_at_k"]; row[f"{scope}_count_minus_binary"]=binary["count_minus_binary_metrics"]["map_at_k"]
            cost+=s6.read_json(OUT/f"{scope}/methods/{params['name']}/runtime.json")["measured_cost_for_selection_seconds"]
        row["eligible"]=row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0
        row["minimum_gain"]=min(row["horizon12_gain"],row["original10_gain"]); row["measured_cost_sum_seconds"]=cost; comparison.append(row)
    eligible=[row for row in comparison if row["eligible"]]; winner=None
    if eligible:
        best=max(x["minimum_gain"] for x in eligible); tied=[x for x in eligible if x["minimum_gain"]==best]
        beta=min(x["beta"] for x in tied); tied=[x for x in tied if x["beta"]==beta]
        cost=min(x["measured_cost_sum_seconds"] for x in tied); tied=[x for x in tied if x["measured_cost_sum_seconds"]==cost]
        if len(tied)==1: winner=next(p for p in CONFIGS if p["name"]==tied[0]["name"])
    save("comparison.json",comparison); save("selection.json",{"selected":winner,"comparison":comparison,"rule_source":"declaration.json",
         "decision":"freeze and refit selected fixed configuration" if winner else "retain Stage 9; no qualifying winner",
         "binary_differences_and_subgroups_used_for_selection":False,"independent_confirmation":False})
    checkpoint("selection_complete",selected=winner)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("prepare","evaluate","select")); phase=parser.parse_args().phase
    if phase=="prepare": prepare()
    elif phase=="evaluate": evaluate()
    else: select()


if __name__=="__main__":main()
