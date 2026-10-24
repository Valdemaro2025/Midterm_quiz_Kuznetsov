#!/usr/bin/env python3
"""Resume the saved Stage 11 declaration; never rerun diagnosis or redeclare."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("stage11_saved_study",HERE/"study.py")
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
spec=importlib.util.spec_from_file_location("stage11_ease",HERE/"ease_model.py")
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
ROOT,OUT,np=s.ROOT,s.OUT,s.np
s6,s8,s9=s.s6,s.s8,s.s9


def save(path,value): s.save(path,value)


def resume_record():
    path=OUT/"resume_state.json"
    if path.exists(): return
    declaration=s6.read_json(OUT/"declaration.json")
    assert declaration["configurations"]==s.CONFIGS
    assert (OUT/"diagnostics/complete.json").exists()
    protected=[OUT/"declaration.json",OUT/"duplicate_audit.json",OUT/"diagnostics/complete.json",
               OUT/"diagnostics/horizon12.json",OUT/"diagnostics/original10.json",OUT/"source_attribution.json"]
    save("resume_state.json",{"checkpoint_found":s6.read_json(OUT/"checkpoints/progress.json"),
         "initial_attempt_log":(OUT/"attempts.jsonl").read_text(),
         "protected_stage11_files":{str(p.relative_to(OUT)):s6.sha256_file(p) for p in protected},
         "initial_state":"diagnosis and declaration completed; no EASE fit or candidate evaluation recorded",
         "reported_interruption":"unknown certificate verification error; no model failure recorded on disk",
         "diagnostics_repeated":False,"declaration_replaced":False})


def int_scores(path):
    return {u:{int(k):float(v) for k,v in rows.items()} for u,rows in s6.read_json(path).items()}


def inputs(scope):
    """Use valid existing preprocessing; labels remain evaluation-only."""
    started=time.perf_counter()
    cache=OUT/f"cache/{scope}"
    users=s6.read_json(cache/"cohort.json"); candidates=s6.read_json(cache/"candidates.json")
    labels=s6.read_json(cache/"labels.json")
    for filename,value in (("cohort.json",users),("candidates.json",candidates),("labels.json",labels)):
        assert value==s6.read_json(ROOT/f"outputs/stage8/{scope}/{filename}")
    if scope=="original10":
        loaded=s6.load_stage4_data(s6.load_config(ROOT/"config/stage3.json"))
        primary=loaded["count_variants"]["primary80"]; mapping=loaded["user_to_train_idx"]
        popularity_ids,popularity_stats=loaded["popularity_val"],loaded["popularity"]
        cutoff=s6.PRIMARY_END
        table,_=s6.load_table(ROOT/"outputs/stage3/interactions/train_user_airing.npz",ROOT/"outputs/stage3/airings/airings_train.csv")
        assert np.all(table["end"]<=cutoff)
        max_end=int(table["end"].max()); del table
    else:
        cfg=s6.load_config(ROOT/"config/stage5.json")
        table,_=s6.load_table(cfg.out("interactions","history_user_airing.npz"),cfg.out("airings","airings_history.csv"))
        cutoff=s8.HORIZON_CUTOFF
        train=s6.slice_table(table,s6.START,cutoff)
        primary=s6.aggregate(train,cutoff,{"threshold":.8,"weighting":"count"})
        assert np.all(train["end"]<=cutoff)
        max_end=int(train["end"].max())
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        # Identity lookup can contain other historical IDs; learned matrix rows
        # below are only users with actual pre-cutoff >=80% observations.
        mapping={u:i for i,u in enumerate(ids)}
        variants={"primary80":primary}
        popularity_ids,popularity_stats,pop_rows=s6.build_popularity(variants,candidates)
        assert pop_rows==s6.read_json(cache/"popularity.json")
        del table,train
    data={"cohort":users,"labels":labels,"val_candidates":candidates,"count_variants":{"primary80":primary},
          "user_to_train_idx":mapping,"popularity_val":popularity_ids,"popularity":popularity_stats,"cutoff":cutoff}
    baseline=int_scores(cache/"S9_scores.json"); repeat=int_scores(cache/"repeat50_scores.json")
    expected=s6.read_json(ROOT/f"outputs/stage9/reproduction/{scope}/final_implementation_top5.json")
    provenance={"training_cutoff":cutoff,"max_training_airing_end":max_end,"labels_used_as_training_features":False,
                "input_loading_and_training_aggregation_seconds":time.perf_counter()-started,
                "feature_threshold":">=80% actual observed coverage","repeat_history_threshold":">=50% actual observed coverage"}
    return data,baseline,repeat,expected,provenance


def resource_plan(data):
    dimensions=e.dimensions(data["count_variants"]["primary80"])
    estimate=e.memory_estimate(*dimensions,len(data["cohort"]),len(data["val_candidates"]))
    try: physical=int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True).strip())
    except (OSError,subprocess.CalledProcessError): physical=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    budget=min(4*1024**3,physical//4)
    return {**estimate,"training_users":dimensions[0],"training_programs":dimensions[1],"binary_nnz":dimensions[2],
            "physical_memory_bytes":physical,"fixed_fit_budget_bytes":budget,"allowed":estimate["conservative_peak_bytes"]<=budget}


def combine(data,S9,normalized,beta):
    """No renormalization of S9; EASE was normalized after candidate filtering."""
    started=time.perf_counter(); predictions={}; fallback_users=slots=scored=0
    for i,u in enumerate(data["cohort"]):
        fused={j:(1-beta)*v for j,v in S9[u].items() if v>0} if beta<1 else {}
        if beta>0:
            for j,value in zip(data["val_candidates"],normalized[i]):
                if value>0: fused[j]=fused.get(j,0)+beta*float(value)
        top,info=s6.top_k(fused,data["val_candidates"],data["popularity_val"],data["popularity"])
        predictions[u]=top; fallback_users+=int(info["used_fallback"]); slots+=info["fallback_filled"]; scored+=info["scored_candidates"]
    return predictions,{"users_needing_fallback":fallback_users,"fallback_slots":slots,
                        "mean_scored_candidates_per_user":scored/len(predictions),"S9_renormalized":False},time.perf_counter()-started


def contributions(predictions,relevant):
    result={}; seen=set(); hits=0; denominator=min(5,len(set(relevant)))
    if not denominator: return result
    for rank,item in enumerate(predictions[:5],1):
        if item in relevant and item not in seen:
            hits+=1; result[item]=hits/rank/denominator
        seen.add(item)
    return result


def subgroup_results(data,repeat,baseline,predictions):
    primary=data["count_variants"]["primary80"]
    global_items={j for rows in primary.values() for j in rows}
    partitions={"personal_repeat50":{},"global_primary80":{}}
    for key,names in (("personal_repeat50",("with_personal_repeat50","without_personal_repeat50")),
                      ("global_primary80",("globally_observed","globally_unobserved"))):
        for name in names:
            partitions[key][name]={"pairs":0,"baseline_hits":0,"candidate_hits":0,"hits_gained":0,"hits_lost":0,
                                   "baseline_AP_contribution_sum":0.,"candidate_AP_contribution_sum":0.,"programs":set()}
    users=data["cohort"]; labels=data["labels"]["primary_top5"]
    base_metrics,base_aps=s9.independent_metrics(baseline,labels,users)
    new_metrics,new_aps=s9.independent_metrics(predictions,labels,users)
    for u in users:
        relevant=set(labels[u]); old=contributions(baseline[u],relevant); new=contributions(predictions[u],relevant)
        for j in relevant:
            names={"personal_repeat50":"with_personal_repeat50" if repeat[u].get(j,0)>0 else "without_personal_repeat50",
                   "global_primary80":"globally_observed" if j in global_items else "globally_unobserved"}
            for key,name in names.items():
                row=partitions[key][name]; row["pairs"]+=1; row["programs"].add(j)
                a,b=j in old,j in new
                row["baseline_hits"]+=a; row["candidate_hits"]+=b
                row["hits_gained"]+=b and not a; row["hits_lost"]+=a and not b
                row["baseline_AP_contribution_sum"]+=old.get(j,0); row["candidate_AP_contribution_sum"]+=new.get(j,0)
    for rows in partitions.values():
        for row in rows.values():
            row["distinct_relevant_programs"]=len(row.pop("programs"))
            row["net_hits"]=row["candidate_hits"]-row["baseline_hits"]
            row["AP_contribution_delta_per_all_users"]=(row["candidate_AP_contribution_sum"]-row["baseline_AP_contribution_sum"])/len(users)
    user_groups={}
    for name,condition in (("fewer_than20",lambda n:n<20),("at_least20",lambda n:n>=20)):
        selected=[u for u in users if condition(len(repeat[u]))]
        changes=np.array([new_aps[u]-base_aps[u] for u in selected])
        user_groups[name]={"users":len(selected),"baseline_MAP":float(np.mean([base_aps[u] for u in selected])) if selected else 0.,
            "candidate_MAP":float(np.mean([new_aps[u] for u in selected])) if selected else 0.,
            "delta_MAP":float(changes.mean()) if selected else 0.,"users_better":int(np.sum(changes>1e-14)),
            "users_worse":int(np.sum(changes<-1e-14)),"users_tied":int(np.sum(abs(changes)<=1e-14))}
    return {"pair_partitions":partitions,"user_history_groups":user_groups,
            "definitions":{"personal_history":">=50% actual qualifying training coverage; distinct eligible programs",
                 "global_history":">=80% actual qualifying training coverage, any training user",
                 "AP_attribution":"each relevant item's contribution to ordinary AP; partition contributions sum to overall MAP delta; not new label definitions"},
            "selection_uses_subgroups":False}


def persist(scope,params,predictions,data,baseline,repeat,fallback,runtime,category):
    name=params["name"]; base=f"{scope}/methods/{name}"
    save(f"{base}/top5.json",predictions)
    metrics,aps=s9.independent_metrics(s6.read_json(OUT/f"{base}/top5.json"),data["labels"]["primary_top5"],data["cohort"])
    save(f"{base}/metrics.json",metrics); save(f"{base}/per_user_ap5.json",aps)
    save(f"{base}/parameters.json",params); save(f"{base}/fallback.json",fallback); save(f"{base}/runtime.json",runtime)
    save(f"{base}/recommendation_changes.json",s8.prediction_changes(predictions,baseline,data["cohort"]))
    save(f"{base}/subgroups.json",subgroup_results(data,repeat,baseline,predictions))
    save(f"{base}/provenance.json",{"category":category,"training_cutoff":data["cutoff"],"labels_training_features":False,
          "model":f"../../models/lambda{params.get('lambda','control')}","declaration":"declaration.json","S9_additionally_normalized":False})
    return {"name":name,"status":"completed","map_at_5":metrics["map_at_k"]}


def evaluate():
    resume_record()
    declaration=s6.read_json(OUT/"declaration.json"); assert declaration["configurations"]==s.CONFIGS
    duplicates=s6.read_json(OUT/"duplicate_audit.json")["exact_duplicates"]
    all_results={}
    for scope in s.SCOPES:
        if (OUT/f"{scope}/results.json").exists():
            all_results[scope]=s6.read_json(OUT/f"{scope}/results.json"); continue
        data,S9,repeat,baseline,provenance=inputs(scope)
        save(f"{scope}/input_provenance.json",provenance)
        control={"name":"stage9_control","beta":0}
        control_pred,control_fb,control_seconds=combine(data,S9,None,0)
        assert control_pred==baseline
        persist(scope,control,control_pred,data,baseline,repeat,control_fb,{"direct_rank_seconds":control_seconds},"fixed_control_reproduction")
        plan=resource_plan(data); save(f"{scope}/memory_estimate.json",plan)
        results=[]
        if not plan["allowed"]:
            for params in s.CONFIGS:
                def unavailable(params=params):
                    value={"name":params["name"],"status":"unavailable_resource_budget","memory_estimate":plan}
                    save(f"{scope}/methods/{params['name']}/status.json",value); return value
                results.append(s.attempt(scope,params["name"],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",unavailable))
            save(f"{scope}/results.json",results); all_results[scope]=results; continue
        matrix_start=time.perf_counter()
        X,learned_users,learned_items=e.from_training_counts(data["count_variants"]["primary80"])
        G=e.gram_matrix(X); matrix_seconds=time.perf_counter()-matrix_start
        from scipy.sparse import save_npz
        (OUT/f"{scope}/models").mkdir(parents=True,exist_ok=True)
        save_npz(OUT/f"{scope}/models/training_X.npz",X)
        save(f"{scope}/models/mappings.json",{"learned_user_training_indices":learned_users,"learned_program_ids":learned_items,
              "qualifying_training_users":len(learned_users),"noncohort_training_users":sum(u not in {data['user_to_train_idx'].get(x) for x in data['cohort']} for u in learned_users),
              "learned_noncandidate_programs":len(set(learned_items)-set(data["val_candidates"])),
              "binary_nnz":X.nnz,"dtype":str(X.dtype),"Gram_dtype":str(G.dtype),"Gram_max":float(G.max()),
              "binary_matrix_and_Gram_seconds":matrix_seconds})
        for reg in s.LAMBDAS:
            configs=[p for p in s.CONFIGS if p["lambda"]==reg]
            if all((OUT/f"{scope}/methods/{p['name']}/metrics.json").exists() for p in configs):
                results.extend({"name":p["name"],"status":"completed","map_at_5":s6.read_json(OUT/f"{scope}/methods/{p['name']}/metrics.json")["map_at_k"]} for p in configs)
                continue
            def fit():
                started=time.perf_counter(); B,fit_audit=e.fit_from_gram(G,reg)
                normalized,raw,score_audit=e.candidate_scores(X,B,learned_users,learned_items,data["cohort"],data["user_to_train_idx"],data["val_candidates"])
                elapsed=time.perf_counter()-started
                save(f"{scope}/models/lambda{reg}/fit_audit.json",fit_audit)
                save(f"{scope}/models/lambda{reg}/score_audit.json",score_audit)
                np.savez_compressed(OUT/f"{scope}/models/lambda{reg}/candidate_scores.npz",normalized=normalized,raw=raw)
                del B,raw
                return normalized,elapsed
            try: normalized,fit_score_seconds=s.attempt(scope,f"EASE_lambda{reg}","shared_model_fit_not_new_configuration",fit)
            except MemoryError:
                for params in configs:
                    value={"name":params["name"],"status":"unavailable_actual_memory_failure"}; save(f"{scope}/methods/{params['name']}/status.json",value); results.append(value)
                continue
            for params in configs:
                base=OUT/f"{scope}/methods/{params['name']}"
                if (base/"metrics.json").exists():
                    results.append({"name":params["name"],"status":"completed","map_at_5":s6.read_json(base/"metrics.json")["map_at_k"]}); continue
                def run(params=params):
                    assert not duplicates[params["name"]],"Declared duplicate must be reused, not replaced"
                    predicted,fallback,seconds=combine(data,S9,normalized,params["beta"])
                    runtime={"direct_fusion_and_rank_seconds":seconds,"shared_EASE_fit_score_seconds":fit_score_seconds,
                         "shared_binary_matrix_Gram_seconds":matrix_seconds,
                         "measured_component_cost_for_tie_break_seconds":fit_score_seconds+seconds,
                         "scope":"lambda model reused for four beta variants; component-attributed cost is not independently measured end-to-end runtime"}
                    return persist(scope,params,predicted,data,baseline,repeat,fallback,runtime,
                         "new_unique" if scope=="horizon12" else "repeated_backtest_fit")
                results.append(s.attempt(scope,params["name"],"new_unique" if scope=="horizon12" else "repeated_backtest_fit",run))
                s.checkpoint("evaluating",scope=scope,last_completed=params["name"])
            del normalized; gc.collect()
        names=[r["name"] for r in results if r["status"]=="completed"]
        baseline_aps=s6.read_json(OUT/f"{scope}/methods/stage9_control/per_user_ap5.json")
        users=data["cohort"]; labels=data["labels"]["primary_top5"]
        arrays=[]
        for name in names:
            aps=s6.read_json(OUT/f"{scope}/methods/{name}/per_user_ap5.json")
            arrays.append(np.array([aps[u]-baseline_aps[u] for u in users]))
        delta=np.column_stack(arrays) if arrays else np.zeros((len(users),0))
        groups={"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}
        intervals={}
        for group,mask in groups.items():
            bounds=s8.paired_bootstrap(delta[mask],s.SEED,s.RESAMPLES)
            intervals[group]={name:{"delta_MAP":float(delta[mask,i].mean()),"paired_95_interval":bounds[i],"users":int(mask.sum())} for i,name in enumerate(names)}
        save(f"{scope}/paired_bootstrap.json",{"seed":s.SEED,"resamples":s.RESAMPLES,"cohorts":intervals,
             "uncertainty":"descriptive, selection-unadjusted, overlapping historical periods and repeated selection"})
        save(f"{scope}/results.json",results); all_results[scope]=results
        s.checkpoint(f"{scope}_complete",completed=len(names))
        del X,G,data; gc.collect()
    save("results.json",all_results)


def select():
    if (OUT/"selection.json").exists(): return
    comparison=[]
    for params in s.CONFIGS:
        row={**params}; available=all((OUT/f"{scope}/methods/{params['name']}/metrics.json").exists() for scope in s.SCOPES)
        row["available_both_splits"]=available
        if available:
            costs=[]
            for scope in s.SCOPES:
                metrics=s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")
                interval=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]["all_users"][params["name"]]
                row[f"{scope}_MAP"]=metrics["map_at_k"]; row[f"{scope}_gain"]=interval["delta_MAP"]; row[f"{scope}_interval"]=interval["paired_95_interval"]
                costs.append(s6.read_json(OUT/f"{scope}/methods/{params['name']}/runtime.json")["measured_component_cost_for_tie_break_seconds"])
            row["eligible"]=row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0
            row["minimum_gain"]=min(row["horizon12_gain"],row["original10_gain"]); row["measured_cost_sum_seconds"]=sum(costs)
        else: row["eligible"]=False; row["reason"]="declared fit unavailable; no replacement"
        comparison.append(row)
    eligible=[r for r in comparison if r["eligible"]]
    chosen=None
    if eligible:
        best=max(r["minimum_gain"] for r in eligible); tied=[r for r in eligible if r["minimum_gain"]==best]
        beta=min(r["beta"] for r in tied); tied=[r for r in tied if r["beta"]==beta]
        cost=min(r["measured_cost_sum_seconds"] for r in tied); tied=[r for r in tied if r["measured_cost_sum_seconds"]==cost]
        if len(tied)==1: chosen=next(p for p in s.CONFIGS if p["name"]==tied[0]["name"])
    save("comparison.json",comparison); save("selection.json",{"selected":chosen,"comparison":comparison,
         "decision":"freeze and refit only selected candidate" if chosen else "retain Stage 9; no eligible winner",
         "selection_rule":"immutable declaration.json","selection_uses_diagnostic_subgroups":False,
         "independent_confirmation":False})
    s.checkpoint("selection_complete",selected=chosen)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("evaluate","select")); phase=parser.parse_args().phase
    resume_record()
    if phase=="evaluate": evaluate()
    elif phase=="select": select()


if __name__=="__main__":main()
