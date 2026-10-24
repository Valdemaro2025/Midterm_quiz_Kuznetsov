#!/usr/bin/env python3
"""Fresh diagnostic computations and independent saved-output checks."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("stage8_study",Path(__file__).with_name("study.py"))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
np=m.np


def read_predictions(stage):
    paths={5:"outputs/stage5/kuznetsov_vladimir_midterm_submission.csv",
           6:"outputs/stage6/kuznetsov_vladimir_midterm_submission_challenger.csv",
           7:"outputs/stage7/kuznetsov_vladimir_midterm_submission_ablation.csv"}
    path=m.ROOT/paths[stage]
    with path.open(newline="",encoding="utf-8") as handle:
        predictions={r["user_id"]:[int(x) for x in r["tv_show_id"].split()] for r in csv.DictReader(handle)}
    return path,predictions


def tie_expected_ap(predictions, counts):
    """Analytic expectation under a uniform draw from the fifth-frequency tie."""
    ranked=sorted(counts,key=lambda s:(-counts[s],s))
    if len(ranked)<=5:
        return m.independent_ap(predictions,ranked)
    threshold=counts[ranked[4]]
    fixed={s for s,v in counts.items() if v>threshold}
    boundary={s for s,v in counts.items() if v==threshold}
    take=5-len(fixed); size=len(boundary)
    p=take/size
    pair=take*(take-1)/(size*(size-1)) if size>1 else 0.
    def marginal(s): return 1. if s in fixed else p if s in boundary else 0.
    def joint(a,b):
        if a==b: return marginal(a)
        if a in fixed: return marginal(b)
        if b in fixed: return marginal(a)
        return pair if a in boundary and b in boundary else 0.
    terms=[]; seen=set()
    for i,s in enumerate(predictions[:5]):
        if s in seen:
            continue
        numerator=marginal(s)+math.fsum(joint(s,t) for t in seen)
        terms.append(numerator/(i+1))
        seen.add(s)
    return math.fsum(terms)/5


def diagnostics():
    diagnostic_path=m.OUT/"diagnostics/external_review_verification.json"
    if diagnostic_path.exists():
        old=m.s6.read_json(diagnostic_path)
        if old.get("schema_version")==2:
            return
        m.save("diagnostics/external_review_template_order_bootstrap.json",old)
    review=m.s6.read_json(m.REVIEW)
    findings={}
    method_names={"control":"hybrid_70_30","challenger":"hybrid_repeat_fraction20_h28_w85",
                  "no_decay":"hybrid_repeat_fraction20_nodecay_w85","repeat50":"repeat_50"}
    local={}
    for scope in ("primary","robustness"):
        users=m.s6.read_json(m.ROOT/f"outputs/stage6/{scope}/cohort.json")
        labels=m.s6.read_json(m.ROOT/f"outputs/stage6/{scope}/labels.json")["primary_top5"]
        per_user={}
        for key,name in method_names.items():
            stage=7 if key=="no_decay" else 6
            predictions=m.s6.read_json(m.ROOT/f"outputs/stage{stage}/{scope}/methods/{name}/top5.json")
            per_user[key]=np.array([m.independent_ap(predictions[u],labels[u]) for u in users])
        maps={k:float(v.mean()) for k,v in per_user.items()}
        map_match=all(abs(maps[k]-review["local"][scope]["maps"][k])<1e-12 for k in maps)
        comparisons={}
        for left,right,label in (("challenger","control","challenger_minus_control"),("no_decay","control","no_decay_minus_control"),
                                  ("control","repeat50","control_minus_repeat50")):
            delta=per_user[left]-per_user[right]
            template_interval=m.paired_bootstrap(delta)[0]
            # Reviewer endpoints are reproducible in lexicographic user-ID order.
            # Stage 8's declared comparisons retain the template cohort order.
            interval=m.paired_bootstrap(delta[np.argsort(np.array(users))])[0]
            record={"mean":float(delta.mean()),"paired_95_interval":interval,
                    "resampling_user_order":"lexicographic user-ID order, reproduces external endpoints",
                    "template_order_interval":template_interval,
                    "users_better":int(np.sum(delta>1e-14)),"users_worse":int(np.sum(delta<-1e-14)),
                    "users_tied":int(np.sum(abs(delta)<=1e-14))}
            external=review["local"][scope][label]
            record["matches_external_point_and_counts"]=abs(record["mean"]-external["mean"])<1e-12 and all(record[k]==external[k] for k in ("users_better","users_worse","users_tied"))
            record["matches_external_interval_same_seed"]=bool(np.allclose(interval,external["descriptive_bootstrap_95_ci"],rtol=0,atol=1e-12))
            comparisons[label]=record
        local[scope]={"maps":maps,"matches_external_maps":map_match,"comparisons":comparisons}
    findings["local_metrics_and_paired_uncertainty"]=local

    cfg=m.s6.load_config(m.ROOT/"config/stage5.json")
    variants=m.s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
    ids=m.s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
    mapping={u:i for i,u in enumerate(ids)}
    submissions={}; prediction_sets={}
    for stage in (5,6,7):
        path,predictions=read_predictions(stage)
        prediction_sets[stage]=predictions
        selected=[(mapping.get(u),s) for u,p in predictions.items() for s in p]
        personal={"slots":len(selected)}
        for variant,name in (("positive","fraction_seen_any"),("sens50","fraction_seen_at_least_50"),("primary80","fraction_seen_at_least_80")):
            personal[name]=sum(variants[variant].get(ui,{}).get(si,0)>0 for ui,si in selected)/len(selected)
        personal["median_personal_80_count"]=float(np.median([variants["primary80"].get(ui,{}).get(si,0) for ui,si in selected]))
        external=review["submissions"][f"stage{stage}"]
        submissions[f"stage{stage}"]={"sha256":m.s6.sha256_file(path),"unique_recommended_ids":len(set(s for _,s in selected)),
             "recommended_personal_history":personal,
             "matches_external":m.s6.sha256_file(path)==external["sha256"] and len(set(s for _,s in selected))==external["unique_recommended_ids"]
                and all(abs(v-external["recommended_personal_history"][k])<1e-12 for k,v in personal.items())}
    findings["submissions"]=submissions
    changes={}
    for left,right in ((6,5),(7,5),(7,6)):
        key=f"stage{left}_vs_stage{right}"
        value=m.prediction_changes(prediction_sets[left],prediction_sets[right],list(prediction_sets[5]))
        changes[key]={**value,"matches_external":all(abs(value[k]-review["comparisons"][key][k])<1e-12 for k in value)}
    findings["prediction_diversity_changes"]=changes
    findings["protocol_lengths_weeks"]={"primary":{"training":10,"prediction":10},"later":{"training":14,"prediction":6},"final":{"training":20,"prediction":12}}
    findings["protocol_lengths_match"]=findings["protocol_lengths_weeks"]==review["protocol_lengths_weeks"]
    new_individuals=m.s6.read_json(m.ROOT/"outputs/stage6/plan.json")["individual_repeat_variants"]
    scores={p["name"]:m.s6.read_json(m.ROOT/f"outputs/stage6/primary/methods/{p['name']}/metrics.json")["map_at_k"] for p in new_individuals}
    control=m.s6.read_json(m.ROOT/"outputs/stage6/primary/methods/repeat_50/metrics.json")["map_at_k"]
    findings["selection_restriction"]={"verified_against_plan_and_code": "max(results[2:]" in (m.ROOT/"outputs/stage6/code/study.py").read_text(),
         "repeat50_control_map":control,"best_new_individual_map":max(scores.values()),
         "control_was_stronger_than_new_individuals":control>max(scores.values()),
         "note":"This followed the Stage 6 instruction to choose among the eight new repeats; it left original-repeat 85/15 untested. Stage 8 A addresses that missing comparison."}

    # Independent frequency-boundary sensitivity, without changing frozen labels.
    maps=m.s6.read_json(m.ROOT/"outputs/stage3/views/id_maps.json")["val"]["user_ids"]
    index={u:i for i,u in enumerate(maps)}
    counts=defaultdict(dict)
    with (m.ROOT/"outputs/stage3/interactions/val_user_program_counts.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            value=int(row["qualifying_airings_primary"])
            if value>0: counts[int(row["user_idx"])][int(row["tv_show_id"])]=value
    users=m.s6.read_json(m.ROOT/"outputs/stage6/primary/cohort.json")
    ambiguous=0
    for u in users:
        values=sorted(counts.get(index.get(u),{}).values(),reverse=True)
        ambiguous+=int(len(values)>5 and values[4]==values[5])
    sensitivity={}
    records={row["name"]:row for row in review["label_tie_sensitivity"]["methods"]}
    for key,name in method_names.items():
        stage=7 if key=="no_decay" else 6
        predictions=m.s6.read_json(m.ROOT/f"outputs/stage{stage}/primary/methods/{name}/top5.json")
        reverse=[]; uniform=[]
        for u in users:
            c=counts.get(index.get(u),{})
            labels=sorted(c,key=lambda s:(-c[s],-s))[:5]
            reverse.append(m.independent_ap(predictions[u],labels))
            uniform.append(tie_expected_ap(predictions[u],c))
        result={"descending_id_diagnostic_map":math.fsum(reverse)/len(users),"uniform_boundary_expected_map":math.fsum(uniform)/len(users)}
        result["matches_external"]=abs(result["descending_id_diagnostic_map"]-records[name]["descending_id_label_tiebreak_map"])<1e-12 and abs(result["uniform_boundary_expected_map"]-records[name]["uniform_boundary_tie_expected_map"])<1e-12
        sensitivity[name]=result
    findings["label_tie_sensitivity"]={"ambiguous_fifth_item_users":ambiguous,"matches_external_count":ambiguous==751,
         "four_control_diagnostics":sensitivity,"note":"Diagnostic only; no frozen label replacement or model selection on alternate labels."}

    drift={}
    for name,past_path,future_path,past_weeks,future_weeks in (
        ("train_to_val",m.ROOT/"outputs/stage3/airings/airings_train.csv",m.ROOT/"outputs/stage3/airings/candidate_airings_val.csv",10,10),
        ("history_to_test",cfg.out("airings","airings_history.csv"),cfg.out("airings","candidate_airings_test.csv"),20,12)):
        past=m.cache_schedule(past_path); future=m.cache_schedule(future_path)
        p=Counter(int(x) for x in past["show"]); f=Counter(int(x) for x in future["show"])
        shared=set(p)&set(f)
        ratios=np.array([(f[s]/future_weeks)/(p[s]/past_weeks) for s in sorted(shared)])
        result={"past_ids":len(p),"future_ids":len(f),"future_ids_unseen_in_past_schedule":len(set(f)-set(p)),"shared_ids":len(shared),
             "shared_ids_rate_less_than_half":int(np.sum(ratios<.5)),"shared_ids_rate_more_than_double":int(np.sum(ratios>2)),
             "independent_rate_quantiles_10_25_50_75_90":np.quantile(ratios,[.1,.25,.5,.75,.9]).tolist()}
        external=review["schedule_drift"][name]
        result["counts_match_external"]=all(result[k]==external[k] for k in external if k!="rate_ratio_quantiles")
        result["quantile_note"]="External review does not state quantile probabilities; this session explicitly uses 10/25/50/75/90 percentiles. Quantile equivalence is not assumed."
        drift[name]=result
    findings["schedule_drift"]=drift
    findings["external_reported_kaggle_values"]={"values":review["observed_kaggle_columns_unconfirmed"],
         "attribution":"external Codex review; not local executions or new uploads",
         "stage7_values_independently_verified":False,"column_labels":None,
         "conditional_interpretation":"Reported Stage 7 exceeds Stage 6 but remains below Stage 5 on both unlabeled columns; opposite to the local no-decay effect."}
    save={"schema_version":2,"external_author":"external Codex review","external_path":str(m.REVIEW),"external_sha256":m.s6.sha256_file(m.REVIEW),
          "this_file":"Fresh OpenCode-session diagnostic computations, separately attributed from the external originals",
          "findings":findings}
    m.save("diagnostics/external_review_verification.json",save)
    m.s6.checkpoint("external_diagnostics_verified",ambiguous_fifth_item_users=ambiguous)


def verify():
    checks={}; comparison={}; recomputed={}
    before=m.s6.read_json(m.OUT/"preservation_before.json")
    checks["all_prior_artifacts_byte_preserved"]=all((m.ROOT/p).exists() and m.s6.sha256_file(m.ROOT/p)==h for p,h in before.items())
    checks["external_review_preserved"]=m.s6.sha256_file(m.REVIEW)==m.s6.read_json(m.OUT/"external_review_attribution.json")["sha256"]
    # Raw competition inputs are independently checked against the verified Stage 5 manifest.
    raw=m.s6.read_json(m.ROOT/"outputs/stage5/audit/raw_manifest_after.json")
    checks["raw_data_and_template_unchanged"]=all(m.s6.sha256_file(Path(v["path"]))==v["sha256"] for v in raw.values())
    for scope in ("horizon12","original10"):
        users=m.s6.read_json(m.OUT/f"{scope}/cohort.json")
        labels=m.s6.read_json(m.OUT/f"{scope}/labels.json")["primary_top5"]
        candidates=set(m.s6.read_json(m.OUT/f"{scope}/candidates.json"))
        checks[f"{scope}_cohort_1260"]=len(users)==len(set(users))==1260
        checks[f"{scope}_labels_in_candidates"]=all(set(v)<=candidates and len(v)<=5 for v in labels.values())
        aps={}; predictions={}
        for name in (*m.CONTROLS,*m.NEW):
            directory=m.OUT/f"{scope}/methods/{name}"
            predictions[name]=m.s6.read_json(directory/"top5.json")
            per_user={u:m.independent_ap(predictions[name][u],labels[u]) for u in users}
            saved=m.s6.read_json(directory/"per_user_ap5.json")
            metrics=m.s6.read_json(directory/"metrics.json")
            checks[f"{scope}/{name}/user_aps"]=all(abs(per_user[u]-saved[u])<1e-12 for u in users)
            group={"map_at_k":users,"map_with_positive":[u for u in users if labels[u]],
                   "map_with_at_least_k_positive":[u for u in users if len(labels[u])>=5]}
            values={k:math.fsum(per_user[u] for u in group_users)/len(group_users) if group_users else 0. for k,group_users in group.items()}
            checks[f"{scope}/{name}/independent_metrics"]=all(abs(values[k]-metrics[k])<1e-12 for k in values)
            checks[f"{scope}/{name}/valid_predictions"]=set(predictions[name])==set(users) and all(len(p)==len(set(p))==5 and set(p)<=candidates for p in predictions[name].values())
            aps[name]=np.array([per_user[u] for u in users])
            recomputed[f"{scope}/{name}"]=values
        bootstrap=m.s6.read_json(m.OUT/f"{scope}/paired_bootstrap.json")
        names=list((*m.CONTROLS,*m.NEW))
        deltas=np.column_stack([aps[name]-aps["stage5_hybrid_70_30"] for name in names])
        intervals=m.paired_bootstrap(deltas)
        checks[f"{scope}_bootstrap_reproduced"]=all(np.allclose(intervals[i],bootstrap["cohorts"]["all_users"][name]["paired_95_interval"],rtol=0,atol=1e-12) for i,name in enumerate(names))
        comparison[scope]={}
        for name in names:
            comparison[scope][name]={"map":float(aps[name].mean()),**bootstrap["cohorts"]["all_users"][name]}
        provenance=m.s6.read_json(m.OUT/f"{scope}/provenance.json")
        checks[f"{scope}_no_training_future"]=provenance["max_training_observation_airing_end"]<=provenance["cutoff"]
    horizon=m.s6.read_json(m.OUT/"horizon12/provenance.json")
    checks["horizon_exactly_8_train_12_validation_weeks"]=horizon["training_weeks"]==8 and horizon["validation_weeks"]==12
    checks["horizon_uses_both_schedule_sources"]=all(v>0 for v in horizon["target_airings_by_source"].values())
    checks["horizon_did_not_reuse_weeks21_30_only_pool"]=len(horizon["candidate_ids_only_in_weeks19_20"])>0 and horizon["candidate_count"]>6361
    checks["original_labels_preserved_exactly"]=m.s6.read_json(m.OUT/"original10/labels.json")==m.s6.read_json(m.ROOT/"outputs/stage3/labels/val_labels_frozen.json")
    records=[json.loads(line) for line in (m.OUT/"search_attempts.jsonl").read_text().splitlines()]
    unique={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    completed={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="completed"}
    checks["exactly_four_new_configurations"]=unique==completed==set(m.NEW)
    catalog=[]
    for stage in (3,4):
        for path in sorted((m.ROOT/f"outputs/stage{stage}/methods").glob("*/config.json")):
            catalog.append({"name":path.parent.name,"original_stage":stage,"parameters_path":str(path.relative_to(m.ROOT)),
                            "parameters_sha256":m.s6.sha256_file(path),"new_in_stage8":False})
    prior6=m.s6.read_json(m.ROOT/"outputs/stage6/primary/results.json")
    for result in prior6:
        if result["category"]=="new_unique":
            path=m.ROOT/f"outputs/stage6/primary/methods/{result['name']}/parameters.json"
            catalog.append({"name":result["name"],"original_stage":6,"parameters_path":str(path.relative_to(m.ROOT)),
                            "parameters_sha256":m.s6.sha256_file(path),"new_in_stage8":False})
    stage7=m.s6.read_json(m.ROOT/"outputs/stage7/plan.json")
    path=m.ROOT/f"outputs/stage7/primary/methods/{stage7['name']}/parameters.json"
    catalog.append({"name":stage7["name"],"original_stage":7,"parameters_path":str(path.relative_to(m.ROOT)),
                    "parameters_sha256":m.s6.sha256_file(path),"new_in_stage8":False})
    checks["complete_31_configuration_history_retained"]=len(catalog)==31 and len({r["name"] for r in catalog})==31
    for name in m.NEW:
        catalog.append({"name":name,"original_stage":8,"parameters_path":f"outputs/stage8/horizon12/methods/{name}/parameters.json",
                        "new_in_stage8":True,"evaluations":["horizon12","original10"]})
    m.save("search_history.json",{"unique_configurations":len(catalog),"configurations":catalog,
         "repeated_backtest_fits_are_not_new_configurations":True})
    checks["repeated_fits_not_new_configurations"]=len([r for r in records if r["category"]=="repeated_backtest_fit" and r["event"]=="completed"])==4
    checks["no_failed_attempts"]=not any(r["event"]=="failed" for r in records)
    for scope in ("horizon12","original10"):
        c=next(r["time_epoch"] for r in records if r["scope"]==scope and r["name"]==m.NEW[2] and r["event"]=="completed")
        d=next(r["time_epoch"] for r in records if r["scope"]==scope and r["name"]==m.NEW[3] and r["event"]=="started")
        checks[f"{scope}_C_evaluated_before_D"]=c<d
    best=m.s6.read_json(m.OUT/"current_best_submission.json")
    checks["stage5_best_designation_unchanged"]=best["stage"]==5 and m.s6.sha256_file(m.ROOT/best["path"])==best["sha256"]
    checks["no_stage8_submission_generated"]=not list(m.OUT.glob("*submission*.csv")) and not (m.OUT/"final").exists()
    diag=m.s6.read_json(m.OUT/"diagnostics/external_review_verification.json")["findings"]
    checks["external_maps_verified"]=all(v["matches_external_maps"] for v in diag["local_metrics_and_paired_uncertainty"].values())
    checks["external_paired_intervals_reproduced"]=all(c["matches_external_interval_same_seed"] for v in diag["local_metrics_and_paired_uncertainty"].values() for c in v["comparisons"].values())
    checks["external_submission_diagnostics_verified"]=all(v["matches_external"] for v in diag["submissions"].values())
    checks["external_tie_count_verified"]=diag["label_tie_sensitivity"]["matches_external_count"]
    checks["external_schedule_drift_counts_verified"]=all(v["counts_match_external"] for v in diag["schedule_drift"].values())
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),
             "checks_total":len(checks),"preserved_files":len(before),"new_unique_configurations":4,
             "previous_unique_configurations":31,"combined_unique_configurations":35,"independent_metrics":recomputed}
    m.save("audit/validation.json",summary)
    m.save("comparison.json",comparison)
    # Decision based on evidence magnitude and both splits, not the tiniest positive estimate.
    b=[comparison[scope][m.NEW[1]] for scope in ("horizon12","original10")]
    worthy=all(v["delta_map"]>.005 and v["paired_95_interval"][0]>0 for v in b)
    recommendation={"designated_best":"Stage 5, unchanged", "candidate_meriting_possible_later_refit":m.NEW[1] if worthy else None,
         "refit_performed":False,"submission_generated":False,"uploaded":False,
         "reason":"B has a roughly one-percentage-point gain on both splits with positive descriptive interval lower bounds; it is the strongest candidate for a separately authorized refit, not proven Kaggle improvement" if worthy else
                  "No candidate has sufficiently consistent, nontrivial evidence to justify refitting now",
         "other_candidates":"A is a small point improvement; C alone underperforms; D is compared in the saved tables but is weaker than B",
         "uncertainty":"Intervals are conditional descriptive, not selection-adjusted. Four comparisons, feedback-informed designs, overlapping splits and known local/Kaggle disagreement prevent an independent generalization claim.",
         "horizon_matching_does_not_explain_all_generalization_error":"Stage 6 and Stage 7 still beat Stage 5 in this 12-week backtest despite worse reported Kaggle columns; matching the forecast horizon does not fix the ranking mismatch. Training length is also still 8 weeks here versus 20 weeks for the final historical refit.",
         "official_label_uncertainty":"Frequency ties and fragmentation aggregation remain incompletely specified; local assumptions retained unchanged",
         "external_attribution":"External Codex computations are separately attributed; this session freshly verified relevant diagnostics"}
    m.save("recommendation.json",recommendation)
    m.save("audit/source_code.json",{str(p.relative_to(m.ROOT)):m.s6.sha256_file(p) for p in sorted((m.OUT/"code").glob("*.py"))})
    assert summary["all_passed"],summary
    m.s6.checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",best="Stage 5")
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","preserved_files":len(before),"comparison":comparison,"recommendation":recommendation},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("diagnostics","verify","all"))
    args=parser.parse_args()
    if args.phase in ("diagnostics","all"): diagnostics()
    if args.phase in ("verify","all"): verify()


if __name__=="__main__":main()
