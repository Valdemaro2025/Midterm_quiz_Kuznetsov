#!/usr/bin/env python3
"""Independent Stage 13 metric, fusion, opportunity and preservation checks."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("stage13_audited_study",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
s6,s8,s9,r11,np=s.s6,s.s8,s.s9,s.r11,s.np
ROOT,OUT=s.ROOT,s.OUT


def independent_AP(predictions,relevant):
    relevant=set(relevant)
    if not relevant: return 0.
    seen=set(); hits=0; terms=[]
    for rank,item in enumerate(predictions[:5],1):
        if item in relevant and item not in seen: hits+=1; terms.append(hits/rank)
        seen.add(item)
    return math.fsum(terms)/min(5,len(relevant))


def independent_rank(scores,candidates,popularity,stats):
    allowed=set(candidates)
    ranked=sorted((j for j,v in scores.items() if j in allowed and v>0),key=lambda j:(-scores[j],-stats.get(j,(0,0))[0],-stats.get(j,(0,0))[1],j))
    chosen=ranked[:5]
    for j in popularity:
        if len(chosen)>=5: break
        if j in allowed and j not in chosen: chosen.append(j)
    for j in sorted(allowed):
        if len(chosen)>=5: break
        if j not in chosen: chosen.append(j)
    return chosen


def verify():
    started=time.perf_counter(); checks={}; recomputed={}
    before=s6.read_json(OUT/"preservation_before.json"); after=s.manifest()
    s.save("preservation_after.json",after)
    checks["all_prior_files_and_raw_inputs_byte_preserved"]=before==after
    checks["Stage12_best_required_SHA_preserved"]=s6.sha256_file(s.BEST)==s.BEST_SHA
    selection=s6.read_json(OUT/"selection.json"); compared={row["name"]:row for row in selection["comparison"]}
    for scope in s.SCOPES:
        data,S9,repeat,baseline,raw,provenance=s.load_inputs(scope)
        users=data["cohort"]; labels=data["labels"]["primary_top5"]; candidates=data["val_candidates"]
        checks[f"{scope}/unchanged_labels_users_candidates"]=(data["labels"]==s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json") and users==s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json") and candidates==s6.read_json(ROOT/f"outputs/stage8/{scope}/candidates.json"))
        checks[f"{scope}/training_observations_pre_cutoff"]=provenance["max_training_airing_end"]<=data["cutoff"]
        opportunities=s6.read_json(OUT/f"{scope}/opportunities.json")
        past={int(k):v for k,v in opportunities["past"].items()}; target={int(k):v for k,v in opportunities["target"].items()}
        factors={int(k):v for k,v in opportunities["factors"].items()}
        checks[f"{scope}/target_pool_opportunities_complete"]=sorted(target)==candidates and sum(target.values())==opportunities["target_airings"]
        train_seconds=opportunities["training_window"][1]-opportunities["training_window"][0]
        target_seconds=opportunities["target_window"][1]-opportunities["target_window"][0]
        checks[f"{scope}/timestamp_window_lengths"]=opportunities["training_weeks"]==train_seconds/604800 and opportunities["target_weeks"]==target_seconds/604800 and opportunities["training_weeks"]==(8 if scope=="horizon12" else 10) and opportunities["target_weeks"]==(12 if scope=="horizon12" else 10)
        independent_factors={j:math.sqrt(min(4.,max(.25,target[j]*train_seconds/(past[j]*target_seconds)))) for j in candidates if past.get(j,0)>0}
        checks[f"{scope}/opportunity_formula_independently_recomputed"]=set(factors)==set(independent_factors) and all(abs(factors[j]-independent_factors[j])<1e-14 for j in factors)
        reference={int(k):v for k,v in s6.read_json(ROOT/f"outputs/stage9/reproduction/{scope}/factors.json").items()}
        checks[f"{scope}/exact_established_opportunity_factors"]=factors==reference
        positive=np.maximum(raw,0.)
        active=np.any(positive>0,axis=0)
        missing=[j for i,j in enumerate(candidates) if active[i] and (past.get(j,0)<=0 or j not in factors)]
        checks[f"{scope}/no_positive_EASE_evidence_without_historical_opportunities"]=not missing
        max0=positive.max(axis=1,keepdims=True)
        normal0=np.divide(positive,max0,out=np.zeros_like(positive),where=max0>0)
        multiplied=positive*np.array([factors.get(j,0.) for j in candidates])[None,:]
        maxE=multiplied.max(axis=1,keepdims=True)
        normalE=np.divide(multiplied,maxE,out=np.zeros_like(multiplied),where=maxE>0)
        checks[f"{scope}/E_zero_evidence_remains_zero"]=bool(np.all(normalE[raw<=0]==0))
        old_model=ROOT/f"outputs/stage12/{scope}/models/lambda1000"
        fit=s6.read_json(old_model/"fit_audit.json")
        checks[f"{scope}/negative_float64_coefficients_preserved_in_reused_fit"]=fit["dtype"]=="float64" and fit["negative_coefficients_preserved"] and fit["negative_coefficients"]>0 and fit["zero_diagonal"]
        with np.load(old_model/"scores.npz") as archive: prior_normalized=archive["normalized"]
        checks[f"{scope}/A_to_D_unadjusted_EASE_component_exact"]=np.array_equal(normal0,prior_normalized)
        aps={}
        for params in (s.CONTROL,*s.VARIANTS):
            name=params["name"]; directory=OUT/f"{scope}/methods/{name}"
            predicted=s6.read_json(directory/"top5.json"); saved=s6.read_json(directory/"per_user_ap5.json"); metrics=s6.read_json(directory/"metrics.json")
            fresh={u:independent_AP(predicted[u],labels[u]) for u in users}; aps[name]=np.array([fresh[u] for u in users])
            checks[f"{scope}/{name}/independent_AP_from_saved_predictions"]=all(abs(fresh[u]-saved[u])<1e-12 for u in users)
            groups={"map_at_k":users,"map_with_positive":[u for u in users if labels[u]],"map_with_at_least_k_positive":[u for u in users if len(labels[u])>=5]}
            values={k:math.fsum(fresh[u] for u in group)/len(group) if group else 0. for k,group in groups.items()}
            checks[f"{scope}/{name}/independent_three_cohort_metrics"]=all(abs(values[k]-metrics[k])<1e-12 for k in values)
            checks[f"{scope}/{name}/valid_five_eligible_predictions"]=set(predicted)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predicted.values())
            component=normalE if params["EASE_opportunity_adjustment"] else normal0
            exact=True
            for i,u in enumerate(users):
                fused={j:(1-params["beta"])*v for j,v in S9[u].items() if v>0}
                for j,value in zip(candidates,component[i]):
                    if value>0: fused[j]=fused.get(j,0)+params["beta"]*float(value)
                if independent_rank(fused,candidates,data["popularity_val"],data["popularity"])!=predicted[u]: exact=False; break
            checks[f"{scope}/{name}/independent_fusion_order_and_no_S9_renormalization"]=exact
            subgroup=s6.read_json(directory/"subgroups.json")
            baseline_metrics=s6.read_json(OUT/f"{scope}/methods/stage12_control/metrics.json")
            delta=values["map_at_k"]-baseline_metrics["map_at_k"]
            checks[f"{scope}/{name}/diagnostic_pair_partition_accounting"]=all(sum(x["pairs"] for x in partition.values())==sum(len(labels[u]) for u in users) and abs(sum(x["AP_contribution_delta_per_all_users"] for x in partition.values())-delta)<1e-12 for partition in subgroup["pair_partitions"].values())
            checks[f"{scope}/{name}/history_size_partition"]=sum(x["users"] for x in subgroup["user_history_groups"].values())==1260
            recomputed[f"{scope}/{name}"]=values
        checks[f"{scope}/control_lists_exact_Stage12"]=s6.read_json(OUT/f"{scope}/methods/stage12_control/top5.json")==baseline
        previous_metrics=s6.read_json(ROOT/f"outputs/stage12/{scope}/methods/logcount_ease_l1000_b10/metrics.json")
        checks[f"{scope}/control_MAP_exact_not_rounded"]=all(abs(recomputed[f"{scope}/stage12_control"][k]-previous_metrics[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive"))
        deltas=np.column_stack([aps[p["name"]]-aps["stage12_control"] for p in s.VARIANTS])
        recorded=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]
        masks={"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}
        for group,mask in masks.items():
            bounds=s8.paired_bootstrap(deltas[mask],s.SEED,s.RESAMPLES)
            checks[f"{scope}/{group}/paired_intervals_exact"]=all(np.allclose(bounds[i],recorded[group][p["name"]]["paired_95_interval"],rtol=0,atol=1e-12) for i,p in enumerate(s.VARIANTS))
        for i,p in enumerate(s.VARIANTS):
            checks[f"{scope}/{p['name']}/selection_gain_independent"]=abs(float(deltas[:,i].mean())-compared[p["name"]][f"{scope}_gain"])<1e-12
        del data,raw,positive,normal0,normalE,multiplied; s.gc.collect()
    declaration=s6.read_json(OUT/"declaration.json")
    checks["exactly_five_authorized_variants_no_combined_adjustment_betas"]=declaration["five_variants"]==s.VARIANTS and len(s.VARIANTS)==5 and sum(p["EASE_opportunity_adjustment"] for p in s.VARIANTS)==1 and s.VARIANTS[-1]["beta"]==.1
    for row in compared.values():
        checks[f"selection/{row['name']}/predeclared_rule"]=row["eligible"]==(row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0)
    selected=selection["selected"]
    if selected:
        eligible=[r for r in compared.values() if r["eligible"]]; chosen=compared[selected["name"]]
        checks["selected_maximin_gain_and_eligible"]=chosen["eligible"] and chosen["minimum_gain"]==max(r["minimum_gain"] for r in eligible)
        frozen=s6.read_json(OUT/"selected_configuration.json")
        checks["selected_configuration_frozen"]=frozen["parameters"]==selected and frozen["declaration_sha256"]==s6.sha256_file(OUT/"declaration.json") and frozen["selection_sha256"]==s6.sha256_file(OUT/"selection.json")
        for scope in s.SCOPES:
            checks[f"{scope}/final_prediction_implementation_equivalence"]=s6.read_json(OUT/f"final_reproduction/{scope}/checks.json")["all_passed"]
        cfg=s6.load_config(ROOT/"config/stage5.json")
        final_stats=s6.read_json(OUT/"final/input_statistics.json")
        counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))["primary80"]
        log_square=math.fsum(math.log1p(n)**2 for items in counts.values() for n in items.values())
        nnz=sum(len(items) for items in counts.values())
        scalar=math.sqrt(nnz/log_square) if nnz else 0.
        checks["full_training_scalar_fresh_and_Frobenius_norm_valid"]=abs(scalar-final_stats["normalization_scalar"])<1e-13 and np.isclose(final_stats["X_squared_Frobenius_norm"],nnz,rtol=1e-12,atol=1e-12)
        full=s6.read_json(OUT/"final/opportunities.json")
        checks["full_20_week_training_12_week_target_opportunities"]=full["training_weeks"]==20 and full["target_weeks"]==12 and full["recomputed_for_full_history"]
        checks["full_Stage12_and_S9_control_exact"]=all(s6.read_json(OUT/"final/control_reproduction.json").values())
        count_audit=s6.read_json(OUT/"final/training_count_audit.json")
        checks["full_pre_cutoff_qualifying_counts"]=count_audit["matches_verified_full_history_counts"] and count_audit["max_qualifying_airing_end"]<=s9.TRAIN[1]
        final_fit=s6.read_json(OUT/"final/fit_audit.json")
        checks["full_EASE_float64_zero_diagonal_negative_coefficients"]=final_fit["dtype"]=="float64" and final_fit["zero_diagonal"] and final_fit["negative_coefficients"]>0 and final_fit["negative_coefficients_preserved"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); headers=reader.fieldnames; users=[row["user_id"] for row in reader]
        from stage5.submission import validate_submission
        submission=validate_submission(s.DEST,users,s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"])
        for check in submission["checks"]: checks[f"submission/{check['check']}"]=check["ok"]
        with s.DEST.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); new_headers=reader.fieldnames; predictions={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in reader}
        provenance=s6.read_json(OUT/"final/provenance.json")
        checks["submission_exact_template_headers_and_saved_predictions"]=new_headers==headers and predictions==s6.read_json(OUT/"final/predictions.json")
        checks["submission_SHA_and_required_path"]=s6.sha256_file(s.DEST)==provenance["sha256"] and list(OUT.glob("*.csv"))==[s.DEST]
        checks["Stage12_remains_best_scored_new_score_unknown"]=provenance["best_scored_submission"]==str(s.BEST.relative_to(ROOT)) and provenance["returned_score"] is None
    else:
        checks["no_eligible_candidate_no_CSV"]=not any(row["eligible"] for row in compared.values()) and not s.DEST.exists()
    records=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    new={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    checks["configuration_budget_unique_count_maximum_58"]=len(new)<=5 and 53+len(new)<=58 and new=={p["name"] for p in s.VARIANTS}
    checks["no_failed_model_or_variant_attempts"]=not any(r["event"]=="failed" for r in records)
    checks["declaration_precedes_first_evaluation"]=(OUT/"declaration.json").stat().st_mtime<min(r["time_epoch"] for r in records)
    checks["historical_EASE_fits_reused_no_new_training"]=not any(r["scope"] in s.SCOPES and "fit" in r["category"] and r["category"]!="selected_final_implementation_reproduction_not_new" for r in records)
    checks["new_metadata_best_designation_and_score_observation"]=s6.read_json(OUT/"current_best_submission.json")["stage"]==12 and s6.read_json(OUT/"score_evidence.json")["stage12"]["public"]==.44041 and s6.read_json(OUT/"score_evidence.json")["stage12"]["private"]==.42267
    checks={key:bool(value) for key,value in checks.items()}
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
             "preserved_workspace_files":len(before),"new_unique":len(new),"total_unique":53+len(new),"selected":selected,
             "independent_metrics":recomputed,"verification_seconds":time.perf_counter()-started,"verification":"automated in this OpenCode session; personal verification not claimed"}
    s.save("audit/validation.json",summary)
    s.save("search_history.json",{"previous_unique":53,"new_unique":len(new),"total_unique":53+len(new),"new_configurations":s.VARIANTS,
         "prior_history":"outputs/stage12/search_history.json","prior_history_sha256":s6.sha256_file(ROOT/"outputs/stage12/search_history.json"),
         "refits_reproductions_and_component_reuse_not_new_configurations":True})
    sources={p.resolve() for p in (OUT/"code").glob("*.py")}
    for stage,file in ((6,"study.py"),(8,"study.py"),(9,"refit.py"),(10,"study.py"),(11,"study.py"),(11,"run.py"),(11,"ease_model.py"),(12,"study.py"),(12,"count_input.py")):
        sources.add(ROOT/f"outputs/stage{stage}/code/{file}")
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": sources.add(path)
    s.save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(sources)})
    assert summary["all_passed"],{k:v for k,v in checks.items() if not v}
    s.checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",selected=selected)
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","preserved_files":len(before),"selected":selected,
         "submission":s6.read_json(OUT/"final/provenance.json") if selected else None},indent=2))


if __name__=="__main__":verify()
