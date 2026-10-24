#!/usr/bin/env python3
"""Independent Stage 12 count/matrix/metric/selection/submission audits."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("stage12_audited_study",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
s6,s8,s9,r11,np,e=s.s6,s.s8,s.s9,s.r11,s.np,s.e
ROOT,OUT=s.ROOT,s.OUT


def AP(predicted,relevant):
    relevant=set(relevant)
    if not relevant: return 0.
    seen=set(); hits=0; terms=[]
    for rank,item in enumerate(predicted[:5],1):
        if item in relevant and item not in seen: hits+=1; terms.append(hits/rank)
        seen.add(item)
    return math.fsum(terms)/min(5,len(relevant))


def scalar_reference(counts):
    values=[int(n) for items in counts.values() for n in items.values() if n>0]
    log_squared=math.fsum(math.log1p(n)**2 for n in values)
    scalar=math.sqrt(len(values)/log_squared) if values else 0.
    return scalar,values


def verify():
    started=time.perf_counter(); checks={}; recomputed={}
    before=s6.read_json(OUT/"preservation_before.json"); after=s.manifest()
    s.save("preservation_after.json",after)
    checks["all_prior_workspace_files_and_raw_inputs_byte_preserved"]=before==after
    checks["stage9_required_SHA_preserved"]=s6.sha256_file(s.BEST)==s.BEST_SHA
    selection=s6.read_json(OUT/"selection.json"); rows={r["name"]:r for r in selection["comparison"]}
    for scope in s.SCOPES:
        data,S9,repeat,baseline,controls,provenance=s.load_inputs(scope)
        users=data["cohort"]; labels=data["labels"]["primary_top5"]; candidates=data["val_candidates"]
        checks[f"{scope}/unchanged_labels_cohorts_candidates"]=(data["labels"]==s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json") and
              users==s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json") and candidates==s6.read_json(ROOT/f"outputs/stage8/{scope}/candidates.json"))
        count_audit=s6.read_json(OUT/f"{scope}/training_count_audit.json")
        checks[f"{scope}/qualifying_pre_cutoff_airings_and_unique_interactions"]=count_audit["equals_verified_training_count_cache"] and count_audit["max_qualifying_airing_end"]<=data["cutoff"] and count_audit["distinct_user_airings"]==count_audit["qualifying_airings_sum"]
        from scipy.sparse import load_npz
        X=load_npz(OUT/f"{scope}/models/training_X.npz")
        mapping=s6.read_json(OUT/f"{scope}/models/mappings.json"); stats=s6.read_json(OUT/f"{scope}/models/input_statistics.json")
        old_mapping=s6.read_json(ROOT/f"outputs/stage11/{scope}/models/mappings.json")
        learned_users=mapping["learned_user_training_indices"]; learned_items=mapping["learned_program_ids"]
        checks[f"{scope}/training_axes_unchanged_from_binary"]=learned_users==old_mapping["learned_user_training_indices"] and learned_items==old_mapping["learned_program_ids"] and X.nnz==old_mapping["binary_nnz"]
        scalar,values=scalar_reference(data["count_variants"]["primary80"])
        checks[f"{scope}/independent_training_only_scalar"]=abs(scalar-stats["normalization_scalar"])<1e-13
        checks[f"{scope}/weighted_float64_Frobenius_norm_equals_nnz"]=X.dtype==np.float64 and np.isclose(float(X.multiply(X).sum()),X.nnz,rtol=1e-12,atol=1e-12)
        exact=True
        for i,user in enumerate(learned_users):
            expected=data["count_variants"]["primary80"][user]
            start,end=X.indptr[i:i+2]
            if set(learned_items[j] for j in X.indices[start:end])!=set(expected): exact=False; break
            for column,value in zip(X.indices[start:end],X.data[start:end]):
                if abs(value-scalar*math.log1p(expected[learned_items[column]]))>1e-13: exact=False; break
        checks[f"{scope}/every_weight_matches_training_log_count"]=exact
        checks[f"{scope}/counts_and_frequency_distribution_match"]=stats["total_qualifying_airings"]==sum(values) and {int(k):v for k,v in stats["count_histogram"].items()}==dict(s.Counter(values))
        memory=s6.read_json(OUT/f"{scope}/memory_estimate.json")
        checks[f"{scope}/resource_budget_checked_before_allocation"]=memory["allowed"] and memory["conservative_peak_bytes"]<=memory["fixed_fit_budget_bytes"]
        normalized_by_reg={}
        missing_columns=[i for i,j in enumerate(candidates) if j not in set(learned_items)]
        missing_rows=[i for i,u in enumerate(users) if data["user_to_train_idx"].get(u) not in set(learned_users)]
        for reg in (100,1000):
            fit=s6.read_json(OUT/f"{scope}/models/lambda{reg}/fit_audit.json")
            checks[f"{scope}/lambda{reg}/same_float64_solver_zero_diagonal_column_division"]=(fit["dtype"]=="float64" and fit["zero_diagonal"] and
                 fit["column_division_sample_max_error"]<1e-12 and fit["inverse_residual_sample_max_abs"]<1e-9)
            checks[f"{scope}/lambda{reg}/negative_coefficients_retained"]=fit["negative_coefficients"]>0 and fit["negative_coefficients_preserved"]
            with np.load(OUT/f"{scope}/models/lambda{reg}/scores.npz") as archive: raw=archive["raw"]; normalized=archive["normalized"]
            positive=np.maximum(raw,0.); maximum=positive.max(axis=1,keepdims=True)
            expected=np.divide(positive,maximum,out=np.zeros_like(positive),where=maximum>0)
            checks[f"{scope}/lambda{reg}/completed_score_clip_then_candidate_max"]=np.array_equal(normalized,expected) and raw.dtype==normalized.dtype==np.float64
            checks[f"{scope}/lambda{reg}/missing_history_items_zero"]=bool(np.all(raw[:,missing_columns]==0)) and bool(np.all(raw[missing_rows,:]==0))
            normalized_by_reg[reg]=normalized
            del raw,positive,expected
        all_aps={}; binary_AP={}
        baseline_aps={u:AP(baseline[u],labels[u]) for u in users}
        for name,predictions in controls.items():
            fresh={u:AP(predictions[u],labels[u]) for u in users}
            stored=s6.read_json(OUT/f"{scope}/controls/{name}/per_user_ap5.json")
            source=s6.read_json(OUT/f"{scope}/controls/{name}/provenance.json")
            checks[f"{scope}/control_{name}/reused_immutable_predictions_and_independent_AP"]=s6.sha256_file(ROOT/source["source"])==source["sha256"] and all(abs(fresh[u]-stored[u])<1e-12 for u in users)
            if name!="stage9": binary_AP[name]=fresh
        for params in s.CONFIGS:
            name=params["name"]; directory=OUT/f"{scope}/methods/{name}"
            predictions=s6.read_json(directory/"top5.json"); metrics=s6.read_json(directory/"metrics.json"); stored=s6.read_json(directory/"per_user_ap5.json")
            fresh={u:AP(predictions[u],labels[u]) for u in users}; all_aps[name]=fresh
            checks[f"{scope}/{name}/independent_AP_from_saved_predictions"]=all(abs(fresh[u]-stored[u])<1e-12 for u in users)
            groups={"map_at_k":users,"map_with_positive":[u for u in users if labels[u]],"map_with_at_least_k_positive":[u for u in users if len(labels[u])>=5]}
            values_by_group={k:math.fsum(fresh[u] for u in group)/len(group) if group else 0. for k,group in groups.items()}
            checks[f"{scope}/{name}/independent_three_cohort_metrics"]=all(abs(metrics[k]-values_by_group[k])<1e-12 for k in groups)
            checks[f"{scope}/{name}/five_unique_eligible_predictions"]=set(predictions)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predictions.values())
            matched=s6.read_json(directory/"matched_binary_comparison.json")
            binary_mean=math.fsum(binary_AP[name].values())/len(users)
            checks[f"{scope}/{name}/matched_binary_difference"]=abs((values_by_group["map_at_k"]-binary_mean)-matched["count_minus_binary_metrics"]["map_at_k"])<1e-12
            checks[f"{scope}/{name}/selection_gain_independent"]=abs((values_by_group["map_at_k"]-math.fsum(baseline_aps.values())/len(users))-rows[name][f"{scope}_gain"])<1e-12
            normalized=normalized_by_reg[params["lambda"]]
            reconstructed=r11.combine(data,S9,normalized,params["beta"])[0]
            checks[f"{scope}/{name}/same_fusion_pipeline_no_S9_renormalization"]=reconstructed==predictions
            for reference in ("stage9","binary"):
                subgroup=s6.read_json(directory/f"subgroups_vs_{reference}.json")
                reference_mean=math.fsum(baseline_aps.values())/len(users) if reference=="stage9" else binary_mean
                delta=values_by_group["map_at_k"]-reference_mean
                checks[f"{scope}/{name}/repeat50_pair_diagnostic_additivity_vs_{reference}"]=all(abs(sum(x["AP_contribution_delta_per_all_users"] for x in partition.values())-delta)<1e-12 for partition in subgroup["pair_partitions"].values())
            recomputed[f"{scope}/{name}"]=values_by_group
        delta=np.column_stack([np.array([all_aps[p["name"]][u]-baseline_aps[u] for u in users]) for p in s.CONFIGS])
        matched_delta=np.column_stack([np.array([all_aps[p["name"]][u]-binary_AP[p["name"]][u] for u in users]) for p in s.CONFIGS])
        masks={"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}
        for reference,matrix,key in (("stage9",delta,"delta_MAP"),("binary",matched_delta,"count_minus_binary_MAP")):
            saved=s6.read_json(OUT/f"{scope}/paired_bootstrap_vs_{reference}.json")["cohorts"]
            for group,mask in masks.items():
                bounds=s8.paired_bootstrap(matrix[mask],s.SEED,s.RESAMPLES)
                checks[f"{scope}/{reference}/{group}/paired_intervals_reproduced"]=all(np.allclose(bounds[i],saved[group][p["name"]]["paired_95_interval"],rtol=0,atol=1e-12) and abs(float(matrix[mask,i].mean())-saved[group][p["name"]][key])<1e-12 for i,p in enumerate(s.CONFIGS))
        del X,normalized_by_reg,data; s.gc.collect()
    declared=s6.read_json(OUT/"declaration.json")
    checks["only_four_declared_configurations"]=declared["configurations"]==s.CONFIGS and len(rows)==4
    for row in rows.values():
        checks[f"selection/{row['name']}/preserved_rule"]=row["eligible"]==(row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0)
    selected=selection["selected"]
    if selected:
        chosen=rows[selected["name"]]; eligible=[x for x in rows.values() if x["eligible"]]
        checks["selected_maximum_minimum_gain"]=chosen["eligible"] and chosen["minimum_gain"]==max(x["minimum_gain"] for x in eligible)
        frozen=s6.read_json(OUT/"selected_configuration.json")
        checks["selected_configuration_and_declaration_frozen"]=frozen["parameters"]==selected and frozen["declaration_sha256"]==s6.sha256_file(OUT/"declaration.json") and frozen["selection_sha256"]==s6.sha256_file(OUT/"selection.json")
        for scope in s.SCOPES:
            checks[f"{scope}/final_implementation_historical_equivalence"]=s6.read_json(OUT/f"final_reproduction/{scope}/checks.json")["all_passed"]
            checks[f"{scope}/selected_lists_still_exact"]=s6.read_json(OUT/f"final_reproduction/{scope}/top5.json")==s6.read_json(OUT/f"{scope}/methods/{selected['name']}/top5.json")
        cfg=s6.load_config(ROOT/"config/stage5.json")
        full=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))["primary80"]
        scalar,values=scalar_reference(full); final_stats=s6.read_json(OUT/"final/input_statistics.json")
        checks["final_scalar_independently_recomputed_from_full_training"]=abs(scalar-final_stats["normalization_scalar"])<1e-13 and final_stats["nnz"]==len(values)
        checks["final_weighted_norm_equal_nnz"]=np.isclose(final_stats["X_squared_Frobenius_norm"],final_stats["nnz"],rtol=1e-12,atol=1e-12)
        count_audit=s6.read_json(OUT/"final/training_count_audit.json")
        checks["final_distinct_qualifying_counts_temporally_valid"]=count_audit["equals_verified_full_history_counts"] and count_audit["max_qualifying_airing_end"]<=s9.TRAIN[1]
        final_cfg=s6.read_json(OUT/"final/configuration.json")
        checks["full_20_week_training_12_week_target_and_fresh_factors"]=final_cfg["training_weeks"]==20 and final_cfg["target_weeks"]==12 and final_cfg["scalar_recomputed_from_full_training"] and final_cfg["S9_factors_recomputed"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); template_headers=reader.fieldnames; users=[row["user_id"] for row in reader]
        from stage5.submission import validate_submission
        checked=validate_submission(s.DEST,users,s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"])
        for check in checked["checks"]: checks[f"submission/{check['check']}"]=check["ok"]
        with s.DEST.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); headers=reader.fieldnames; predicted={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in reader}
        provenance=s6.read_json(OUT/"final/provenance.json")
        checks["submission_matches_template_headers_and_saved_predictions"]=headers==template_headers and predicted==s6.read_json(OUT/"final/predictions.json")
        checks["submission_SHA_matches_provenance"]=s6.sha256_file(s.DEST)==provenance["sha256"]
        checks["at_most_one_new_CSV_at_required_path"]=list(OUT.glob("*.csv"))==[s.DEST]
        checks["Stage9_still_best_scored_new_score_unknown"]=provenance["best_scored_submission"]==str(s.BEST.relative_to(ROOT)) and provenance["returned_score"] is None
    else:
        checks["no_qualifying_candidate_no_CSV"]=not any(x["eligible"] for x in rows.values()) and not s.DEST.exists()
    records=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    new={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    checks["unique_budget_four_new_maximum_53"]=new=={p["name"] for p in s.CONFIGS} and 49+len(new)<=53
    checks["four_shared_historical_model_fits"]=len([r for r in records if r["category"]=="shared_fit_not_new_configuration" and r["event"]=="completed"])==4
    checks["no_failed_actual_model_attempts"]=not any(r["event"]=="failed" for r in records)
    checks["declaration_precedes_first_fit"]=(OUT/"declaration.json").stat().st_mtime<min(r["time_epoch"] for r in records)
    checks["Stage9_designated_best_preserved_in_new_metadata"]=s6.read_json(OUT/"current_best_submission.json")["stage"]==9
    s.save("search_history.json",{"previous_unique":49,"new_unique":len(new),"total_unique":49+len(new),"new_configurations":s.CONFIGS,
          "prior_history":"outputs/stage11/search_history.json","prior_history_sha256":s6.sha256_file(ROOT/"outputs/stage11/search_history.json"),
          "refits_reproductions_and_reused_controls_not_new_configurations":True})
    sources={p.resolve() for p in (OUT/"code").glob("*.py")}
    for stage,file in ((6,"study.py"),(8,"study.py"),(9,"refit.py"),(10,"study.py"),(11,"study.py"),(11,"run.py"),(11,"ease_model.py")):
        sources.add(ROOT/f"outputs/stage{stage}/code/{file}")
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": sources.add(path)
    s.save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(sources)})
    checks={key:bool(value) for key,value in checks.items()}
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
             "preserved_workspace_files":len(before),"new_unique":len(new),"total_unique":49+len(new),"independent_metrics":recomputed,
             "selected":selected,"verification_seconds":time.perf_counter()-started,"verification":"automated in OpenCode; not personal user verification"}
    s.save("audit/validation.json",summary)
    assert summary["all_passed"],{k:v for k,v in checks.items() if not v}
    s.checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",selected=selected)
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","preserved_files":len(before),"selected":selected,
          "submission":s6.read_json(OUT/"final/provenance.json") if selected else None},indent=2))


if __name__=="__main__":verify()
