#!/usr/bin/env python3
"""Conditional final refit and independent audits; no diagnosis/search restart."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("stage11_completed_runner",Path(__file__).with_name("run.py"))
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
s,s6,s8,s9,e=r.s,r.s6,r.s8,r.s9,r.e
ROOT,OUT,np=r.ROOT,r.OUT,r.np


def independent_ap(predictions,relevant):
    relevant=set(relevant)
    if not relevant: return 0.
    seen=set(); hits=0; total=0.
    for position,item in enumerate(predictions[:5],1):
        if item in relevant and item not in seen:
            hits+=1; total+=hits/position
        seen.add(item)
    return total/min(5,len(relevant))


def independent_rank(scores,candidates,popularity,stats):
    eligible=set(candidates)
    ordered=sorted((j for j,v in scores.items() if j in eligible and v>0),
                   key=lambda j:(-scores[j],-stats.get(j,(0,0))[0],-stats.get(j,(0,0))[1],j))
    result=ordered[:5]
    for j in popularity:
        if len(result)>=5: break
        if j in eligible and j not in result: result.append(j)
    for j in sorted(eligible):
        if len(result)>=5: break
        if j not in result: result.append(j)
    return result


def finish():
    selected=s6.read_json(OUT/"selection.json")["selected"]
    if selected is None:
        assert not s.DEST.exists()
        s.save("final_decision.json",{"refitted":False,"submission_generated":False,"selected":None,
             "retain":"Stage 9","reason":"No configuration passed the original conservative rule on both splits and the 12-week interval",
             "private_or_public_scores_used_for_selection":False})
        s.checkpoint("no_final_refit_authorized",retain="Stage 9")
        return
    if (OUT/"final/provenance.json").exists(): return
    # Same final fusion implementation reproduces selected historical lists from
    # the fitted, training-only score caches. No successful study fit is rerun.
    for scope in s.SCOPES:
        data,S9,repeat,baseline,_=r.inputs(scope)
        with np.load(OUT/f"{scope}/models/lambda{selected['lambda']}/candidate_scores.npz") as cache:
            normalized=cache["normalized"]
        predicted,_,_=r.combine(data,S9,normalized,selected["beta"])
        expected=s6.read_json(OUT/f"{scope}/methods/{selected['name']}/top5.json")
        assert predicted==expected
        s.save(f"final_reproduction/{scope}.json",{"exact_top5":True,"users":len(predicted),"model_reused_not_refitted":True})
    def refit():
        assert not s.DEST.exists()
        started=time.perf_counter()
        cfg=s6.load_config(ROOT/"config/stage5.json")
        history=s8.raw_schedule(); historical=s8.subset(history,s9.TRAIN[1])
        target,target_audit=s9.read_target_schedule(cfg.schedule_files["test"],s9.TARGET)
        candidates=sorted(set(int(j) for j in target["show"]))
        assert candidates==s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"] and len(candidates)==6632
        assert s9.identity_set(historical)==s9.identity_set(s8.cache_schedule(cfg.out("airings","airings_history.csv")))
        assert s9.identity_set(target)==s9.identity_set(s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv")))
        counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle: users=[row["user_id"] for row in csv.DictReader(handle)]
        popularity,stats,pop_rows=s6.build_popularity(counts,candidates)
        data={"cohort":users,"count_variants":counts,"user_to_train_idx":{u:i for i,u in enumerate(ids)},
              "val_candidates":candidates,"popularity_val":popularity,"popularity":stats,"cutoff":s9.TRAIN[1]}
        factors=s9.factors_for(Counter(int(j) for j in historical["show"]),Counter(int(j) for j in target["show"]),s9.TRAIN,s9.TARGET,candidates)
        repeat=s6.scores_for(users,data["user_to_train_idx"],counts["sens50"],candidates)
        adjusted=s9.adjust_repeat_scores(repeat,factors,candidates)
        ar,ar_audit=s8.ar_scores(data)
        S9={u:s8.blend(adjusted[u],ar[u],.7,.3) for u in users}
        with s.BEST.open(newline="",encoding="utf-8") as handle:
            expected={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in csv.DictReader(handle)}
        assert s6.rank_scores(S9,data)[0]==expected
        plan=r.resource_plan(data); s.save("final/memory_estimate.json",plan)
        if not plan["allowed"]: raise MemoryError("Declared full-history EASE fit exceeds fixed memory budget; no model replacement")
        X,learned_users,learned_items=e.from_training_counts(counts["primary80"])
        G=e.gram_matrix(X); B,fit_audit=e.fit_from_gram(G,selected["lambda"])
        normalized,_,score_audit=e.candidate_scores(X,B,learned_users,learned_items,users,data["user_to_train_idx"],candidates)
        predicted,fallback,fusion_seconds=r.combine(data,S9,normalized,selected["beta"])
        from stage5.submission import write_submission,validate_submission
        write_submission(s.DEST,users,predicted)
        checks=validate_submission(s.DEST,users,candidates); assert checks["all_passed"]
        s.save("final/submission_checks.json",checks); s.save("final/predictions.json",predicted)
        s.save("final/factors.json",factors); s.save("final/EASE_fit_audit.json",fit_audit); s.save("final/EASE_score_audit.json",score_audit)
        s.save("final/fallback.json",fallback); s.save("final/AR_audit.json",ar_audit); s.save("final/popularity.json",pop_rows)
        s.save("final/cold_start.json",s6.cold_start_stats(users,data["user_to_train_idx"],counts))
        s.save("final/configuration.json",{"selected":selected,"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET),
             "training_weeks":s9.weeks(s9.TRAIN),"target_weeks":s9.weeks(s9.TARGET),"EASE_schedule_adjusted":False,"S9_renormalized":False})
        s.save("final/provenance.json",{"submission":str(s.DEST.relative_to(ROOT)),"sha256":s6.sha256_file(s.DEST),"selected":selected,
             "best_scored_submission":str(s.BEST.relative_to(ROOT)),"returned_score":None,"future_viewing_used":False,
             "elapsed_refit_with_schedule_IO_seconds":time.perf_counter()-started,"direct_fusion_seconds":fusion_seconds,
             "assistance":"EASE suggested by external AI assistant Codex; implementation and automated verification in this OpenCode session; personal verification pending"})
        s.checkpoint("selected_full_refit_complete",sha256=s6.sha256_file(s.DEST))
    s.attempt("full_history",selected["name"],"conditional_refit_not_new_configuration",refit)


def verify():
    started=time.perf_counter(); checks={}
    before=s6.read_json(OUT/"preservation_before.json"); after=s.manifest()
    s.save("preservation_after.json",after)
    checks["all_earlier_files_and_raw_inputs_byte_preserved"]=before==after
    checks["stage9_best_required_SHA"]=s6.sha256_file(s.BEST)==s.BEST_SHA
    resume=s6.read_json(OUT/"resume_state.json")
    checks["declaration_diagnostics_attribution_unchanged_on_resume"]=all(s6.sha256_file(OUT/path)==digest for path,digest in resume["protected_stage11_files"].items())
    log=(OUT/"attempts.jsonl").read_text()
    checks["genuine_initial_attempt_log_preserved"]=log.startswith(resume["initial_attempt_log"])
    records=[json.loads(line) for line in log.splitlines()]
    checks["completed_diagnoses_not_repeated"]=len([x for x in records if x["category"]=="diagnostic_not_new_configuration" and x["event"]=="completed"])==2
    selection=s6.read_json(OUT/"selection.json"); comparison={x["name"]:x for x in selection["comparison"]}
    recomputed={}; summaries={}
    for scope in s.SCOPES:
        data,S9,repeat,baseline,input_provenance=r.inputs(scope)
        users=data["cohort"]; labels=data["labels"]["primary_top5"]; candidates=data["val_candidates"]
        checks[f"{scope}/immutable_users_labels_candidates"]=(users==s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json") and data["labels"]==s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json") and candidates==s6.read_json(ROOT/f"outputs/stage8/{scope}/candidates.json"))
        checks[f"{scope}/no_training_future"]=input_provenance["max_training_airing_end"]<=data["cutoff"]
        X_path=OUT/f"{scope}/models/training_X.npz"
        if not X_path.exists():
            checks[f"{scope}/resource_failure_reported"]=not s6.read_json(OUT/f"{scope}/memory_estimate.json")["allowed"]
            continue
        from scipy.sparse import load_npz
        X=load_npz(X_path); mappings=s6.read_json(OUT/f"{scope}/models/mappings.json")
        learned_users=mappings["learned_user_training_indices"]; learned_items=mappings["learned_program_ids"]
        expected_users=sorted(data["count_variants"]["primary80"])
        expected_items=sorted({j for items in data["count_variants"]["primary80"].values() for j in items})
        checks[f"{scope}/all_training_users_and_items_mapped"]=learned_users==expected_users and learned_items==expected_items
        checks[f"{scope}/binary_float64_matrix"]=X.dtype==np.float64 and bool(np.all(X.data==1)) and X.nnz==sum(len(x) for x in data["count_variants"]["primary80"].values())
        checks[f"{scope}/not_restricted_to_cohort_or_target_pool"]=mappings["noncohort_training_users"]>0 and mappings["learned_noncandidate_programs"]>0
        matrix_matches=True
        for i,u in enumerate(learned_users):
            programs={learned_items[j] for j in X.indices[X.indptr[i]:X.indptr[i+1]]}
            if programs!=set(data["count_variants"]["primary80"][u]): matrix_matches=False; break
        checks[f"{scope}/matrix_exactly_pre_cutoff_primary80"]=matrix_matches
        budget=s6.read_json(OUT/f"{scope}/memory_estimate.json")
        checks[f"{scope}/preallocation_resource_budget_obeyed"]=budget["allowed"] and budget["conservative_peak_bytes"]<=budget["fixed_fit_budget_bytes"]
        checks[f"{scope}/Gram_float64_and_no_small_integer_overflow"]=mappings["Gram_dtype"]=="float64" and mappings["Gram_max"]<=len(learned_users)
        learned_set=set(learned_items)
        missing_columns=[i for i,j in enumerate(candidates) if j not in learned_set]
        missing_rows=[i for i,u in enumerate(users) if data["user_to_train_idx"].get(u) not in set(learned_users)]
        normalized_by_reg={}
        for reg in s.LAMBDAS:
            directory=OUT/f"{scope}/models/lambda{reg}"
            if not (directory/"candidate_scores.npz").exists(): continue
            fit=s6.read_json(directory/"fit_audit.json")
            checks[f"{scope}/lambda{reg}/zero_diagonal_column_division"]=fit["zero_diagonal"] and fit["column_division_sample_max_error"]<1e-12
            checks[f"{scope}/lambda{reg}/solver_residual_float64"]=fit["dtype"]=="float64" and fit["inverse_residual_sample_max_abs"]<1e-9
            checks[f"{scope}/lambda{reg}/negative_coefficients_preserved"]=fit["negative_coefficients"]>0 and fit["negative_coefficients_preserved"] and fit["coefficient_min"]<0
            with np.load(directory/"candidate_scores.npz") as archive: normalized=archive["normalized"]; raw=archive["raw"]
            positive=np.maximum(raw,0.); maximum=positive.max(axis=1,keepdims=True)
            expected=np.divide(positive,maximum,out=np.zeros_like(positive),where=maximum>0)
            checks[f"{scope}/lambda{reg}/completed_score_clip_candidate_only_normalization"]=normalized.dtype==raw.dtype==np.float64 and np.array_equal(normalized,expected) and normalized.shape==(1260,len(candidates))
            checks[f"{scope}/lambda{reg}/missing_items_users_zero"]=bool(np.all(raw[:,missing_columns]==0)) and bool(np.all(raw[missing_rows,:]==0))
            normalized_by_reg[reg]=normalized
            del raw,positive,expected
        names=[{"name":"stage9_control","beta":0},*s.CONFIGS]
        aps_by_name={}; delta_arrays=[]; available=[]
        for params in names:
            name=params["name"]; directory=OUT/f"{scope}/methods/{name}"
            if not (directory/"metrics.json").exists(): continue
            predictions=s6.read_json(directory/"top5.json"); saved_aps=s6.read_json(directory/"per_user_ap5.json")
            metrics=s6.read_json(directory/"metrics.json")
            aps={u:independent_ap(predictions[u],labels[u]) for u in users}
            checks[f"{scope}/{name}/independent_saved_prediction_AP"]=all(abs(aps[u]-saved_aps[u])<1e-12 for u in users)
            groups={"map_at_k":users,"map_with_positive":[u for u in users if labels[u]],"map_with_at_least_k_positive":[u for u in users if len(labels[u])>=5]}
            fresh={k:math.fsum(aps[u] for u in group)/len(group) if group else 0. for k,group in groups.items()}
            checks[f"{scope}/{name}/independent_three_cohort_metrics"]=all(abs(fresh[k]-metrics[k])<1e-12 for k in fresh)
            checks[f"{scope}/{name}/prediction_format_candidates"]=set(predictions)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predictions.values())
            # Recompute mixture and ranking from saved component scores, not by
            # refitting EASE or rerunning any successfully completed experiment.
            beta=params["beta"]; normalized=normalized_by_reg.get(params.get("lambda"))
            candidate_index={j:i for i,j in enumerate(candidates)}
            exact=True
            for i,u in enumerate(users):
                scores={j:(1-beta)*v for j,v in S9[u].items() if v>0} if beta<1 else {}
                if beta>0:
                    for j,value in zip(candidates,normalized[i]):
                        if value>0: scores[j]=scores.get(j,0)+beta*float(value)
                if independent_rank(scores,candidates,data["popularity_val"],data["popularity"])!=predictions[u]: exact=False; break
            checks[f"{scope}/{name}/independent_fusion_without_S9_renormalization"]=exact
            subgroup=s6.read_json(directory/"subgroups.json")
            base_metric=s6.read_json(OUT/f"{scope}/methods/stage9_control/metrics.json")["map_at_k"]
            pair_total=sum(len(labels[u]) for u in users)
            checks[f"{scope}/{name}/diagnostic_partitions_account_for_pairs_and_MAP"]=all(
                 sum(x["pairs"] for x in partition.values())==pair_total and
                 abs(sum(x["AP_contribution_delta_per_all_users"] for x in partition.values())-(fresh["map_at_k"]-base_metric))<1e-12
                 for partition in subgroup["pair_partitions"].values())
            checks[f"{scope}/{name}/history_size_subgroups_partition_users"]=sum(x["users"] for x in subgroup["user_history_groups"].values())==1260
            recomputed[f"{scope}/{name}"]=fresh; aps_by_name[name]=np.array([aps[u] for u in users])
        checks[f"{scope}/beta0_exact_stage9_top5"]=s6.read_json(OUT/f"{scope}/methods/stage9_control/top5.json")==baseline
        completed=[p["name"] for p in s.CONFIGS if p["name"] in aps_by_name]
        delta=np.column_stack([aps_by_name[name]-aps_by_name["stage9_control"] for name in completed])
        saved_intervals=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]
        masks={"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}
        for group,mask in masks.items():
            bounds=s8.paired_bootstrap(delta[mask],s.SEED,s.RESAMPLES)
            checks[f"{scope}/{group}/paired_bootstrap_reproduced"]=all(np.allclose(bounds[i],saved_intervals[group][name]["paired_95_interval"],rtol=0,atol=1e-12) for i,name in enumerate(completed))
        for i,name in enumerate(completed):
            checks[f"{scope}/{name}/selection_gain_matches_independent_AP"]=abs(float(delta[:,i].mean())-comparison[name][f"{scope}_gain"])<1e-12
        summaries[scope]={"model_dimensions":mappings,"memory":budget}
        del normalized_by_reg,X,data; r.gc.collect()
    declaration=s6.read_json(OUT/"declaration.json")
    checks["only_original_eight_configurations_evaluated"]=declaration["configurations"]==s.CONFIGS and set(comparison)=={p["name"] for p in s.CONFIGS}
    for row in comparison.values():
        expected=row.get("available_both_splits",False) and row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0
        checks[f"selection/{row['name']}/frozen_rule_exact"]=row["eligible"]==expected
    eligible=[x for x in comparison.values() if x["eligible"]]
    if selection["selected"] is None:
        checks["no_eligible_winner_no_refit_no_CSV"]=not eligible and not s.DEST.exists() and not list(OUT.glob("*.csv")) and not (OUT/"final").exists()
    else:
        chosen=comparison[selection["selected"]["name"]]
        checks["selected_maximin_rule"]=chosen["eligible"] and chosen["minimum_gain"]==max(x["minimum_gain"] for x in eligible)
        provenance=s6.read_json(OUT/"final/provenance.json")
        checks["conditional_CSV_SHA"]=s6.sha256_file(s.DEST)==provenance["sha256"]
        checks["conditional_CSV_format"]=s6.read_json(OUT/"final/submission_checks.json")["all_passed"]
    new={x["name"] for x in records if x["category"]=="new_unique" and x["event"]=="started"}
    checks["configuration_count_retries_count_once"]=new=={p["name"] for p in s.CONFIGS} and 41+len(new)<=49
    checks["no_failed_actual_model_attempts"]=not any(x["event"]=="failed" for x in records)
    checks["exactly_four_shared_historical_EASE_fits"]=len([x for x in records if x["category"]=="shared_model_fit_not_new_configuration" and x["event"]=="completed"])==4
    first_fit=min(x["time_epoch"] for x in records if x["category"]=="shared_model_fit_not_new_configuration")
    checks["declaration_existed_before_first_model_fit"]=(OUT/"declaration.json").stat().st_mtime<first_fit
    checks["stage9_still_designated_best_scored"]=s6.read_json(OUT/"current_best_submission.json")["stage"]==9
    checks["external_assistance_and_personal_verification_accurately_attributed"]=(s6.read_json(OUT/"source_attribution.json")["personal_verification_status"]=="pending; no personal verification claimed")
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
             "preserved_workspace_files":len(before),"new_unique_configurations":len(new),"total_unique_configurations":41+len(new),
             "independent_metrics":recomputed,"verification_seconds":time.perf_counter()-started,
             "verification_type":"automated OpenCode-session checks; not personal user verification"}
    s.save("audit/validation.json",summary)
    s.save("search_history.json",{"previous_unique":41,"new_unique":len(new),"total_unique":41+len(new),
         "prior_history":"outputs/stage10/search_history.json","prior_history_sha256":s6.sha256_file(ROOT/"outputs/stage10/search_history.json"),
         "new_configurations":s.CONFIGS,"interrupted_retry_rule":"unique configuration names counted once; diagnostic/model-fit/reproduction attempts not added"})
    sources={p.resolve() for p in (OUT/"code").glob("*.py")}
    for stage,file in ((6,"study.py"),(8,"study.py"),(9,"refit.py"),(10,"study.py")): sources.add(ROOT/f"outputs/stage{stage}/code/{file}")
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": sources.add(path)
    s.save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(sources)})
    assert summary["all_passed"],{k:v for k,v in checks.items() if not v}
    s.checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",selected=selection["selected"])
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","preserved_files":len(before),"selection":selection["selected"],
                      "decision":selection["decision"],"total_unique_configurations":41+len(new)},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("final","verify","all")); phase=parser.parse_args().phase
    if phase in ("final","all"): finish()
    if phase in ("verify","all"): verify()


if __name__=="__main__":main()
