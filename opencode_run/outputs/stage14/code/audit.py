#!/usr/bin/env python3
"""Conditional finish and independent preference-AR/statistic/metric audits."""
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
spec=importlib.util.spec_from_file_location("stage14_saved_study",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
s6,s8,s9,r11,np=s.s6,s.s8,s.s9,s.r11,s.np
ROOT,OUT=s.ROOT,s.OUT


def independent_AP(predictions,relevant):
    relevant=set(relevant)
    if not relevant: return 0.
    hits=0; seen=set(); terms=[]
    for rank,item in enumerate(predictions[:5],1):
        if item in relevant and item not in seen: hits+=1; terms.append(hits/rank)
        seen.add(item)
    return math.fsum(terms)/min(5,len(relevant))


def independent_rank(scores,candidates,popularity,stats):
    allowed=set(candidates)
    eligible=sorted((j for j,v in scores.items() if j in allowed and v>0),key=lambda j:(-scores[j],-stats.get(j,(0,0))[0],-stats.get(j,(0,0))[1],j))
    result=eligible[:5]
    for j in [*popularity,*sorted(allowed)]:
        if len(result)>=5: break
        if j in allowed and j not in result: result.append(j)
    return result


def finish():
    selection=s6.read_json(OUT/"selection.json"); selected=selection["selected"]
    if selected is None:
        assert not s.DEST.exists()
        s.save("final_decision.json",{"refitted":False,"submission_generated":False,"retain":"Stage 13",
             "reason":"No hybrid passed the frozen conservative rule; standalone results are diagnostic only"})
        s.checkpoint("no_final_refit_authorized",retain="Stage 13")
        return
    assert selected["role"]=="hybrid"
    if not (OUT/"selected_configuration.json").exists():
        s.save("selected_configuration.json",{"parameters":selected,"declaration_sha256":s6.sha256_file(OUT/"declaration.json"),
             "selection_sha256":s6.sha256_file(OUT/"selection.json"),"new_unique_from_refit":0})
    for scope in s.SCOPES:
        if (OUT/f"final_reproduction/{scope}.json").exists(): continue
        data,repeat,R,original,original_norm,E,baseline,factors,provenance=s.load_inputs(scope)
        normalized=r11.int_scores(OUT/f"{scope}/AR_models/{selected['variant']}/normalized_scores.json")
        predicted=s.rank_hybrid(data,R,normalized,E)[0]
        assert predicted==s6.read_json(OUT/f"{scope}/methods/{selected['name']}/top5.json")
        s.save(f"final_reproduction/{scope}.json",{"exact_top5":True,"same_final_prediction_implementation":True})
    if (OUT/"final/provenance.json").exists(): return
    def refit():
        assert not s.DEST.exists(); started=time.perf_counter()
        cfg=s6.load_config(ROOT/"config/stage5.json")
        history=s8.raw_schedule(); hist=s8.subset(history,s9.TRAIN[1]); target,_=s9.read_target_schedule(cfg.schedule_files["test"],s9.TARGET)
        candidates=sorted(set(int(j) for j in target["show"]))
        assert candidates==s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"] and len(candidates)==6632
        assert s9.identity_set(hist)==s9.identity_set(s8.cache_schedule(cfg.out("airings","airings_history.csv")))
        assert s9.identity_set(target)==s9.identity_set(s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv")))
        counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle: users=[r["user_id"] for r in csv.DictReader(handle)]
        popularity,stats,pop_rows=s6.build_popularity(counts,candidates)
        data={"cohort":users,"count_variants":counts,"user_to_train_idx":{u:i for i,u in enumerate(ids)},"val_candidates":candidates,
              "popularity_val":popularity,"popularity":stats,"cutoff":s9.TRAIN[1]}
        factors=s9.factors_for(Counter(int(j) for j in hist["show"]),Counter(int(j) for j in target["show"]),s9.TRAIN,s9.TARGET,candidates)
        repeat=s6.scores_for(users,data["user_to_train_idx"],counts["sens50"],candidates)
        R=s.positive_normalize(s9.adjust_repeat_scores(repeat,factors,candidates),candidates)
        original,original_details=s8.ar_scores(data); original_norm=s.positive_normalize(original,candidates)
        module=s.p13.p12.count_module(); plan=r11.resource_plan(data)
        if not plan["allowed"]: raise MemoryError("Fixed full-history components exceed unchanged resource budget")
        X,learned_users,learned_items,input_stats=module.transformed_matrix(counts["primary80"])
        G=module.gram_matrix(X); B,fit_audit=s.e.fit_from_gram(G,1000)
        _,raw,score_audit=s.e.candidate_scores(X,B,learned_users,learned_items,users,data["user_to_train_idx"],candidates)
        E=s.p13.component(raw,candidates,factors,True)
        with s.BEST.open(newline="",encoding="utf-8") as handle: prior={r["user_id"]:[int(j) for j in r["tv_show_id"].split()] for r in csv.DictReader(handle)}
        assert s.rank_hybrid(data,R,original_norm,E)[0]==prior
        if selected["variant"]=="A":
            replacement=s.positive_normalize(s.opportunity_adjust(original,factors,candidates),candidates); details=original_details
        else:
            preferred=s.ar_module().fit(counts["primary80"],selected["K"],candidates)
            scores,details=s.ar_module().user_scores(data,preferred)
            s.ar_module().save_model(OUT/"final/preference_model",preferred)
            replacement=s.positive_normalize(scores,candidates)
        predicted,fallback,seconds=s.rank_hybrid(data,R,replacement,E)
        from stage5.submission import write_submission,validate_submission
        write_submission(s.DEST,users,predicted); audit=validate_submission(s.DEST,users,candidates); assert audit["all_passed"]
        s.save("final/submission_checks.json",audit); s.save("final/predictions.json",predicted); s.save("final/rule_details.json",details)
        s.save("final/input_statistics.json",input_stats); s.save("final/EASE_fit_audit.json",fit_audit); s.save("final/EASE_score_audit.json",score_audit)
        s.save("final/opportunity_factors.json",factors); s.save("final/fallback.json",fallback); s.save("final/popularity.json",pop_rows)
        s.save("final/configuration.json",{"selected":selected,"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET),
             "training_weeks":s9.weeks(s9.TRAIN),"target_weeks":s9.weeks(s9.TARGET),"EASE_input_scalar":input_stats["normalization_scalar"]})
        s.save("final/provenance.json",{"submission":str(s.DEST.relative_to(ROOT)),"sha256":s6.sha256_file(s.DEST),"selected":selected,
             "best_scored_submission":str(s.BEST.relative_to(ROOT)),"returned_score":None,"future_viewing_used":False,
             "refit_including_IO_seconds":time.perf_counter()-started,"hybrid_prediction_seconds":seconds})
        s.checkpoint("selected_full_refit_complete",sha256=s6.sha256_file(s.DEST))
    s.attempt("full_history",selected["name"],"conditional_refit_not_new_configuration",refit)


def verify_model(scope,variant,data,checks):
    from scipy.sparse import load_npz
    directory=OUT/f"{scope}/AR_models/{variant}"
    model_stats=s6.read_json(directory/"statistics.json")
    A=load_npz(directory/"A.npz"); Y=load_npz(directory/"Y.npz"); C=load_npz(directory/"C.npz")
    users=model_stats["users"]; items=model_stats["items"]; columns={j:i for i,j in enumerate(items)}; K=model_stats["K"]
    source=data["count_variants"]["primary80"]
    checks[f"{scope}/{variant}/all_training_axes_and_original_AR_population"]=(users==sorted(source) and items==sorted({j for rows in source.values() for j in rows}) and len(users)==s6.read_json(ROOT/f"outputs/stage8/{scope}/association_details.json")["training_users"])
    checks[f"{scope}/{variant}/float64_fractional_asymmetric_statistics"]=(A.dtype==Y.dtype==C.dtype==np.float64 and (C-C.T).nnz>0 and np.any(C.data!=np.floor(C.data)))
    memberships_valid=True; binary_valid=True; row_mass=0.
    for i,u in enumerate(users):
        counts=source[u]; positives=list(counts.values())
        if len(positives)<=K: expected={j:1. for j in counts}
        else:
            boundary=float(np.partition(np.array(positives,dtype=np.float64),len(positives)-K)[len(positives)-K])
            above=sum(n>boundary for n in positives); tied=sum(n==boundary for n in positives)
            expected={j:1. if n>boundary else (K-above)/tied for j,n in counts.items() if n>=boundary}
        a=A.getrow(i); y=Y.getrow(i)
        if set(items[j] for j in a.indices)!=set(counts) or not np.all(a.data==1): binary_valid=False
        actual={items[j]:float(v) for j,v in zip(y.indices,y.data)}
        if set(actual)!=set(expected) or any(abs(actual[j]-expected[j])>1e-12 for j in actual): memberships_valid=False
        row_mass+=min(K,len(counts))
        if abs(float(y.sum())-min(K,len(counts)))>1e-12: memberships_valid=False
    checks[f"{scope}/{variant}/binary_any_primary80_A_exact"]=binary_valid
    checks[f"{scope}/{variant}/independent_full_catalogue_fractional_memberships"]=memberships_valid
    checks[f"{scope}/{variant}/total_membership_mass"]=abs(float(Y.sum())-row_mass)<1e-8
    support_A=np.asarray(A.sum(axis=0)).ravel(); support_Y=np.asarray(Y.sum(axis=0)).ravel()
    checks[f"{scope}/{variant}/correct_antecedent_and_consequent_marginals"]=np.allclose(support_A,model_stats["support_A"],rtol=0,atol=1e-12) and np.allclose(support_Y,model_stats["support_Y"],rtol=0,atol=1e-12) and np.allclose(C.diagonal(),support_Y,rtol=1e-12,atol=1e-12)
    # Selected C columns are separately summed over users with A_ui=1.
    sample_columns=sorted(set([0,len(items)//2,len(items)-1]))
    expected_columns=np.asarray(A.T.dot(Y[:,sample_columns].toarray()))
    checks[f"{scope}/{variant}/independent_C_column_reference"]=np.allclose(C[:,sample_columns].toarray(),expected_columns,rtol=1e-12,atol=1e-12)
    saved=s6.read_json(directory/"rules.json"); candidates=set(data["val_candidates"])
    valid=True; complete=True; total=0
    for i,program in enumerate(items):
        source_row=C.getrow(i); expected=[]
        if support_A[i]>0:
            for j,mass in zip(source_row.indices,source_row.data):
                if i==j or items[j] not in candidates or mass<5 or support_Y[j]<=0: continue
                confidence=float(mass)/float(support_A[i]); lift=confidence/(float(support_Y[j])/len(users))
                if confidence>=.005 and lift>=1: expected.append((int(j),confidence,float(mass),lift))
        expected.sort(key=lambda x:(-x[1],x[0])); expected=expected[:50]
        actual=saved.get(str(program),[]); total+=len(actual)
        if [row["program"] for row in actual]!=[items[row[0]] for row in expected]: complete=False
        for row in actual:
            j=columns[row["program"]]; mass=float(C[i,j]); confidence=mass/support_A[i] if support_A[i]>0 else 0
            lift=confidence/(support_Y[j]/len(users)) if support_Y[j]>0 else 0
            if i==j or row["program"] not in candidates or mass<5 or confidence<.005 or lift<1 or abs(row["mass"]-mass)>1e-12 or abs(row["confidence"]-confidence)>1e-12 or abs(row["lift"]-lift)>1e-12: valid=False
        if len(actual)>50: valid=False
    checks[f"{scope}/{variant}/retained_rule_mass_confidence_correct_Y_lift"]=valid
    checks[f"{scope}/{variant}/complete_pruned_rules_and_tie_order"]=complete and total==model_stats["details"]["total_rules"]
    # Independently aggregate max-confidence rules from the original top-50
    # training-count ordered personal antecedents, including noncandidate history.
    raw_scores=s6.read_json(directory/"raw_scores.json"); aggregate_valid=True
    for u in data["cohort"]:
        counts=source.get(data["user_to_train_idx"].get(u),{})
        antecedents=sorted(counts,key=lambda j:(-counts[j],columns[j]))[:50]
        expected={}
        for i_program in antecedents:
            for rule in saved.get(str(i_program),[]): expected[rule["program"]]=max(expected.get(rule["program"],0),rule["confidence"])
        actual={int(k):v for k,v in raw_scores[u].items()}
        if actual!=expected: aggregate_valid=False; break
    checks[f"{scope}/{variant}/independent_antecedents_and_max_aggregation"]=aggregate_valid


def verify():
    started=time.perf_counter(); checks={}; recomputed={}
    before=s6.read_json(OUT/"preservation_before.json"); after=s.manifest(); s.save("preservation_after.json",after)
    checks["all_earlier_files_and_raw_inputs_preserved_byte_for_byte"]=before==after
    checks["Stage13_best_required_SHA_preserved"]=s6.sha256_file(s.BEST)==s.BEST_SHA
    selection=s6.read_json(OUT/"selection.json"); compared={row["name"]:row for row in selection["comparison"]}
    for scope in s.SCOPES:
        data,repeat,R,original,original_norm,E,baseline,factors,provenance=s.load_inputs(scope)
        users=data["cohort"]; candidates=data["val_candidates"]; labels=data["labels"]["primary_top5"]
        checks[f"{scope}/immutable_labels_users_and_candidate_pool"]=(data["labels"]==s6.read_json(ROOT/f"outputs/stage8/{scope}/labels.json") and users==s6.read_json(ROOT/f"outputs/stage8/{scope}/cohort.json") and candidates==s6.read_json(ROOT/f"outputs/stage8/{scope}/candidates.json"))
        checks[f"{scope}/all_learning_uses_pre_cutoff_counts"]=provenance["max_training_airing_end"]<=data["cutoff"]
        for variant in ("B","C"): verify_model(scope,variant,data,checks)
        savedA=r11.int_scores(OUT/f"{scope}/AR_models/A/raw_scores.json")
        expectedA={u:{j:v*factors[j] for j,v in rows.items() if j in candidates and v>0} for u,rows in original.items()}
        checks[f"{scope}/A_original_completed_scores_adjusted_before_normalization"]=savedA==expectedA
        ar_components={"control":original_norm}
        for variant in ("A","B","C"):
            raw=r11.int_scores(OUT/f"{scope}/AR_models/{variant}/raw_scores.json")
            stored=r11.int_scores(OUT/f"{scope}/AR_models/{variant}/normalized_scores.json")
            expected={u:{j:v/max(rows.values()) for j,v in rows.items() if v>0} if rows else {} for u,rows in raw.items()}
            checks[f"{scope}/{variant}/independent_candidate_max_normalization"]=stored==expected
            ar_components[variant]=stored
        aps={}
        control_params={"name":"stage13_control","role":"control","variant":"control"}
        for params in (control_params,*s.CONFIGS):
            name=params["name"]; directory=OUT/f"{scope}/methods/{name}"
            predictions=s6.read_json(directory/"top5.json"); recorded=s6.read_json(directory/"per_user_ap5.json"); metrics=s6.read_json(directory/"metrics.json")
            values={u:independent_AP(predictions[u],labels[u]) for u in users}; aps[name]=np.array([values[u] for u in users])
            checks[f"{scope}/{name}/independent_AP_from_saved_predictions"]=all(abs(values[u]-recorded[u])<1e-12 for u in users)
            groups={"map_at_k":users,"map_with_positive":[u for u in users if labels[u]],"map_with_at_least_k_positive":[u for u in users if len(labels[u])>=5]}
            fresh={k:math.fsum(values[u] for u in group)/len(group) if group else 0. for k,group in groups.items()}
            checks[f"{scope}/{name}/independent_three_cohort_metrics"]=all(abs(fresh[k]-metrics[k])<1e-12 for k in fresh)
            checks[f"{scope}/{name}/valid_predictions"]=set(predictions)==set(users) and all(len(p)==len(set(p))==5 and set(p)<=set(candidates) for p in predictions.values())
            independent=True; replacement=ar_components[params["variant"]]
            for i,u in enumerate(users):
                if params["role"]=="standalone": scores=replacement[u]
                else:
                    intermediate={j:.7*v for j,v in R[u].items()}
                    for j,v in replacement[u].items(): intermediate[j]=intermediate.get(j,0)+.3*v
                    scores={j:.9*v for j,v in intermediate.items()}
                    for j,v in zip(candidates,E[i]):
                        if v>0: scores[j]=scores.get(j,0)+.1*float(v)
                if independent_rank(scores,candidates,data["popularity_val"],data["popularity"])!=predictions[u]: independent=False; break
            checks[f"{scope}/{name}/unchanged_repeat_EASE_exact_hybrid_arithmetic"]=independent
            subgroup=s6.read_json(directory/"subgroups.json")
            baseline_metric=s6.read_json(OUT/f"{scope}/methods/stage13_control/metrics.json")["map_at_k"]
            checks[f"{scope}/{name}/familiar_unobserved_diagnostic_AP_accounting"]=all(abs(sum(r["AP_contribution_delta_per_all_users"] for r in partition.values())-(fresh["map_at_k"]-baseline_metric))<1e-12 for partition in subgroup["pair_partitions"].values())
            recomputed[f"{scope}/{name}"]=fresh
        checks[f"{scope}/exact_Stage13_control_reproduction"]=s6.read_json(OUT/f"{scope}/methods/stage13_control/top5.json")==baseline
        original_metrics=s6.read_json(ROOT/f"outputs/stage13/{scope}/methods/E_EASE_opportunity/metrics.json")
        checks[f"{scope}/control_MAP_verified_without_rounded_values"]=all(abs(recomputed[f"{scope}/stage13_control"][k]-original_metrics[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive"))
        delta=np.column_stack([aps[p["name"]]-aps["stage13_control"] for p in s.CONFIGS]); saved=s6.read_json(OUT/f"{scope}/paired_bootstrap.json")["cohorts"]
        for group,mask in {"all_users":np.ones(len(users),dtype=bool),"positive":np.array([bool(labels[u]) for u in users]),"at_least_5":np.array([len(labels[u])>=5 for u in users])}.items():
            bounds=s8.paired_bootstrap(delta[mask],s.SEED,s.RESAMPLES)
            checks[f"{scope}/{group}/paired_intervals_reproduced"]=all(np.allclose(bounds[i],saved[group][p["name"]]["paired_95_interval"],rtol=0,atol=1e-12) for i,p in enumerate(s.CONFIGS))
        for i,p in enumerate(s.CONFIGS):
            checks[f"{scope}/{p['name']}/selection_gain_independent"]=abs(float(delta[:,i].mean())-compared[p["name"]][f"{scope}_gain"])<1e-12
        del data,E; s.gc.collect()
    for row in compared.values():
        checks[f"selection/{row['name']}/hybrid_only_predeclared_screen"]=row["eligible"]==(row["role"]=="hybrid" and row["horizon12_gain"]>=.002 and row["original10_gain"]>=.002 and row["horizon12_interval"][0]>0)
    if selection["selected"] is None:
        checks["no_qualifying_hybrid_no_refit_no_CSV"]=not any(x["eligible"] for x in compared.values()) and not s.DEST.exists() and not list(OUT.glob("*.csv")) and not (OUT/"final").exists()
    else:
        provenance=s6.read_json(OUT/"final/provenance.json")
        checks["selected_hybrid_not_standalone"]=selection["selected"]["role"]=="hybrid"
        checks["conditional_final_CSV_SHA_and_checks"]=s6.sha256_file(s.DEST)==provenance["sha256"] and s6.read_json(OUT/"final/submission_checks.json")["all_passed"]
    declaration=s6.read_json(OUT/"declaration.json")
    checks["only_six_authorized_recommendation_configurations"]=declaration["new_recommendation_configurations"]==s.CONFIGS and len(compared)==6
    records=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    new={r["name"] for r in records if r["category"]=="new_unique" and r["event"]=="started"}
    checks["standalone_included_unique_count_max64"]=new=={p["name"] for p in s.CONFIGS} and 58+len(new)<=64
    checks["no_failed_actual_attempts"]=not any(r["event"]=="failed" for r in records)
    checks["declaration_before_first_evaluation"]=(OUT/"declaration.json").stat().st_mtime<min(r["time_epoch"] for r in records)
    checks["best_scored_designation_Stage13"]=s6.read_json(OUT/"current_best_submission.json")["stage"]==13
    checks={key:bool(value) for key,value in checks.items()}
    s.save("search_history.json",{"previous_unique":58,"new_unique":len(new),"total_unique":58+len(new),"new_configurations":s.CONFIGS,
         "prior_history":"outputs/stage13/search_history.json","prior_history_sha256":s6.sha256_file(ROOT/"outputs/stage13/search_history.json"),
         "shared_AR_fits_and_repeated_backtests_not_new_configurations":True})
    sources={p.resolve() for p in (OUT/"code").glob("*.py")}
    for stage,file in ((3,"association.py"),(4,"association.py")):
        sources.add(ROOT/f"src/stage{stage}/{file}")
    for stage,file in ((6,"study.py"),(8,"study.py"),(9,"refit.py"),(10,"study.py"),(11,"study.py"),(11,"run.py"),(11,"ease_model.py"),(12,"study.py"),(12,"count_input.py"),(13,"study.py")):
        sources.add(ROOT/f"outputs/stage{stage}/code/{file}")
    for module in sys.modules.values():
        filename=getattr(module,"__file__",None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix==".py": sources.add(path)
    s.save("code_provenance.json",{str(p.relative_to(ROOT)):s6.sha256_file(p) for p in sorted(sources)})
    summary={"all_passed":all(checks.values()),"checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),
             "preserved_workspace_files":len(before),"new_unique":len(new),"total_unique":58+len(new),"selected":selection["selected"],
             "independent_metrics":recomputed,"verification_seconds":time.perf_counter()-started,"verification":"automated within this OpenCode session, not personal user verification"}
    s.save("audit/validation.json",summary)
    assert summary["all_passed"],{k:v for k,v in checks.items() if not v}
    s.checkpoint("verified_complete",checks=f"{sum(checks.values())}/{len(checks)}",selected=selection["selected"])
    print(json.dumps({"checks":f"{sum(checks.values())}/{len(checks)}","preserved_files":len(before),"selected":selection["selected"],
          "decision":selection["decision"],"total_unique":58+len(new)},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("final","verify","all")); phase=parser.parse_args().phase
    if phase in ("final","all"): finish()
    if phase in ("verify","all"): verify()


if __name__=="__main__":main()
