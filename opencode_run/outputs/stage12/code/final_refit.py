#!/usr/bin/env python3
"""Final implementation shared by selected historical reproductions and refit."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from collections import Counter
from pathlib import Path

spec=importlib.util.spec_from_file_location("stage12_saved_study",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
s6,s8,s9,r11,np,e=s.s6,s.s8,s.s9,s.r11,s.np,s.e
ROOT,OUT=s.ROOT,s.OUT


def fit_predict(data,S9,params):
    started=time.perf_counter(); module=s.count_module()
    plan=r11.resource_plan(data)
    if not plan["allowed"]: raise MemoryError("Selected weighted fit exceeds unchanged memory budget; no replacement")
    matrix_start=time.perf_counter()
    X,learned_users,learned_items,statistics=module.transformed_matrix(data["count_variants"]["primary80"])
    G=module.gram_matrix(X); matrix_seconds=time.perf_counter()-matrix_start
    if X.nnz:
        B,fit_audit=e.fit_from_gram(G,params["lambda"])
        normalized,raw,score_audit=e.candidate_scores(X,B,learned_users,learned_items,data["cohort"],data["user_to_train_idx"],data["val_candidates"])
    else:
        normalized=np.zeros((len(data["cohort"]),len(data["val_candidates"])),dtype=np.float64); raw=normalized.copy()
        fit_audit={"degenerate":True,"no_solve":True}; score_audit={"all_zero":True}
    predicted,fallback,fusion_seconds=r11.combine(data,S9,normalized,params["beta"])
    runtime={"matrix_transform_and_Gram_seconds":matrix_seconds,"direct_fusion_rank_seconds":fusion_seconds,
             "fit_audit_seconds":fit_audit.get("direct_Cholesky_solve_coefficient_audit_seconds",0.),
             "weighted_XB_filter_normalize_seconds":score_audit.get("direct_XB_filter_clip_normalize_seconds",0.),
             "direct_fit_predict_loaded_training_counts_seconds":time.perf_counter()-started,
             "scope":"weighted input, Gram, EASE fit/query and fusion with already provided Stage 9 scores; cache/schedule I/O excluded"}
    return predicted,{"input_statistics":statistics,"memory_estimate":plan,"fit_audit":fit_audit,"score_audit":score_audit,
                      "fallback":fallback,"runtime":runtime,"learned_user_training_indices":learned_users,"learned_program_ids":learned_items}


def execute():
    selection=s6.read_json(OUT/"selection.json"); params=selection["selected"]
    if params is None:
        assert not s.DEST.exists()
        s.save("final_decision.json",{"refitted":False,"submission_generated":False,"retain":"Stage 9","reason":"no eligible candidate"})
        return
    if not (OUT/"selected_configuration.json").exists():
        s.save("selected_configuration.json",{"parameters":params,"input":"global-training-scaled log1p >=80% qualifying-airing counts",
             "declaration_sha256":s6.sha256_file(OUT/"declaration.json"),"selection_sha256":s6.sha256_file(OUT/"selection.json"),
             "selection_rule":"maximum minimum gain, then lower beta, then lower measured cost","unique_configurations":53,
             "new_configurations_from_refit":0})
    for scope in s.SCOPES:
        if (OUT/f"final_reproduction/{scope}/checks.json").exists():
            assert s6.read_json(OUT/f"final_reproduction/{scope}/checks.json")["all_passed"]; continue
        def reproduce(scope=scope):
            started=time.perf_counter(); data,S9,repeat,baseline,controls,provenance=s.load_inputs(scope)
            predicted,audit=fit_predict(data,S9,params)
            saved=s6.read_json(OUT/f"{scope}/methods/{params['name']}/top5.json")
            original_stats=s6.read_json(OUT/f"{scope}/models/input_statistics.json")
            checks={"top5_exact":predicted==saved,"training_scalar_exact":audit["input_statistics"]["normalization_scalar"]==original_stats["normalization_scalar"]}
            metrics,aps=s9.independent_metrics(predicted,data["labels"]["primary_top5"],data["cohort"])
            old_metrics=s6.read_json(OUT/f"{scope}/methods/{params['name']}/metrics.json")
            checks["independent_metrics_exact"]=all(abs(metrics[k]-old_metrics[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive"))
            s.save(f"final_reproduction/{scope}/top5.json",predicted)
            s.save(f"final_reproduction/{scope}/metrics.json",metrics); s.save(f"final_reproduction/{scope}/per_user_ap5.json",aps)
            s.save(f"final_reproduction/{scope}/audit.json",{**audit,"total_reproduction_seconds":time.perf_counter()-started})
            s.save(f"final_reproduction/{scope}/checks.json",{"all_passed":all(checks.values()),"checks":checks})
            assert all(checks.values()),checks
            s.checkpoint("selected_historical_equivalence",scope=scope)
        s.attempt(scope,params["name"],"final_implementation_reproduction_not_new_configuration",reproduce)
    if (OUT/"final/provenance.json").exists():
        assert s.DEST.exists(); return
    def refit():
        started=time.perf_counter(); assert not s.DEST.exists(),"Refuse to overwrite a submission"
        cfg=s6.load_config(ROOT/"config/stage5.json")
        history=s8.raw_schedule(); historical=s8.subset(history,s9.TRAIN[1])
        target,target_audit=s9.read_target_schedule(cfg.schedule_files["test"],s9.TARGET)
        candidates=sorted(set(int(j) for j in target["show"]))
        assert len(candidates)==6632 and candidates==s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"]
        assert s9.identity_set(historical)==s9.identity_set(s8.cache_schedule(cfg.out("airings","airings_history.csv")))
        assert s9.identity_set(target)==s9.identity_set(s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv")))
        counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        table,_=s6.load_table(cfg.out("interactions","history_user_airing.npz"),cfg.out("airings","airings_history.csv"))
        native_counts,count_audit=s.count_module().count_distinct_airings(table,s9.TRAIN[0],s9.TRAIN[1])
        assert native_counts==counts["primary80"]
        s.save("final/training_count_audit.json",{**count_audit,"equals_verified_full_history_counts":True})
        del table,native_counts
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); template_headers=reader.fieldnames; users=[row["user_id"] for row in reader]
        popularity,stats,pop_rows=s6.build_popularity(counts,candidates)
        data={"cohort":users,"count_variants":counts,"user_to_train_idx":{u:i for i,u in enumerate(ids)},
              "val_candidates":candidates,"popularity_val":popularity,"popularity":stats,"cutoff":s9.TRAIN[1]}
        factors=s9.factors_for(Counter(int(j) for j in historical["show"]),Counter(int(j) for j in target["show"]),s9.TRAIN,s9.TARGET,candidates)
        repeat=s6.scores_for(users,data["user_to_train_idx"],counts["sens50"],candidates)
        adjusted=s9.adjust_repeat_scores(repeat,factors,candidates)
        ar,ar_audit=s8.ar_scores(data)
        S9={u:s8.blend(adjusted[u],ar[u],.7,.3) for u in users}
        with s.BEST.open(newline="",encoding="utf-8") as handle:
            baseline={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in csv.DictReader(handle)}
        assert s6.rank_scores(S9,data)[0]==baseline,"Full-history S9 component must reproduce the best CSV exactly"
        predicted,audit=fit_predict(data,S9,params)
        from stage5.submission import write_submission,validate_submission
        write_submission(s.DEST,users,predicted)
        checks=validate_submission(s.DEST,users,candidates); assert checks["all_passed"]
        with s.DEST.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); assert reader.fieldnames==template_headers
        s.save("final/submission_checks.json",checks); s.save("final/predictions.json",predicted)
        for key,value in audit.items(): s.save(f"final/{key}.json",value)
        s.save("final/stage9_factors.json",factors); s.save("final/AR_audit.json",ar_audit); s.save("final/popularity.json",pop_rows)
        s.save("final/cold_start.json",s6.cold_start_stats(users,data["user_to_train_idx"],counts))
        s.save("final/target_schedule_audit.json",target_audit)
        s.save("final/configuration.json",{"selected":params,"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET),
             "training_weeks":s9.weeks(s9.TRAIN),"target_weeks":s9.weeks(s9.TARGET),"normalization_scalar":audit["input_statistics"]["normalization_scalar"],
             "scalar_recomputed_from_full_training":True,"S9_factors_recomputed":True,"S9_renormalized":False,"EASE_schedule_adjusted":False})
        comparison=s8.prediction_changes(predicted,baseline,users)
        comparison["stage12_distinct_programs"]=len({j for rows in predicted.values() for j in rows})
        comparison["stage9_distinct_programs"]=len({j for rows in baseline.values() for j in rows})
        s.save("final/comparison_with_stage9.json",comparison)
        s.save("final/provenance.json",{"submission":str(s.DEST.relative_to(ROOT)),"sha256":s6.sha256_file(s.DEST),"selected":params,
             "candidates":6632,"template_users":len(users),"future_viewing_used":False,"best_scored_submission":str(s.BEST.relative_to(ROOT)),
             "returned_score":None,"source_manifest":"preservation_before.json","code_provenance":"code_provenance.json",
             "full_refit_with_schedule_counts_IO_seconds":time.perf_counter()-started,
             "assistance":"Codex suggested the experiment; implementation and automated checks performed in this OpenCode session; personal verification pending",
             "normalization_convention":"explicit experiment convention, not standard/proven-optimal paper preprocessing"})
        s.checkpoint("full_refit_complete",sha256=s6.sha256_file(s.DEST))
    s.attempt("full_history",params["name"],"conditional_refit_not_new_configuration",refit)


if __name__=="__main__": execute()
