#!/usr/bin/env python3
"""Conditional Stage 13 refit, sharing the exact historical prediction path."""
from __future__ import annotations

import csv
import importlib.util
import time
from collections import Counter
from pathlib import Path

spec=importlib.util.spec_from_file_location("stage13_saved_study",Path(__file__).with_name("study.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
s6,s8,s9,r11,np,e=s.s6,s.s8,s.s9,s.r11,s.np,s.e
ROOT,OUT=s.ROOT,s.OUT


def execute():
    selection=s6.read_json(OUT/"selection.json"); selected=selection["selected"]
    if selected is None:
        assert not s.DEST.exists()
        s.save("final_decision.json",{"refitted":False,"submission_generated":False,"retain":"Stage 12","reason":"no qualifying winner"})
        return
    if not (OUT/"selected_configuration.json").exists():
        s.save("selected_configuration.json",{"parameters":selected,"declaration_sha256":s6.sha256_file(OUT/"declaration.json"),
             "selection_sha256":s6.sha256_file(OUT/"selection.json"),"new_unique_from_refit":0,"total_unique":58,
             "postprocessing":"study.predict is the exact common historical/final implementation"})
    for scope in s.SCOPES:
        if (OUT/f"final_reproduction/{scope}/checks.json").exists():
            assert s6.read_json(OUT/f"final_reproduction/{scope}/checks.json")["all_passed"]; continue
        def reproduce(scope=scope):
            data,S9,repeat,baseline,raw,provenance=s.load_inputs(scope)
            cached=s6.read_json(OUT/f"{scope}/opportunities.json")
            past={int(k):v for k,v in cached["past"].items()}; target={int(k):v for k,v in cached["target"].items()}
            factors=s9.factors_for(past,target,tuple(cached["training_window"]),tuple(cached["target_window"]),data["val_candidates"])
            predicted,fallback,runtime=s.predict(data,S9,raw,selected,factors)
            expected=s6.read_json(OUT/f"{scope}/methods/{selected['name']}/top5.json")
            metrics,aps=s9.independent_metrics(predicted,data["labels"]["primary_top5"],data["cohort"])
            prior_metrics=s6.read_json(OUT/f"{scope}/methods/{selected['name']}/metrics.json")
            checks={"top5_exact":predicted==expected,"independent_metrics_match":all(abs(metrics[k]-prior_metrics[k])<1e-12 for k in ("map_at_k","map_with_positive","map_with_at_least_k_positive")),
                    "reused_verified_EASE_fit":True}
            s.save(f"final_reproduction/{scope}/top5.json",predicted); s.save(f"final_reproduction/{scope}/metrics.json",metrics)
            s.save(f"final_reproduction/{scope}/per_user_ap5.json",aps); s.save(f"final_reproduction/{scope}/runtime.json",runtime)
            s.save(f"final_reproduction/{scope}/checks.json",{"all_passed":all(checks.values()),"checks":checks})
            assert all(checks.values()),checks
            s.checkpoint("selected_historical_equivalence",scope=scope)
        s.attempt(scope,selected["name"],"selected_final_implementation_reproduction_not_new",reproduce)
    if (OUT/"final/provenance.json").exists():
        assert s.DEST.exists(); return
    def refit():
        assert not s.DEST.exists(),"Refuse to overwrite an existing submission"
        started=time.perf_counter(); cfg=s6.load_config(ROOT/"config/stage5.json")
        history=s8.raw_schedule(); historical=s8.subset(history,s9.TRAIN[1])
        target,target_audit=s9.read_target_schedule(cfg.schedule_files["test"],s9.TARGET)
        candidates=sorted(set(int(j) for j in target["show"]))
        assert len(candidates)==6632 and candidates==s6.read_json(cfg.out("candidates","candidates_test.json"))["candidate_ids"]
        assert s9.identity_set(historical)==s9.identity_set(s8.cache_schedule(cfg.out("airings","airings_history.csv")))
        assert s9.identity_set(target)==s9.identity_set(s8.cache_schedule(cfg.out("airings","candidate_airings_test.csv")))
        past=Counter(int(j) for j in historical["show"]); future=Counter(int(j) for j in target["show"])
        factors=s9.factors_for(past,future,s9.TRAIN,s9.TARGET,candidates)
        counts=s6.read_count_variants(cfg.out("interactions","history_user_program_counts.csv"))
        table,_=s6.load_table(cfg.out("interactions","history_user_airing.npz"),cfg.out("airings","airings_history.csv"))
        count_module=s.p12.count_module()
        reconstructed,count_audit=count_module.count_distinct_airings(table,s9.TRAIN[0],s9.TRAIN[1])
        assert reconstructed==counts["primary80"]
        s.save("final/training_count_audit.json",{**count_audit,"matches_verified_full_history_counts":True}); del table,reconstructed
        ids=s6.read_json(cfg.out("views","id_maps_history.json"))["history"]["user_ids"]
        with cfg.submission_file.open(newline="",encoding="utf-8") as handle:
            reader=csv.DictReader(handle); headers=reader.fieldnames; users=[row["user_id"] for row in reader]
        popularity,stats,pop_rows=s6.build_popularity(counts,candidates)
        data={"cohort":users,"count_variants":counts,"user_to_train_idx":{u:i for i,u in enumerate(ids)},"val_candidates":candidates,
              "popularity_val":popularity,"popularity":stats,"cutoff":s9.TRAIN[1]}
        S9_started=time.perf_counter()
        repeat=s6.scores_for(users,data["user_to_train_idx"],counts["sens50"],candidates)
        adjusted=s9.adjust_repeat_scores(repeat,factors,candidates)
        ar,ar_audit=s8.ar_scores(data)
        S9={u:s8.blend(adjusted[u],ar[u],.7,.3) for u in users}
        S9_seconds=time.perf_counter()-S9_started
        oldS9=ROOT/"outputs/stage9/kuznetsov_vladimir_midterm_submission_schedule.csv"
        with oldS9.open(newline="",encoding="utf-8") as handle:
            previousS9={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in csv.DictReader(handle)}
        assert s6.rank_scores(S9,data)[0]==previousS9
        memory=r11.resource_plan(data); s.save("final/memory_estimate.json",memory)
        if not memory["allowed"]: raise MemoryError("Fixed full-history fit exceeds unchanged resource budget; no model replacement")
        EASE_started=time.perf_counter()
        X,learned_users,learned_items,statistics=count_module.transformed_matrix(counts["primary80"])
        G=count_module.gram_matrix(X); B,fit_audit=e.fit_from_gram(G,1000)
        normalized,raw,score_audit=e.candidate_scores(X,B,learned_users,learned_items,users,data["user_to_train_idx"],candidates)
        EASE_seconds=time.perf_counter()-EASE_started
        # Preserve the unadjusted full Stage 12 control exactly, not merely its
        # historical scores, before generating a new opportunity-adjusted CSV.
        control_pred,_,_=s.predict(data,S9,raw,s.CONTROL)
        with s.BEST.open(newline="",encoding="utf-8") as handle:
            previous={row["user_id"]:[int(j) for j in row["tv_show_id"].split()] for row in csv.DictReader(handle)}
        assert control_pred==previous,"Rebuilt full Stage 12 control does not match the best CSV"
        previous_stats=s6.read_json(ROOT/"outputs/stage12/final/input_statistics.json")
        assert statistics["normalization_scalar"]==previous_stats["normalization_scalar"]
        predicted,fallback,runtime=s.predict(data,S9,raw,selected,factors)
        from stage5.submission import write_submission,validate_submission
        write_submission(s.DEST,users,predicted)
        checked=validate_submission(s.DEST,users,candidates); assert checked["all_passed"]
        with s.DEST.open(newline="",encoding="utf-8") as handle: assert csv.DictReader(handle).fieldnames==headers
        s.save("final/submission_checks.json",checked); s.save("final/predictions.json",predicted)
        s.save("final/input_statistics.json",statistics); s.save("final/fit_audit.json",fit_audit); s.save("final/score_audit.json",score_audit)
        s.save("final/AR_audit.json",ar_audit); s.save("final/fallback.json",fallback); s.save("final/popularity.json",pop_rows)
        s.save("final/cold_start.json",s6.cold_start_stats(users,data["user_to_train_idx"],counts))
        s.save("final/opportunities.json",{"past":past,"target":future,"factors":factors,"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET),
             "training_weeks":s9.weeks(s9.TRAIN),"target_weeks":s9.weeks(s9.TARGET),"recomputed_for_full_history":True})
        s.save("final/configuration.json",{"selected":selected,"training_window":list(s9.TRAIN),"target_window":list(s9.TARGET),
             "normalization_scalar":statistics["normalization_scalar"],"scalar_recalculated_from_full_training":True,"S9_renormalized":False,
             "EASE_adjustment_order":"complete scores -> clip zero -> multiply factor -> candidate-only per-user max"})
        comparison=s8.prediction_changes(predicted,previous,users)
        comparison["stage13_distinct_programs"]=len({j for rows in predicted.values() for j in rows})
        comparison["stage12_distinct_programs"]=len({j for rows in previous.values() for j in rows})
        s.save("final/comparison_with_stage12.json",comparison)
        s.save("final/control_reproduction.json",{"full_Stage12_top5_exact":True,"full_S9_top5_exact":True,"full_training_scalar_exact":True})
        s.save("final/runtime.json",{**runtime,"S9_fit_and_score_seconds":S9_seconds,"weighted_EASE_matrix_fit_query_seconds":EASE_seconds,
             "full_refit_with_schedule_counts_and_IO_seconds":time.perf_counter()-started})
        s.save("final/target_schedule_audit.json",target_audit)
        s.save("final/provenance.json",{"submission":str(s.DEST.relative_to(ROOT)),"sha256":s6.sha256_file(s.DEST),"selected":selected,
             "candidates":6632,"users":1260,"best_scored_submission":str(s.BEST.relative_to(ROOT)),"returned_score":None,
             "future_viewing_used":False,"source_manifest":"preservation_before.json","code_provenance":"code_provenance.json",
             "assistance":"Codex suggested the follow-up; calculations and automated checks performed in this OpenCode session; personal verification pending",
             "opportunity_adjustment_interpretation":"forecasting heuristic, not calibrated probability"})
        s.checkpoint("selected_full_refit_complete",sha256=s6.sha256_file(s.DEST))
    s.attempt("full_history",selected["name"],"conditional_final_refit_not_new_configuration",refit)


if __name__=="__main__":execute()
