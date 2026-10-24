#!/usr/bin/env python3
"""Stage 11 project-only diagnostics and fixed EASE study orchestration."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("stage10_project_helpers",ROOT/"outputs/stage10/code/study.py")
s10=importlib.util.module_from_spec(spec); spec.loader.exec_module(s10)
s9,s8,s6=s10.s9,s10.s8,s10.s6
np=s10.np
OUT=ROOT/"outputs/stage11"
s10.OUT=OUT/"cache"; s9.OUT=OUT/"cache"; s8.OUT=OUT/"cache"; s6.OUT=OUT/"cache"
BEST=ROOT/"outputs/stage9/kuznetsov_vladimir_midterm_submission_schedule.csv"
BEST_SHA="f88ac045cd9c4347e72fd7a3a90f4c97748d44a055af8faf5c93a684fa0f9ea9"
DEST=OUT/"kuznetsov_vladimir_midterm_submission_ease.csv"
SCOPES=("horizon12","original10")
LAMBDAS=(100,1000)
BETAS=(.1,.2,.3,1.)
CONFIGS=[{"name":f"ease_l{reg}_b{round(beta*100)}","lambda":reg,"beta":beta} for reg in LAMBDAS for beta in BETAS]
SEED,RESAMPLES=137,10000


def save(path,value):
    path=OUT/path; path.parent.mkdir(parents=True,exist_ok=True); s6.write_json(path,value)


def checkpoint(phase,**details):
    value={"phase":phase,"time_epoch":time.time(),**details}
    save("checkpoints/progress.json",value); save(f"checkpoints/{time.time_ns()}_{phase}.json",value)


def manifest():
    # Hashing prior files is preservation bookkeeping, not consumption of their
    # diagnostic findings. No outside-project diagnostic file is opened.
    return {str(p.relative_to(ROOT)):{"sha256":s6.sha256_file(p),"bytes":p.stat().st_size}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_relative_to(OUT)}


def attempt(scope,name,category,function):
    def log(event,**details):
        with (OUT/"attempts.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps({"scope":scope,"name":name,"category":category,"event":event,"time_epoch":time.time(),**details},sort_keys=True)+"\n")
    log("started")
    try: result=function()
    except Exception:
        log("failed",traceback=traceback.format_exc()); checkpoint("failed",scope=scope,name=name); raise
    log("completed"); return result


def load_split(scope,history=None):
    return s10.load_split(scope,history)


def baseline_scores(data,repeat,ar,past,target):
    factors=s9.factors_for(past,target,(s6.START,data["cutoff"]),(data["cutoff"],s6.FINAL_END),data["val_candidates"])
    adjusted=s9.adjust_repeat_scores(repeat,factors,data["val_candidates"])
    # Deliberately do not max-normalize this completed S9 score again.
    scores={u:s8.blend(adjusted[u],ar[u],.7,.3) for u in data["cohort"]}
    predictions,detail=s6.rank_scores(scores,data)
    expected=s6.read_json(ROOT/f"outputs/stage9/reproduction/{scope_name(data)}/final_implementation_top5.json")
    assert predictions==expected,"beta=0 Stage 9 reproduction failed"
    return scores,predictions,detail


def scope_name(data):
    return "horizon12" if data["cutoff"]==s8.HORIZON_CUTOFF else "original10"


def pair_diagnosis(data,repeat,ar,baseline):
    groups={name:{"pairs":0,"top5_pairs":0,"programs":set(),"top5_programs":set()} for name in
            ("personal_repeat50","no_repeat50_positive_AR","neither_repeat50_nor_AR")}
    global_users=defaultdict(set)
    for user,items in data["count_variants"]["primary80"].items():
        for program in items: global_users[program].add(user)
    unobserved={"without_repeat50":{"pairs":0,"programs":set()},"without_any_positive_coverage":{"pairs":0,"programs":set()}}
    all_pairs=0; personal_sizes={}
    for u in data["cohort"]:
        index=data["user_to_train_idx"].get(u)
        any_items=data["count_variants"]["positive"].get(index,{})
        personal_sizes[u]=len(repeat[u])
        for program in data["labels"]["primary_top5"][u]:
            all_pairs+=1
            key="personal_repeat50" if repeat[u].get(program,0)>0 else "no_repeat50_positive_AR" if ar[u].get(program,0)>0 else "neither_repeat50_nor_AR"
            row=groups[key]; row["pairs"]+=1; row["programs"].add(program)
            if program in baseline[u]: row["top5_pairs"]+=1; row["top5_programs"].add(program)
            other_users=global_users.get(program,set())-({index} if index is not None else set())
            if other_users and repeat[u].get(program,0)<=0:
                unobserved["without_repeat50"]["pairs"]+=1; unobserved["without_repeat50"]["programs"].add(program)
            if other_users and program not in any_items:
                unobserved["without_any_positive_coverage"]["pairs"]+=1; unobserved["without_any_positive_coverage"]["programs"].add(program)
    for row in groups.values():
        row["distinct_programs"]=len(row.pop("programs")); row["distinct_top5_programs"]=len(row.pop("top5_programs"))
    for row in unobserved.values(): row["distinct_programs"]=len(row.pop("programs"))
    learned=set(global_users)
    relevant_global=[(u,s) for u in data["cohort"] for s in data["labels"]["primary_top5"][u] if s in learned]
    relevant_missing=[(u,s) for u in data["cohort"] for s in data["labels"]["primary_top5"][u] if s not in learned]
    return {"personal_history_definition":"at least one training airing with >=50% actual observed coverage; eligible program IDs only",
         "AR_definition":"positive original association80 candidate score, training-only",
         "global_definition":"at least one >=80% training airing from any training user; all training users included",
         "total_relevant_user_program_pairs":all_pairs,"groups":groups,
         "personally_unobserved_with_other_user_primary80_evidence":unobserved,
         "globally_primary80_observed_relevant_pairs":len(relevant_global),"globally_primary80_unobserved_relevant_pairs":len(relevant_missing),
         "distinct_relevant_programs":len({s for u in data["cohort"] for s in data["labels"]["primary_top5"][u]}),
         "users_fewer_than20_personal_repeat50_eligible":sum(n<20 for n in personal_sizes.values()),
         "users_at_least20_personal_repeat50_eligible":sum(n>=20 for n in personal_sizes.values()),
         "distinct_counts_are_not_additive_across_user_pair_groups":True,
         "interpretation":"EASE could supply or rerank scores for personally unobserved items learned from other training users, including items missed by pruned AR rules. It cannot learn an item with no >=80% global training evidence, or personalize a user with no usable >=80% history. Diagnostics are not proof of improvement; labels are not training features."}


def diagnose():
    if (OUT/"diagnostics/complete.json").exists(): return
    if not (OUT/"preservation_before.json").exists(): save("preservation_before.json",manifest())
    assert s6.sha256_file(BEST)==BEST_SHA
    save("source_attribution.json",{"external_assistance":"Codex, an external AI assistant, suggested EASE after discussing existing results with the user",
         "personal_analysis_or_verification_claimed":False,"personal_verification_status":"pending; no personal verification claimed",
         "implementation_and_diagnostic_execution":"performed in this OpenCode session using project data and primary paper",
         "external_Codex_diagnostic_files_read_or_used":False,"outside_project_implementations_read_or_used":False,
         "paper":{"author":"Harald Steck","title":"Embarrassingly Shallow Autoencoders for Sparse Data","venue":"WWW 2019",
                  "url":"https://arxiv.org/abs/1905.03375","full_text_consulted":"https://arxiv.org/html/1905.03375v1",
                  "relevant_sections":"3.1 Eq.4 and Eq.8; 3.3 Algorithm 1; 4.2/5.2 negative weights"}})
    history=s8.raw_schedule(); summary={}
    for scope in SCOPES:
        def run(scope=scope):
            data,repeat,ar,past,target,shared=load_split(scope,history)
            scores,baseline,detail=baseline_scores(data,repeat,ar,past,target)
            diagnosis=pair_diagnosis(data,repeat,ar,baseline)
            save(f"diagnostics/{scope}.json",diagnosis)
            save(f"cache/{scope}/S9_scores.json",scores); save(f"cache/{scope}/repeat50_scores.json",repeat)
            save(f"cache/{scope}/association_scores.json",ar); save(f"cache/{scope}/baseline_top5.json",baseline)
            save(f"cache/{scope}/baseline_fallback.json",detail); save(f"cache/{scope}/shared_runtime.json",shared)
            summary[scope]=diagnosis
        attempt(scope,"baseline_coverage_diagnosis","diagnostic_not_new_configuration",run)
    save("diagnostics/complete.json",{"completed_before_EASE_implementation":True,"splits":summary,"time_epoch":time.time()})
    checkpoint("independent_diagnosis_complete")


def declare():
    if (OUT/"declaration.json").exists(): return
    assert (OUT/"diagnostics/complete.json").exists()
    catalog=s6.read_json(ROOT/"outputs/stage8/search_history.json")["configurations"]
    catalog+=s6.read_json(ROOT/"outputs/stage10/search_history.json")["new_configurations"]
    assert len(catalog)==41
    checked=[]
    for row in catalog:
        name=row["name"]
        if "parameters_path" in row:
            path=ROOT/row["parameters_path"]; params=s6.read_json(path)
        else:
            path=ROOT/f"outputs/stage10/horizon12/methods/{name}/parameters.json"; params=s6.read_json(path)
        checked.append({"name":name,"path":str(path.relative_to(ROOT)),"sha256":s6.sha256_file(path),
                        "has_EASE_model":("ease" in name.lower() or params.get("method")=="EASE")})
    duplicates={p["name"]:[] for p in CONFIGS}
    # No preceding configuration is EASE; do not substitute another experiment.
    assert not any(r["has_EASE_model"] for r in checked),"A previous EASE configuration needs exact-match review before evaluation"
    declarations={}
    for stage,file in ((6,"plan.json"),(7,"plan.json"),(8,"plan.json"),(9,"configuration.json"),(10,"declaration.json")):
        path=ROOT/f"outputs/stage{stage}/{file}"
        declarations[str(path.relative_to(ROOT))]={"sha256":s6.sha256_file(path),"parameters":s6.read_json(path)}
    save("duplicate_audit.json",{"checked_41_parameter_records":checked,"prior_declarations":declarations,
         "exact_duplicates":duplicates,"replace_duplicate_slot":False})
    save("declaration.json",{"declared_before_model_evaluation":True,"suggestion_source":"external AI assistant Codex; not personal user analysis",
         "independent_diagnosis":"diagnostics/complete.json; produced in this session before EASE implementation",
         "configurations":CONFIGS,"previous_unique":41,"new_unique_budget":8,"maximum_total_unique":49,"control_beta":0,
         "implementation":{"X":"binary float64 CSR from all >=80% actual training interactions, one per user/program; all training users/items, including noncandidate history",
             "mapping":"sorted training user indices and integer program IDs; no validation/test interactions in learned mapping",
             "G":"X.T @ X in float64, never integer co-occurrence",
             "P":"float64 inverse of G+lambda I using Cholesky factorization and solve against identity",
             "B":"-P_ij / P_jj columnwise, zero diagonal; preserve negative coefficients",
             "prediction":"X_user @ B; no seen-item exclusion; missing users/items zero",
             "fusion_component":"filter to eligible target candidates, clip completed prediction scores to zero, max-normalize positives separately per user",
             "schedule_adjustment_of_EASE":False,"S9":"exact .70*normalized adjusted repeat50 + .30*normalized original AR80; do not renormalize S9",
             "combined":"(1-beta)*S9 + beta*normalized_positive_EASE",
             "fitting_order":"sequential split, lambda; each fit reused for all four beta values; no replacements for resource failure",
             "memory":"estimate before allocation; conservative 8 dense item-square buffers + CSR Gram worst-case + cohort score arrays; fixed fit cap min(4 GiB, physical_memory/4); resource failures reported, no smaller model substituted"},
         "fixed_baseline":{"source":"outputs/stage9/configuration.json","repeat_coverage":.5,"count_weighting":True,"time_decay":None,
             "schedule_alpha":.5,"rate_ratio_clip":[.25,4.],"repeat_weight":.7,"association_weight":.3,"association":s6.AR_PARAMS,
             "normalization_and_ranking":s6.fusion_rules(),"previously_seen_items_allowed":True},
         "splits":"unchanged Stage 8 horizon12 and original10 labels/users/candidates/temporal rules",
         "diagnostic_groups":{"relevant_pair_personal_history":">=50% training-airing evidence",
             "globally_observed":">=80% training-airing evidence from any training user",
             "user_history_size":"number of eligible programs with personal >=50% training evidence; <20 vs >=20",
             "use":"diagnostics only; never model variants or selection objectives"},
         "bootstrap":{"seed":SEED,"resamples":RESAMPLES,"interval":"paired-user percentile 95%","user_order":"template order","cohorts":"all, positive, >=5 positives"},
         "selection":{"gain_each_split_at_least":.002,"horizon12_interval_lower_strictly_above":0.,
             "objective":"maximize smaller of the two gains","tie_break_1":"lower beta","tie_break_2":"lower measured computational cost, summed across splits; shared EASE fit and scoring plus direct fusion runtime",
             "remaining_exact_tie":"retain Stage 9 rather than invent another rule"},
         "uncertainty":"overlapping historical periods, repeatedly used users/labels, descriptive selection-unadjusted intervals; not independent confirmation",
         "kaggle_scores_not_optimization_targets":True,"conditional_refit":"one selected config only; no CSV if none qualifies",
         "output":str(DEST.relative_to(ROOT)),"no_external_diagnostic_dependency":True})
    save("current_best_submission.json",{"stage":9,"path":str(BEST.relative_to(ROOT)),"sha256":BEST_SHA,"private":.41656,"public":.43185,
         "designation":"best scored; unchanged until later returned score establishes otherwise"})
    checkpoint("eight_configurations_declared")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("diagnose","declare","prepare")); phase=parser.parse_args().phase
    if phase in ("diagnose","prepare"): diagnose()
    if phase in ("declare","prepare"): declare()


if __name__=="__main__":main()
