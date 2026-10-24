"""Float64 asymmetric any-history -> fractional training-preference rules."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix,save_npz

MIN_MASS=5.
MIN_CONFIDENCE=.005
MIN_LIFT=1.
MAX_RULES=50
MAX_ANTECEDENTS=50


def membership(counts,K):
    if K<=0: raise ValueError("K must be positive")
    positive={}
    for program,value in counts.items():
        n=float(value)
        if not math.isfinite(n) or n<0 or not n.is_integer():
            raise ValueError("Expected nonnegative integer qualifying-airing counts")
        if n>0: positive[int(program)]=n
    if len(positive)<=K: return {j:1. for j in positive}
    boundary=sorted(positive.values(),reverse=True)[K-1]
    above=sum(n>boundary for n in positive.values()); tied=sum(n==boundary for n in positive.values())
    weight=(K-above)/tied
    return {j:1. if n>boundary else weight for j,n in positive.items() if n>=boundary}


def matrices(user_items,K):
    valid={u:{int(j):n for j,n in items.items() if n>0} for u,items in user_items.items() if any(n>0 for n in items.values())}
    users=sorted(valid); items=sorted({j for rows in valid.values() for j in rows}); columns={j:i for i,j in enumerate(items)}
    a_rows=[]; a_cols=[]; y_rows=[]; y_cols=[]; y_values=[]
    for row,u in enumerate(users):
        for j in valid[u]: a_rows.append(row); a_cols.append(columns[j])
        weights=membership(valid[u],K)
        for j,value in weights.items(): y_rows.append(row); y_cols.append(columns[j]); y_values.append(value)
    shape=(len(users),len(items))
    A=csr_matrix((np.ones(len(a_rows),dtype=np.float64),(a_rows,a_cols)),shape=shape,dtype=np.float64)
    A.sum_duplicates(); A.data[:]=1.; A.sort_indices()
    Y=csr_matrix((np.asarray(y_values,dtype=np.float64),(y_rows,y_cols)),shape=shape,dtype=np.float64)
    Y.sum_duplicates(); Y.sort_indices()
    C=(A.T@Y).tocsr()
    support_A=np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
    support_Y=np.asarray(Y.sum(axis=0)).ravel().astype(np.float64)
    expected=np.array([min(K,len(valid[u])) for u in users],dtype=np.float64)
    actual=np.asarray(Y.sum(axis=1)).ravel()
    assert np.allclose(actual,expected,rtol=0,atol=1e-12)
    assert A.dtype==Y.dtype==C.dtype==np.float64
    assert np.allclose(C.diagonal(),support_Y,rtol=1e-12,atol=1e-12)
    return {"users":users,"items":items,"item_col":columns,"user_items":valid,"A":A,"Y":Y,"C":C,
            "support_A":support_A,"support_Y":support_Y,"n_users":len(users),"K":K,
            "membership_row_sum_max_error":float(np.max(abs(actual-expected))) if len(users) else 0.}


def rule_statistics(model,i,j):
    if model["n_users"]<=0 or model["support_A"][i]<=0 or model["support_Y"][j]<=0: return None
    mass=float(model["C"][i,j])
    confidence=mass/float(model["support_A"][i])
    lift=confidence/(float(model["support_Y"][j])/model["n_users"])
    return mass,confidence,lift


def build_rules(model,candidates):
    candidates=set(int(j) for j in candidates); rules={}
    if model["n_users"]<=0: return rules
    for i in range(len(model["items"])):
        if model["support_A"][i]<=0: continue
        row=model["C"].getrow(i); entries=[]
        for j,mass in zip(row.indices,row.data):
            if i==j or model["items"][int(j)] not in candidates or mass<MIN_MASS or model["support_Y"][j]<=0: continue
            confidence=float(mass)/float(model["support_A"][i])
            lift=confidence/(float(model["support_Y"][j])/model["n_users"])
            if confidence>=MIN_CONFIDENCE and lift>=MIN_LIFT:
                entries.append((int(j),float(confidence),float(mass),float(lift)))
        entries.sort(key=lambda r:(-r[1],r[0]))
        if entries: rules[i]=entries[:MAX_RULES]
    return rules


def antecedents(counts,item_col):
    return [column for n,column in sorted(((float(n),item_col[j]) for j,n in counts.items() if j in item_col and n>0),
                                          key=lambda pair:(-pair[0],pair[1]))[:MAX_ANTECEDENTS]]


def fit(user_items,K,candidates):
    if K not in (5,10): raise ValueError("Only declared K=5 or K=10 fits are allowed")
    model=matrices(user_items,K); model["rules"]=build_rules(model,candidates)
    difference=model["C"]-model["C"].T
    model["details"]={"training_users":model["n_users"],"training_items":len(model["items"]),"K":K,
         "antecedents_with_rules":len(model["rules"]),"total_rules":sum(len(v) for v in model["rules"].values()),
         "aggregate":"max","score":"confidence","A_dtype":str(model["A"].dtype),"Y_dtype":str(model["Y"].dtype),"C_dtype":str(model["C"].dtype),
         "A_nnz":model["A"].nnz,"Y_nnz":model["Y"].nnz,"C_nnz":model["C"].nnz,
         "C_asymmetry_max_abs":float(np.max(abs(difference.data))) if difference.nnz else 0.,
         "fractional_C_entries":int(np.sum(model["C"].data!=np.floor(model["C"].data))),
         "membership_row_sum_max_error":model["membership_row_sum_max_error"],"total_membership_mass":float(model["Y"].sum()),
         "consequent_marginal":"support_Y, never support_A_j","C_rounded":False,
         "min_effective_mass":MIN_MASS,"min_confidence":MIN_CONFIDENCE,"min_lift":MIN_LIFT,
         "max_rules_per_antecedent":MAX_RULES,"max_antecedents_per_user":MAX_ANTECEDENTS,
         "interpretation":"retrospective preference associations, not calibrated future Top-5 probabilities"}
    return model


def user_scores(data,model):
    scores={}; with_rules=0
    for u in data["cohort"]:
        index=data["user_to_train_idx"].get(u)
        items=model["user_items"].get(index,{}) if index is not None else {}
        row={}
        for i in antecedents(items,model["item_col"]):
            for j,confidence,mass,lift in model["rules"].get(i,[]):
                program=model["items"][j]
                if confidence>row.get(program,0.): row[program]=confidence
        scores[u]=row; with_rules+=bool(row)
    return scores,{**model["details"],"users_with_rules":with_rules}


def save_model(directory,model):
    from stage2.io_utils import write_json
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for name in ("A","Y","C"): save_npz(directory/f"{name}.npz",model[name])
    write_json(directory/"statistics.json",{"users":model["users"],"items":model["items"],"support_A":model["support_A"].tolist(),
         "support_Y":model["support_Y"].tolist(),"K":model["K"],"details":model["details"]})
    write_json(directory/"rules.json",{str(model["items"][i]):[{"program":model["items"][j],"confidence":confidence,"mass":mass,"lift":lift}
         for j,confidence,mass,lift in rows] for i,rows in model["rules"].items()})
