"""EASE from Steck (WWW 2019), equations 4 and 8, implemented here in float64."""
from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix, vstack


def dimensions(user_items):
    users=[u for u,items in user_items.items() if any(v>0 for v in items.values())]
    items={s for u in users for s,v in user_items[u].items() if v>0}
    return len(users),len(items),sum(sum(v>0 for v in user_items[u].values()) for u in users)


def memory_estimate(nusers,nitems,nnz,ncohort,ncandidates):
    dense=8*nitems*nitems
    sparse_x=12*nnz+4*(nusers+1)
    worst_sparse_gram=12*nitems*nitems+4*(nitems+1)
    scores=8*ncohort*(nitems+3*ncandidates)
    return {"dense_item_square_buffer_bytes":dense,"assumed_dense_buffers":8,
            "sparse_X_bytes":sparse_x,"worst_sparse_Gram_bytes":worst_sparse_gram,
            "cohort_score_buffer_bytes":scores,
            "conservative_peak_bytes":8*dense+sparse_x+worst_sparse_gram+scores,
            "note":"conservative pre-allocation estimate; fit sequentially, no simultaneous lambda models"}


def binary_matrix(pairs):
    """Collapse arbitrary repeated positive pairs to binary float64 CSR entries."""
    pairs=list(pairs)
    users=sorted({u for u,s in pairs}); items=sorted({int(s) for u,s in pairs})
    user_rows={u:i for i,u in enumerate(users)}; item_columns={s:i for i,s in enumerate(items)}
    rows=[user_rows[u] for u,s in pairs]; cols=[item_columns[int(s)] for u,s in pairs]
    X=coo_matrix((np.ones(len(rows),dtype=np.float64),(rows,cols)),shape=(len(users),len(items)),dtype=np.float64).tocsr()
    X.sum_duplicates(); X.data[:]=1.; X.sort_indices()
    return X,users,items


def from_training_counts(user_items):
    return binary_matrix((u,s) for u,items in user_items.items() for s,count in items.items() if count>0)


def gram_matrix(X):
    if X.dtype!=np.float64 or not np.all(X.data==1):
        raise ValueError("EASE requires binary float64 training matrix")
    G=(X.T@X).toarray(order="F")
    assert G.dtype==np.float64
    assert np.array_equal(np.diag(G),np.asarray(X.sum(axis=0)).ravel())
    assert np.all(G>=0)
    return G


def array_digest(array):
    digest=hashlib.sha256()
    for i in range(0,len(array),128):
        digest.update(np.ascontiguousarray(array[i:i+128]).tobytes())
    return digest.hexdigest()


def fit_from_gram(G,regularization):
    """Solve for P using SPD Cholesky, then divide each column by its P_jj."""
    started=time.perf_counter()
    if G.dtype!=np.float64 or regularization<=0:
        raise ValueError("Need float64 Gram and positive regularization")
    n=G.shape[0]
    if G.shape!=(n,n) or n==0: raise ValueError("Nonempty square Gram required")
    A=np.array(G,dtype=np.float64,order="F",copy=True)
    A.flat[::n+1]+=regularization
    factor=cho_factor(A,lower=True,overwrite_a=True,check_finite=True)
    P=cho_solve(factor,np.eye(n,dtype=np.float64,order="F"),overwrite_b=True,check_finite=True)
    diag=np.diag(P).copy()
    assert np.all(diag>0) and np.all(np.isfinite(P))
    sample=sorted(set([0,n//2,n-1]))
    cols=P[:,sample].copy()
    unit_columns=np.zeros_like(cols)
    unit_columns[sample,np.arange(len(sample))]=1.
    residual=G@cols+regularization*cols-unit_columns
    expected=-cols/diag[sample][None,:]
    for k,j in enumerate(sample): expected[j,k]=0.
    # Reuse inverse storage for B. Negative learned coefficients are retained.
    P/=-diag[None,:]
    np.fill_diagonal(P,0.)
    B=P
    audit={"lambda":regularization,"dtype":str(B.dtype),"items":n,
           "zero_diagonal":bool(np.all(np.diag(B)==0)),
           "column_division_sample_max_error":float(np.max(np.abs(B[:,sample]-expected))),
           "inverse_residual_sample_max_abs":float(np.max(np.abs(residual))),
           "negative_coefficients":int(np.sum(B<0)),"positive_coefficients":int(np.sum(B>0)),
           "coefficient_min":float(B.min()),"coefficient_max":float(B.max()),
           "negative_coefficients_preserved":True,"B_sha256":array_digest(B),
           "direct_Cholesky_solve_coefficient_audit_seconds":time.perf_counter()-started}
    return B,audit


def candidate_scores(X,B,learned_users,learned_items,cohort,user_to_train_idx,candidates):
    """Complete XB first; filter targets before clipping/max-normalization."""
    started=time.perf_counter()
    rows={u:i for i,u in enumerate(learned_users)}; columns={s:i for i,s in enumerate(learned_items)}
    zero=csr_matrix((1,len(learned_items)),dtype=np.float64)
    queries=vstack([X.getrow(rows[user_to_train_idx[u]]) if u in user_to_train_idx and user_to_train_idx[u] in rows else zero
                    for u in cohort],format="csr",dtype=np.float64)
    completed=np.asarray(queries@B,dtype=np.float64)
    raw=np.zeros((len(cohort),len(candidates)),dtype=np.float64)
    target_indices=[i for i,s in enumerate(candidates) if s in columns]
    learned_indices=[columns[candidates[i]] for i in target_indices]
    raw[:,target_indices]=completed[:,learned_indices]
    assert np.all(np.isfinite(raw))
    positive=np.maximum(raw,0.)
    maximum=positive.max(axis=1,keepdims=True)
    normalized=np.divide(positive,maximum,out=np.zeros_like(positive),where=maximum>0)
    audit={"completed_score_negative_count":int(np.sum(raw<0)),"completed_score_positive_count":int(np.sum(raw>0)),
           "missing_learned_target_items":len(candidates)-len(target_indices),
           "users_without_usable_primary80_history":int(np.sum(np.diff(queries.indptr)==0)),
           "users_with_positive_EASE_candidates":int(np.sum(maximum[:,0]>0)),
           "previously_watched_items_excluded":False,"candidate_filter_precedes_max_normalization":True,
           "negative_coefficients_clipped":False,"completed_negative_scores_clipped_for_fusion":True,
           "direct_XB_filter_clip_normalize_seconds":time.perf_counter()-started}
    return normalized,raw,audit
