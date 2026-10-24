"""The single Stage 12 input change: global training-scaled log1p counts."""
from __future__ import annotations

import math
from collections import Counter,defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def count_distinct_airings(table,start,cutoff):
    """Verify counts from unioned coverage, collapsing duplicate representations.

    Distinct airing indices remain distinct repeat observations. This is not a
    replacement for upstream unioning of viewing fragments within each airing.
    """
    valid=((table["start"]>=start)&(table["start"]<cutoff)&(table["end"]<=cutoff)
           &(table["end"]>table["start"])&(table["airing_duration"]>0)
           &(table["covered_sec"]*5>=table["airing_duration"]*4))
    selected=np.flatnonzero(valid)
    counts=defaultdict(Counter)
    if not len(selected):
        return {},{"qualifying_rows":0,"distinct_user_airings":0,"duplicate_representations_dropped":0,
                   "qualifying_airings_sum":0,"max_qualifying_airing_end":None,"cutoff":cutoff}
    users=np.asarray(table["user_idx"][selected],dtype=np.int64)
    airing=np.asarray(table["airing_idx"][selected],dtype=np.int64)
    if np.any(users<0) or np.any(airing<0): raise ValueError("Negative identity index")
    base=int(airing.max())+1
    if int(users.max())>(np.iinfo(np.int64).max-int(airing.max()))//base:
        raise ValueError("Interaction identity encoding would overflow int64")
    keys=users*base+airing
    _,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
    for field in ("show_id","airing_duration","covered_sec","start","end"):
        values=table[field][selected]
        if not np.array_equal(values,values[first][inverse]):
            raise ValueError(f"Conflicting duplicate representation: {field}")
    unique=selected[first]
    for user,program in zip(table["user_idx"][unique],table["show_id"][unique]): counts[int(user)][int(program)]+=1
    result={u:dict(items) for u,items in counts.items()}
    return result,{"qualifying_rows":len(selected),"distinct_user_airings":len(unique),
             "duplicate_representations_dropped":len(selected)-len(unique),"qualifying_airings_sum":sum(sum(x.values()) for x in result.values()),
             "max_qualifying_airing_end":int(table["end"][unique].max()),"cutoff":cutoff,
             "qualification":"5*unioned_covered_seconds >= 4*original_duration; fully contained training airings"}


def transformed_matrix(user_items):
    triples=[]
    for user,items in user_items.items():
        for program,value in items.items():
            number=float(value)
            if not math.isfinite(number) or number<0 or not number.is_integer():
                raise ValueError("Qualifying-airing counts must be finite nonnegative integers")
            if number>0: triples.append((user,int(program),number))
    users=sorted({u for u,j,n in triples}); items=sorted({j for u,j,n in triples})
    user_rows={u:i for i,u in enumerate(users)}; item_cols={j:i for i,j in enumerate(items)}
    values=np.array([n for u,j,n in triples],dtype=np.float64)
    rows=[user_rows[u] for u,j,n in triples]; cols=[item_cols[j] for u,j,n in triples]
    X=coo_matrix((values,(rows,cols)),shape=(len(users),len(items)),dtype=np.float64).tocsr()
    X.sum_duplicates(); X.sort_indices()
    # Each dictionary entry is an already verified aggregate; do not binarize it.
    positive_counts=X.data.copy()
    Z=np.log1p(positive_counts)
    squared_sum=float(np.dot(Z,Z))
    scale=math.sqrt(X.nnz/squared_sum) if X.nnz and squared_sum>0 else 0.
    X.data[:]=scale*Z
    frobenius_squared=float(np.dot(X.data,X.data))
    histogram=Counter(int(n) for n in positive_counts)
    statistics={"training_users":len(users),"training_programs":len(items),"nnz":X.nnz,"dtype":str(X.dtype),
         "total_qualifying_airings":int(positive_counts.sum()),"count_histogram":dict(sorted(histogram.items())),
         "count_quantile_probabilities":[0,.25,.5,.75,.9,.99,1],
         "count_quantiles":np.quantile(positive_counts,[0,.25,.5,.75,.9,.99,1]).tolist() if X.nnz else [],
         "log_squared_sum":squared_sum,"normalization_scalar":scale,"X_squared_Frobenius_norm":frobenius_squared,
         "squared_norm_minus_nnz":frobenius_squared-X.nnz,"degenerate_no_positive_entries":X.nnz==0,
         "scalar_fit_scope":"all training positive counts only; no labels, target schedule or target behavior",
         "normalization_status":"declared experiment convention, not standard paper preprocessing or proven optimal"}
    assert np.isclose(frobenius_squared,X.nnz,rtol=1e-12,atol=1e-12)
    return X,users,items,statistics


def gram_matrix(X):
    if X.dtype!=np.float64 or not np.all(np.isfinite(X.data)) or np.any(X.data<0):
        raise ValueError("Expected finite nonnegative float64 weighted training matrix")
    G=(X.T@X).toarray(order="F")
    assert G.dtype==np.float64
    expected=np.asarray(X.multiply(X).sum(axis=0)).ravel()
    assert np.allclose(np.diag(G),expected,rtol=1e-12,atol=1e-12)
    return G
