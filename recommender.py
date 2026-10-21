"""Reproducible implicit-feedback recommenders; no competition solution code used."""
import re,time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def ap_at_5(predicted,relevant):
    relevant=set(relevant)
    if not relevant:return 0.0
    hits=0;total=0.;seen=set()
    for rank,item in enumerate(predicted[:5],1):
        if item in relevant and item not in seen:
            hits+=1;total+=hits/rank
        seen.add(item)
    return total/min(5,len(relevant))


def rank_five(scores,candidates,popularity):
    """Positive model scores, then common popularity fallback; ID resolves ties."""
    assert len(candidates)>=5 and len(set(candidates))==len(candidates)
    scores=np.asarray(scores);candidates=np.asarray(candidates)
    valid=np.isfinite(scores)&(scores>0)
    ordered=np.flatnonzero(valid)[np.lexsort((candidates[valid],-scores[valid]))]
    result=ordered[:5].tolist()
    if len(result)<5:
        for j in np.lexsort((candidates,-np.asarray(popularity))):
            if int(j) not in result:result.append(int(j))
            if len(result)==5:break
    return candidates[result].tolist()


def build_axes(events,prediction_users,candidates,cutoff):
    """Vocabulary from past behaviour and supplied prediction identities only."""
    past=events['end']<=cutoff
    users=np.unique(np.concatenate([events['user_id'][past],prediction_users]))
    items=np.unique(np.concatenate([events['item'][past],np.asarray(candidates,dtype=np.int64)]))
    return users,items


def interaction_matrix(events, users, items, cutoff, half_life=28, threshold=.8, mode='count'):
    # Complete broadcasts ending by cutoff only; never consume future behaviour.
    mask=(events['end']<=cutoff)&(events['coverage']>=threshold)
    u=events['user_id'][mask];i=events['item'][mask]
    weight=np.ones(len(u),dtype=np.float64)
    if mode=='coverage':weight=events['coverage'][mask].copy()
    if half_life:
        age=(cutoff-events['end'][mask])/86400.
        weight*=np.exp2(-age/half_life)
    rr=np.searchsorted(users,u);cc=np.searchsorted(items,i)
    matrix=sparse.coo_matrix((weight,(rr,cc)),shape=(len(users),len(items))).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros()
    return matrix


def ground_truth(events,cutoff,end):
    mask=(events['start']>=cutoff)&(events['end']<=end)&(events['coverage']>=.8)
    counts={}
    for user,item in zip(events['user_id'][mask],events['item'][mask]):
        key=int(user);ii=int(item);d=counts.setdefault(key,{});d[ii]=d.get(ii,0)+1
    truth={u:[i for i,n in sorted(d.items(),key=lambda x:(-x[1],x[0]))[:5]] for u,d in counts.items()}
    return truth,counts


def content_features(items,metadata):
    # Field-prefixed tokens: no artificial ID features, future metadata or labels.
    docs=[]
    for item in items:
        attrs=metadata.get(str(item),{});tokens=[]
        for field,values in attrs.items():
            prefix={'tv_show_title':'title','tv_show_category':'category','tv_show_genre_1':'genre','tv_show_genre_2':'genre','tv_show_genre_3':'genre','year_of_production':'year'}.get(field,field)
            words=set()
            for text in values:
                words.update(re.findall(r'[^\W_]+',text.lower()))
            tokens.extend(prefix+'_'+word for word in sorted(words) if len(word)>1)
        docs.append(' '.join(tokens))
    # Empty unknown items transform to zero vectors, followed by common fallback.
    vectorizer=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',sublinear_tf=True,min_df=1,max_features=30000,dtype=np.float64)
    vectorizer.fit([doc for doc in docs if doc])
    features=vectorizer.transform(docs)
    return features,vectorizer


def cbf_scores(matrix,features,user_index,candidate_index):
    profiles=normalize(matrix[user_index]@features,norm='l2')
    return (profiles@features[candidate_index].T).toarray()


def als_half_step(matrix,other,factors,regularization,alpha):
    """Exact sparse normal equations, c_ui=1+alpha*log(1+r_ui)."""
    base=other.T@other+regularization*np.eye(factors)
    result=np.zeros((matrix.shape[0],factors))
    for row in range(matrix.shape[0]):
        left,right=matrix.indptr[row:row+2];idx=matrix.indices[left:right]
        if len(idx)==0:continue
        y=other[idx];extra=alpha*np.log1p(matrix.data[left:right])
        a=base+(y.T*extra)@y;b=y.T@(1+extra)
        result[row]=np.linalg.solve(a,b)
    return result


def implicit_als(matrix,factors=32,regularization=1.,alpha=10.,iterations=12,seed=42):
    rng=np.random.default_rng(seed);item=rng.normal(0,.01,(matrix.shape[1],factors))
    transposed=matrix.T.tocsr();user=None
    with threadpool_limits(limits=1):
        for epoch in range(iterations):
            user=als_half_step(matrix,item,factors,regularization,alpha)
            item=als_half_step(transposed,user,factors,regularization,alpha)
    return user,item


def association_scores(matrix,user_index,candidate_index,min_support=3,shrinkage=10):
    """Singleton rules i -> j, smoothed confidence n(i,j)/(n(i)+shrinkage)."""
    binary=matrix.copy();binary.data[:]=1
    counts=(binary.T@binary).tocoo();keep=counts.data>=min_support
    counts=sparse.coo_matrix((counts.data[keep],(counts.row[keep],counts.col[keep])),shape=counts.shape).tocsr()
    counts.setdiag(0);counts.eliminate_zeros()
    support=np.asarray(binary.sum(axis=0)).ravel()
    rules=sparse.diags(1/(support+shrinkage))@counts
    profiles=normalize(matrix[user_index],norm='l1')
    return (profiles@rules[:,candidate_index]).toarray()


def normalized_scores(scores):
    z=np.maximum(np.nan_to_num(scores),0.)
    scale=z.max(axis=1,keepdims=True)
    return np.divide(z,scale,out=np.zeros_like(z),where=scale>0)
