"""Read-only audit of the genuine OpenCode run; no training or model selection."""
from pathlib import Path
import csv, json, hashlib, math, sys, shutil
from collections import Counter
import numpy as np

ROOT=Path('/Users/vladimirkuznecov/Desktop/Midterm_quiz')
OUT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from stage4.ranking import top_k

def read(p): return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def metrics(pred,labels,users):
    aps={}
    for u in users:
        rel=set(labels[u]); seen=set(); hits=0; total=0.
        for k,j in enumerate(pred[u][:5],1):
            if j in rel and j not in seen: hits+=1; total+=hits/k
            seen.add(j)
        aps[u]=total/min(5,len(rel)) if rel else 0.
    groups=[users,[u for u in users if labels[u]],[u for u in users if len(set(labels[u]))>=5]]
    return {'all':sum(aps[u] for u in groups[0])/len(groups[0]),'positive':sum(aps[u] for u in groups[1])/len(groups[1]),'five':sum(aps[u] for u in groups[2])/len(groups[2]),'sizes':list(map(len,groups))},aps
def norm(d):
    m=max(d.values(),default=0)
    return {j:v/m for j,v in d.items() if v>0} if m else {}

report={'scope':'Codex report audit, outside the unmodified OpenCode export; two frozen standalone-component reporting checks, no refit/search/selection','methods':{},'controls':{}}
examples=read('outputs/stage4/two_user/analysis.json')
users2=[examples['heavy_user']['user_id'],examples['light_user']['user_id']]
meta={int(r['tv_show_id']):r for r in csv.DictReader((ROOT/'outputs/stage3/metadata/metadata_train.csv').open())}
paths={'Popularity':'outputs/stage3/methods/popularity/top5.json','CBF':'outputs/stage4/methods/cbf_genre_cast_log_pop/top5.json','AR':'outputs/stage4/methods/ar_c5_conf_max_80/top5.json'}
for scope in ['original10','horizon12']:
    users=read(f'outputs/stage13/{scope}/cohort.json'); candidates=read(f'outputs/stage13/{scope}/candidates.json'); allowed=set(candidates)
    labels=read(f'outputs/stage13/{scope}/labels.json')['primary_top5']
    factors={int(j):v for j,v in read(f'outputs/stage13/{scope}/opportunities.json')['factors'].items()}
    repeat={u:{int(j):v for j,v in d.items()} for u,d in read(f'outputs/stage10/cache/{scope}/repeat50_scores.json').items()}
    ar={u:{int(j):v for j,v in d.items()} for u,d in read(f'outputs/stage10/cache/{scope}/association_scores.json').items()}
    if scope=='original10':
        rows=list(csv.DictReader((ROOT/'outputs/stage3/candidates/popularity_val.csv').open()))
    else: rows=read('outputs/stage11/cache/horizon12/popularity.json')
    pop=[int(r['tv_show_id']) for r in rows]
    stats={int(r['tv_show_id']):(int(r['distinct_training_users_primary']),int(r['qualifying_training_airings_primary'])) for r in rows}
    with np.load(ROOT/f'outputs/stage12/{scope}/models/lambda1000/scores.npz') as d: raw=d['raw']
    R={}; A={}; E={}; H={}
    for i,u in enumerate(users):
        rs=norm({j:v*factors[j] for j,v in repeat[u].items() if j in allowed and v>0})
        ars=norm({j:v for j,v in ar[u].items() if j in allowed and v>0})
        es=norm({j:float(v)*factors[j] for j,v in zip(candidates,raw[i]) if v>0})
        # Preserve original float-operation order to verify exact Top-5 control.
        s9={j:.7*rs.get(j,0)+.3*ars.get(j,0) for j in rs.keys()|ars.keys()}
        fused={j:.9*v for j,v in s9.items() if v>0}
        for j,v in es.items(): fused[j]=fused.get(j,0)+.1*v
        for dst,scores in [(R,rs),(A,ars),(E,es),(H,fused)]: dst[u]=top_k(scores,candidates,pop,stats)[0]
    expected=read(f'outputs/stage13/{scope}/methods/E_EASE_opportunity/top5.json')
    assert H==expected,scope+' fused control mismatch'
    for name,pred in [('Adjusted repeat',R),('Adjusted EASE',E),('Final hybrid',H)]+([(k,read(v)) for k,v in paths.items()] if scope=='original10' else []):
        m,aps=metrics(pred,labels,users)
        report['methods'].setdefault(name,{})[scope]={'metrics':m,'two_users':{u:pred[u] for u in users2},'two_user_AP':{u:aps[u] for u in users2}}
        if name in ['Adjusted repeat','Adjusted EASE']:
            p=OUT/f'{scope}_{name.lower().replace(" ","_")}_top5.json'; p.write_text(json.dumps(pred,sort_keys=True,indent=2))
    saved=read(f'outputs/stage13/{scope}/methods/E_EASE_opportunity/metrics.json')
    assert abs(report['methods']['Final hybrid'][scope]['metrics']['all']-saved['map_at_k'])<1e-12
    report['controls'][scope]={'exact_final_top5':True,'independent_MAP':True,'cohort':len(users),'candidates':len(candidates)}
    if scope=='original10': report['example_labels']={u:labels[u] for u in users2}
report['users']={k:examples[k] for k in ['heavy_user','light_user','selection_rule']}
all_items=set(j for m in report['methods'].values() for scope in m.values() for ids in scope['two_users'].values() for j in ids)|set(j for ids in report['example_labels'].values() for j in ids)
report['metadata']={str(j):{'title':meta[j]['canonical_title'],'category':meta[j]['category']} for j in sorted(all_items)}
csvpath=ROOT/'outputs/stage13/kuznetsov_vladimir_midterm_submission_refined_ease.csv'
assert sha(csvpath)=='628edd64a9c4ecc26df1f0c2174d7378db4c15ebc4f01615a51f841ff44779d4'
template=list(csv.DictReader((ROOT/'sweettv-tv-program-recommender/submission.csv').open()))
with csvpath.open() as f:
    rr=csv.DictReader(f); header=rr.fieldnames; sub=list(rr)
with (ROOT/'sweettv-tv-program-recommender/submission.csv').open() as f: templateheader=csv.DictReader(f).fieldnames
candidatefile=read('outputs/stage5/candidates/candidates_test.json'); allowed=set(candidatefile['candidate_ids'])
assert header==templateheader and len(sub)==1260 and len(allowed)==6632
assert [r['user_id'] for r in sub]==[r['user_id'] for r in template]
assert all(len(ids:=r['tv_show_id'].split())==len(set(ids))==5 and all(int(j) in allowed for j in ids) for r in sub)
report['submission']={'sha256':sha(csvpath),'header':header,'users':1260,'candidates':6632,'format_checks':True}
session=Path('/Users/vladimirkuznecov/Downloads/sweet-tv-midterm-stage-1-investigation-plan.json')
d=json.loads(session.read_text()); parts=[p for m in d['messages'] for p in m.get('parts',[])]
report['session']={'sha256':sha(session),'bytes':session.stat().st_size,'messages':len(d['messages']),'parts':dict(Counter(p.get('type') for p in parts)),'native_shape':list(d),'unmodified':True}
report['prior_audit']=read('outputs/stage14/audit/validation.json')
(OUT/'report_data.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'methods':{k:{s:v['metrics']['all'] for s,v in d.items()} for k,d in report['methods'].items()},'submission':report['submission'],'controls':report['controls'],'session':report['session']},ensure_ascii=False,indent=2))
