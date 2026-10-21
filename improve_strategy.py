"""One controlled iteration: schedule exposure and partial-view evidence.
All additions are disclosed selection trials on the same holdout, not a new test.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from recommender import interaction_matrix,ground_truth,association_scores,normalized_scores,build_axes
from run_experiments import evaluate,save_predictions


def run(cache,out):
    e=dict(np.load(cache/'events.npz'));cat=json.loads((cache/'catalog.json').read_text())['validation']
    test=np.array([int(u) for u in json.loads((cache/'test_users.json').read_text())],dtype=np.uint64)
    candidates=np.array(cat['candidates']);users,items=build_axes(e,test,candidates,cat['cutoff']);ui=np.searchsorted(users,test);ci=np.searchsorted(items,candidates)
    raw=interaction_matrix(e,users,items,cat['cutoff'],half_life=0)
    common=interaction_matrix(e,users,items,cat['cutoff'],half_life=28)
    history=np.asarray(raw[ui].sum(axis=1)).ravel();pop=np.asarray(common.sum(axis=0)).ravel()[ci]
    truth,counts=ground_truth(e,cat['cutoff'],cat['end'])
    results=json.loads((out/'validation_results.json').read_text());examples=json.loads((out/'examples.json').read_text())
    # Training and prediction windows both last 10 weeks. Ratios are shrunk and bounded.
    train=np.array([cat['training_airings'].get(str(i),0) for i in candidates],dtype=float)
    future=np.array([cat['future_airings'].get(str(i),0) for i in candidates],dtype=float)
    exposure=np.clip((future+10)/(train+10),.25,4.)
    t=time.perf_counter();rules=association_scores(common,ui,ci);rule_seconds=time.perf_counter()-t
    repeat=raw[ui][:,ci].toarray();trials=[]
    for exponent in [.25,.5,1.]:
        trials.append((f'repeat_schedule_e{exponent}',repeat*exposure**exponent,{'kind':'repeat','half_life_days':0,'threshold':.8,'mode':'count','exposure_exponent':exponent,'exposure_smoothing':10,'exposure_clip':[.25,4]},0))
    for threshold in [.2,.5]:
        t=time.perf_counter();m=interaction_matrix(e,users,items,cat['cutoff'],half_life=0,threshold=threshold,mode='coverage');s=m[ui][:,ci].toarray()
        trials.append((f'repeat_partial_t{threshold}',s,{'kind':'repeat','half_life_days':0,'threshold':threshold,'mode':'coverage','exposure_exponent':0},time.perf_counter()-t))
    for name,s,settings,fit_seconds in trials:
        metrics,rows=evaluate(s,candidates,pop,test,truth,counts,history)
        repeat_fraction=np.mean([i in set(items[raw[ui[n]].indices]) for n,r in enumerate(rows) for i in r['predicted']])
        r={'method':name,'settings':settings,'fit_seconds':fit_seconds,'score_seconds':0,**metrics,'repeat_fraction':float(repeat_fraction)}
        results.append(r);save_predictions(out/(name+'_validation.csv'),rows)
        for ex in examples:
            pos=int(np.flatnonzero(test==int(ex['user_id']))[0]);ex['methods'][name]=rows[pos]
        print(json.dumps(r),flush=True)
    best_repeat=max([r for r in results if r['method'].startswith('repeat_')],key=lambda r:r['map5_primary'])
    name=best_repeat['method']
    if name=='repeat_h0':best_score=repeat;best_settings={'kind':'repeat','half_life_days':0,'threshold':.8,'mode':'count','exposure_exponent':0}
    else:
        match=next(t for t in trials if t[0]==name);best_score=match[1];best_settings=match[2]
    for weight in [.1,.25]:
        s=(1-weight)*normalized_scores(best_score)+weight*normalized_scores(rules)
        metrics,rows=evaluate(s,candidates,pop,test,truth,counts,history)
        settings={'kind':'hybrid','repeat_parameters':best_settings,'rule_half_life_days':28,'min_support_users':3,'shrinkage':10,'other_weight':weight,'fusion':'nonnegative scores / per-user maximum, then convex sum'}
        name=f'improved_hybrid_w{weight}';r={'method':name,'settings':settings,'fit_seconds':rule_seconds,'score_seconds':0,**metrics}
        r['repeat_fraction']=float(np.mean([i in set(items[raw[ui[n]].indices]) for n,rr in enumerate(rows) for i in rr['predicted']]))
        results.append(r);save_predictions(out/(name+'_validation.csv'),rows)
        for ex in examples:ex['methods'][name]=rows[int(np.flatnonzero(test==int(ex['user_id']))[0])]
        print(json.dumps(r),flush=True)
    best=max(results,key=lambda r:r['map5_primary'])
    # Expand old selection settings into self-contained reproducible configuration.
    if 'kind' not in best['settings']:
        if best['method'].startswith('hybrid_association'):
            best['settings'].update({'kind':'hybrid','repeat_parameters':{'half_life_days':0,'threshold':.8,'mode':'count','exposure_exponent':0},'rule_half_life_days':28,'min_support_users':3,'shrinkage':10})
        elif best['method'].startswith('repeat_'):
            best['settings'].update({'kind':'repeat','threshold':.8,'mode':'count','exposure_exponent':0})
    (out/'validation_results.json').write_text(json.dumps(results,indent=2))
    (out/'selected_strategy.json').write_text(json.dumps(best,indent=2))
    (out/'examples.json').write_text(json.dumps(examples,indent=2))
    print('SELECTED',json.dumps(best),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.cache_dir,a.output_dir)
