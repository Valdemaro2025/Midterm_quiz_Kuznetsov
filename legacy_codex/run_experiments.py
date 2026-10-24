"""Compare models on one documented temporal holdout; retain complete results."""
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy import sparse
from recommender import (ap_at_5,rank_five,interaction_matrix,ground_truth,content_features,
                         cbf_scores,implicit_als,association_scores,normalized_scores,build_axes)


def evaluate(scores,candidates,pop,users,truth,counts,history):
    rows=[];predictions=[]
    for n,user in enumerate(users):
        rec=rank_five(scores[n],candidates,pop);predictions.append(rec)
        rel=truth.get(int(user),[]);ap=ap_at_5(rec,rel)
        rows.append({'user_id':str(user),'ap5':ap,'relevant_items':len(rel),
                     'at_least_5_relevant':len(counts.get(int(user),{}))>=5,
                     'no_high_coverage_history':bool(history[n]==0),'predicted':rec,'truth':rel})
    main=[r for r in rows if r['at_least_5_relevant']]
    positive=[r for r in rows if r['relevant_items']]
    metrics={'map5_primary':float(np.mean([r['ap5'] for r in main])),
             'primary_users':len(main),'map5_all_positive':float(np.mean([r['ap5'] for r in positive])),
             'positive_users':len(positive),'map5_all_template':float(np.mean([r['ap5'] for r in rows])),
             'template_users':len(rows),'no_high_coverage_history_users':sum(r['no_high_coverage_history'] for r in rows),
             'unique_recommended_ids':len(set(i for p in predictions for i in p)),
             'repeat_fraction':None}
    return metrics,rows


def save_predictions(path,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','tv_show_id','ap5','truth'])
        for r in rows:w.writerow([r['user_id'],' '.join(map(str,r['predicted'])),f"{r['ap5']:.12f}",' '.join(map(str,r['truth']))])


def run(cache,out):
    out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    events=dict(np.load(cache/'events.npz'));catalog=json.loads((cache/'catalog.json').read_text())['validation']
    user_test=np.array([int(u) for u in json.loads((cache/'test_users.json').read_text())],dtype=np.uint64)
    candidates=np.array(catalog['candidates'],dtype=np.int64)
    users,items=build_axes(events,user_test,candidates,catalog['cutoff'])
    ui=np.searchsorted(users,user_test);ci=np.searchsorted(items,candidates)
    truth,counts=ground_truth(events,catalog['cutoff'],catalog['end'])
    matrices={h:interaction_matrix(events,users,items,catalog['cutoff'],half_life=h) for h in [0,14,28,56]}
    matrix=matrices[28];history=np.asarray(matrices[0][ui].sum(axis=1)).ravel()
    pop=np.asarray(matrix.sum(axis=0)).ravel()[ci]
    results=[];scores_cache={};allrows={}
    def record(name,scores,settings,fit_seconds,score_seconds):
        metrics,rows=evaluate(scores,candidates,pop,user_test,truth,counts,history)
        row={'method':name,'settings':settings,'fit_seconds':fit_seconds,'score_seconds':score_seconds,**metrics}
        repeat=[]
        for n,r in enumerate(rows):
            known=set(items[matrices[0][ui[n]].indices]);repeat.extend(i in known for i in r['predicted'])
        row['repeat_fraction']=float(np.mean(repeat))
        results.append(row);scores_cache[name]=scores;allrows[name]=rows
        save_predictions(out/(name+'_validation.csv'),rows)
        (out/'validation_results.json').write_text(json.dumps(results,indent=2))
        print(name,json.dumps(row),flush=True)
    t=time.perf_counter();record('popularity',np.broadcast_to(pop,(len(ui),len(ci))),{'half_life_days':28},0,time.perf_counter()-t)
    for h,m in matrices.items():
        t=time.perf_counter();s=m[ui][:,ci].toarray();record('repeat_h'+str(h),s,{'half_life_days':h},0,time.perf_counter()-t)
    t=time.perf_counter();features,vectorizer=content_features(items,catalog['training_metadata']);fit=time.perf_counter()-t
    t=time.perf_counter();s=cbf_scores(matrix,features,ui,ci);record('content',s,{'half_life_days':28,'features':'field-prefixed TF-IDF from training metadata only','max_features':30000,'vocabulary_size':len(vectorizer.vocabulary_)},fit,time.perf_counter()-t)
    for f,alpha,reg in [(32,10.,1.),(64,10.,1.),(32,40.,1.)]:
        t=time.perf_counter();uf,it=implicit_als(matrix,factors=f,alpha=alpha,regularization=reg,iterations=12);fit=time.perf_counter()-t
        t=time.perf_counter();s=uf[ui]@it[ci].T;name=f'als_f{f}_a{int(alpha)}'
        record(name,s,{'factors':f,'alpha':alpha,'regularization':reg,'iterations':12,'seed':42,'half_life_days':28,'confidence':'1+alpha*log1p(decayed high-coverage count)'},fit,time.perf_counter()-t)
    t=time.perf_counter();s=association_scores(matrix,ui,ci);record('association',s,{'half_life_days':28,'min_support_users':3,'shrinkage':10,'rule':'singleton i -> j confidence n(i,j)/(n(i)+10); self-rule excluded'},time.perf_counter()-t,0)
    # Small, disclosed score-fusion grid. Every component evaluated individually above.
    best_als=max([r for r in results if r['method'].startswith('als_')],key=lambda r:r['map5_primary'])['method']
    best_repeat=max([r for r in results if r['method'].startswith('repeat_')],key=lambda r:r['map5_primary'])['method']
    for component in [best_als,'content','association']:
        for weight in [.1,.25,.5]:
            t=time.perf_counter();s=(1-weight)*normalized_scores(scores_cache[best_repeat])+weight*normalized_scores(scores_cache[component])
            record(f'hybrid_{component}_w{weight}',s,{'repeat_component':best_repeat,'other_component':component,'other_weight':weight,'fusion':'nonnegative scores / per-user maximum, then convex sum'},0,time.perf_counter()-t)
    best=max(results,key=lambda r:r['map5_primary'])
    (out/'selected_strategy.json').write_text(json.dumps(best,indent=2))
    # Select examples using TRAINING activity only, before inspecting recommendation hits.
    eligible=np.flatnonzero(history>0);activity=history[eligible]
    examples=[]
    for label,q in [('light',.1),('heavy',.9)]:
        target=np.quantile(activity,q);pos=int(eligible[np.lexsort((user_test[eligible],np.abs(activity-target)))[0]])
        user=int(user_test[pos]);known=matrices[0][ui[pos]].tocoo();order=np.lexsort((items[known.col],-known.data))[:8]
        examples.append({'label':label,'user_id':str(user),'training_high_coverage_airings':float(history[pos]),
          'training_top':[{'id':int(items[known.col[j]]),'count':float(known.data[j])} for j in order],
          'truth':truth.get(user,[]),'methods':{k:allrows[k][pos] for k in ['popularity',best_repeat,'content',best_als,'association',best['method']]}})
    (out/'examples.json').write_text(json.dumps(examples,indent=2))
    # Save complete primary score vector for reproducible comparison, not just aggregate.
    np.savez_compressed(cache/'validation_score_vectors.npz',**{k:np.array([r['ap5'] for r in v]) for k,v in allrows.items()})
    protocol={'train_airings_end_at_or_before':'2020-05-18 00:00:00','validation_airings_start_at_or_after':'2020-05-18 00:00:00','validation_airings_end_at_or_before':'2020-07-27 00:00:00','candidate_count':len(candidates),'repeat_policy':'allowed for all models','target':'top 5 IDs by number of >=80% covered broadcasts; numeric ID ascending on count ties','primary_population':'template users with >=5 distinct qualifying validation programs','all_positive_population':'template users with >=1 qualifying validation program','empty_truth':'AP=0; additionally reported all-template mean','ranking':'positive primary score descending; numeric ID ascending for ties; common decayed popularity fills missing slots','all_models_train_only':True,'new_item_metadata':'no future schedule attributes used as features; only candidate availability','random_seed':42,'selection_warning':'Hyperparameters and hybrid selected on this holdout; scores are selection estimates, not an independent test.','environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'elapsed_seconds':time.perf_counter()-begin}
    (out/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','MAP@5_primary','MAP@5_all_positive','MAP@5_all_template','fit_seconds','score_seconds','catalog_coverage','repeat_fraction'])
        for r in results:w.writerow([r['method'],r['map5_primary'],r['map5_all_positive'],r['map5_all_template'],r['fit_seconds'],r['score_seconds'],r['unique_recommended_ids']/len(candidates),r['repeat_fraction']])
    print('SELECTED',json.dumps(best),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cache-dir',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);x=a.parse_args();run(x.cache_dir,x.output_dir)
