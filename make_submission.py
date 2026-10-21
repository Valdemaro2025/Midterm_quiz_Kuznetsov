"""Refit the validation-selected strategy before the test cutoff and export CSV."""
import argparse,csv,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from recommender import interaction_matrix,association_scores,normalized_scores,rank_five,build_axes


def run(cache,results,output):
    start=time.perf_counter();events=dict(np.load(cache/'events.npz'))
    cat=json.loads((cache/'catalog.json').read_text())['test']
    selected=json.loads((results/'selected_strategy.json').read_text());settings=selected['settings']
    template=json.loads((cache/'test_users.json').read_text())
    test=np.array([int(u) for u in template],dtype=np.uint64)
    candidates=np.array(cat['candidates'],dtype=np.int64)
    users,items=build_axes(events,test,candidates,cat['cutoff']);ui=np.searchsorted(users,test);ci=np.searchsorted(items,candidates)
    # The complete log ends BEFORE 27 July; assert this rather than silently using it.
    audit=json.loads((cache/'data_audit.json').read_text())
    last_view_stop=audit['views']['max_end']
    assert int(np.datetime64(last_view_stop,'s').astype('int64'))<=cat['cutoff']
    par=settings.get('repeat_parameters',settings)
    repeat=interaction_matrix(events,users,items,cat['cutoff'],half_life=par['half_life_days'],threshold=par['threshold'],mode=par['mode'])
    repeat_scores=repeat[ui][:,ci].toarray()
    if par.get('exposure_exponent',0):
        train=np.array([cat['training_airings'].get(str(i),0) for i in candidates],dtype=float)
        future=np.array([cat['future_airings'].get(str(i),0) for i in candidates],dtype=float)
        exposure=np.clip((future+10)/(train+10),.25,4.)
        repeat_scores*=exposure**par['exposure_exponent']
    common=interaction_matrix(events,users,items,cat['cutoff'],half_life=28)
    pop=np.asarray(common.sum(axis=0)).ravel()[ci]
    if settings['kind']=='hybrid':
        scores=association_scores(common,ui,ci,min_support=settings['min_support_users'],shrinkage=settings['shrinkage'])
        weight=settings['other_weight'];scores=(1-weight)*normalized_scores(repeat_scores)+weight*normalized_scores(scores)
    elif settings['kind']=='repeat':scores=repeat_scores
    else:raise ValueError('Unsupported selected strategy: '+str(settings))
    predictions=[rank_five(scores[n],candidates,pop) for n in range(len(test))]
    with output.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','tv_show_id'])
        for user,ids in zip(template,predictions):w.writerow([user,' '.join(map(str,ids))])
    # Reopen written file and validate independently of its writer.
    valid=validate_submission(output,template,set(candidates.tolist()))
    manifest={'selected_method':selected['method'],'selected_settings':settings,'validation_map5_primary':selected['map5_primary'],
      'validation_primary_users':selected['primary_users'],'training_cutoff':'2020-07-27 00:00:00',
      'max_view_stop':last_view_stop,
      'max_training_airing_end':str(np.datetime64(int(events['end'][events['end']<=cat['cutoff']].max()),'s')),
      'candidate_count':len(candidates),'no_partial_history_template_users':int((np.asarray(repeat[ui].sum(axis=1)).ravel()==0).sum()),
      'unique_recommended_programs':len(set(i for p in predictions for i in p)),
      'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':len(predictions),'validation_checks':valid,
      'elapsed_seconds':time.perf_counter()-start,'python':platform.python_version(),
      'kaggle_status':'NOT SUBMITTED; no public/private score is claimed'}
    (results/'submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


def validate_submission(path,template,candidates):
    with path.open(newline='') as f:
        r=csv.DictReader(f);assert r.fieldnames==['user_id','tv_show_id'];rows=list(r)
    assert [r['user_id'] for r in rows]==template
    assert len(set(r['user_id'] for r in rows))==len(template)
    for row in rows:
        parts=row['tv_show_id'].split(' ');assert len(parts)==5 and all(parts)
        ids=list(map(int,parts));assert len(set(ids))==5
        assert all(i>0 and i in candidates for i in ids)
    return {'header':True,'template_user_ids_and_order':True,'unique_users':True,'five_distinct_positive_ids_each':True,'all_ids_in_future_schedule':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cache_dir,a.results_dir,a.output)
