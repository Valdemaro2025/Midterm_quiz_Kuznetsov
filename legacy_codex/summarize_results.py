"""Produce reviewable tables, examples and a student-verification worksheet."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from recommender import ap_at_5

LABELS={'popularity':'Popularity','repeat_h0':'Repeat (80%)','repeat_partial_t0.2':'Partial repeat (20%)','content':'Content TF-IDF','als_f32_a10':'Implicit ALS','association':'Association rules','improved_hybrid_w0.1':'Selected hybrid'}

def load_predictions(path):
    with path.open(newline='') as f:
        return {r['user_id']:r for r in csv.DictReader(f)}

def run(cache,out):
    res=json.loads((out/'validation_results.json').read_text());selected=json.loads((out/'selected_strategy.json').read_text())
    examples=json.loads((out/'examples.json').read_text());cat=json.loads((cache/'catalog.json').read_text())['validation']
    for r in res:
        if 'cold_start_users' in r:r['no_high_coverage_history_users']=r.pop('cold_start_users')
    if 'cold_start_users' in selected:selected['no_high_coverage_history_users']=selected.pop('cold_start_users')
    (out/'selected_strategy.json').write_text(json.dumps(selected,indent=2))
    (out/'validation_results.json').write_text(json.dumps(res,indent=2))
    with (out/'comparison.csv').open('w',newline='') as f:
        fields=['method','map5_primary','map5_all_positive','map5_all_template','fit_seconds','score_seconds','unique_recommended_ids','repeat_fraction']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(res)
    shutil.copy2(cache/'data_audit.json',out/'data_audit.json')
    lines=['# Local validation results','', 'Local selection estimates; no Kaggle score is claimed.','', '| Method | MAP@5 (1,150 users) | All positive users | All template users |','|---|---:|---:|---:|']
    report_names=list(LABELS)
    if selected['method'] not in report_names:report_names.append(selected['method'])
    byname={r['method']:r for r in res}
    for name in report_names:
        r=byname[name];lines.append(f"| {LABELS.get(name,name)} | {r['map5_primary']:.6f} | {r['map5_all_positive']:.6f} | {r['map5_all_template']:.6f} |")
    lines+=['','All '+str(len(res))+' trials are retained in validation_results.json.','', '# Two users, selected from training activity only','']
    enriched=[]
    def lookup(i,training=False):
        if not training and str(i) in cat['lookup']:return cat['lookup'][str(i)]
        md=cat['training_metadata'].get(str(i),{});titles=md.get('tv_show_title',[]);cats=md.get('tv_show_category',[])
        return {'title':min(titles,key=len) if titles else '(unavailable)','category':', '.join(cats) if cats else '(unavailable)'}
    for ex in examples:
        lines+=['## '+ex['label']+' user '+ex['user_id'],'',f"Training >=80% airings: {int(ex['training_high_coverage_airings'])}.",'','### Training favourites','']
        enriched.append({**ex,'training_top':[{**v,**lookup(v['id'],True)} for v in ex['training_top']],'methods':{}})
        for v in enriched[-1]['training_top']:lines.append(f"- {v['id']} - {v['title']} [{v['category']}]: {int(v['count'])} airings")
        for name in report_names:
            pred=load_predictions(out/(name+'_validation.csv'))[ex['user_id']]
            ids=list(map(int,pred['tv_show_id'].split()));truth=list(map(int,pred['truth'].split()))
            detail=[{'rank':rank,'id':i,**lookup(i),'hit':i in truth} for rank,i in enumerate(ids,1)]
            enriched[-1]['methods'][name]={'ap5':float(pred['ap5']),'recommendations':detail,'truth':[{'id':i,**lookup(i)} for i in truth]}
            lines+=['','### '+LABELS.get(name,name)+f" (AP@5={float(pred['ap5']):.6f})",'','| Rank | ID | Title | Category | Relevant |','|---:|---:|---|---|---|']
            for d in detail:lines.append(f"| {d['rank']} | {d['id']} | {d['title'].replace('|','/')} | {d['category']} | {'yes' if d['hit'] else 'no'} |")
        lines+=['','Validation target: '+', '.join(str(i)+' '+lookup(i)['title'] for i in ex['truth']), '']
    (out/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'examples_enriched.json').write_text(json.dumps(enriched,indent=2,ensure_ascii=False))
    # Paired diagnostic, not a fresh test or correction for model-selection multiplicity.
    best=load_predictions(out/(selected['method']+'_validation.csv'));base=load_predictions(out/'repeat_h0_validation.csv')
    # The main population can be recovered from target list lengths: ground_truth returns <=5,
    # so use full-count information from events to disambiguate ties/eligibility.
    from recommender import ground_truth
    events=dict(np.load(cache/'events.npz'));truth,counts=ground_truth(events,cat['cutoff'],cat['end'])
    keys=[u for u in best if len(counts.get(int(u),{}))>=5]
    delta=np.array([float(best[u]['ap5'])-float(base[u]['ap5']) for u in keys])
    rng=np.random.default_rng(42);boot=np.mean(delta[rng.integers(0,len(delta),(2000,len(delta)))],axis=1)
    paired={'versus':'repeat_h0','users':len(delta),'mean_delta_map5':float(delta.mean()),'improved_users':int((delta>1e-10).sum()),'worse_users':int((delta< -1e-10).sum()),'unchanged_users':int((np.abs(delta)<=1e-10).sum()),'descriptive_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'warning':'Same holdout used for selection; interval does not adjust for selection and is not independent generalization evidence.'}
    (out/'paired_diagnostic.json').write_text(json.dumps(paired,indent=2))
    heavy=next(x for x in enriched if x['label']=='heavy')['methods']['als_f32_a10']
    hit_ranks=[r['rank'] for r in heavy['recommendations'] if r['hit']]
    notes=['# Personal verification worksheet','', 'These checks were executed by the AI. Complete them yourself before claiming personal verification in the report.','', '1. Inspect `results/join_verification.json`: choose a raw user/channel interval and its airing. Recompute the intersection and coverage. Confirm the numerator is watched seconds and the denominator is programme duration, not the viewing-log duraton field.','', '2. For the heavy example user, inspect the Implicit ALS recommendation table in results/analysis.md. Relevant hits occur at ranks '+', '.join(map(str,hit_ranks))+'. Compute AP@5 as the sum of precision at each hit, divided by 5. Record your calculation in your own words.','', '3. Explain why a programme watched before may be recommended again in this competition.','', '4. Compare one failure: a familiar training favourite absent from the validation Top-5, or a plausible content recommendation with no validation hit. Distinguish “not in the target Top-5” from “never watched” or “disliked”.','', '5. Inspect all five IDs of the Kaggle CSV for one user, and verify the programme titles in the test schedule.','', '6. Add your actual Kaggle public/private screenshot, GitHub URL, and original native AI-session export before final submission. Do not edit or reconstruct the session log.']
    (out.parent/'PERSONAL_VERIFICATION.md').write_text('\n'.join(notes),encoding='utf-8')
    print(json.dumps({'selected':selected['method'],'MAP5':selected['map5_primary'],'paired':paired},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--results-dir',type=Path,required=True);a=p.parse_args();run(a.cache_dir,a.results_dir)
