#!/usr/bin/env python3
"""Portable packaging of the frozen Stage 13 strategy, added during report audit.

No parameter search. Imports the original OpenCode implementations, rebuilds
the full-history join from the five official CSVs, and requires the scored hash.
"""
from pathlib import Path
import argparse, dataclasses, hashlib, importlib.util, json, math, sys, time
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'opencode_run/src'))
from stage2.config import load_config
from stage5.history import (build_history_schedule,build_test_candidates,build_history_views,
    build_history_interactions,read_count_variants,build_popularity,read_template_users)
from stage5.predict import fit_predict_hybrid
from stage5.submission import write_submission,validate_submission
from stage4.ranking import top_k

def module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/'opencode_run'/relative)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def normalize(d):
    maximum=max(d.values(),default=0.)
    return {j:v/maximum for j,v in d.items() if v>0} if maximum>0 else {}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data-dir',type=Path,required=True)
    a.add_argument('--cache-dir',type=Path,default=ROOT/'reproduction_cache')
    a.add_argument('--reuse-preprocessing',action='store_true',help='Use already rebuilt Stage 5 compatible preprocessing in cache-dir')
    a.add_argument('--output',type=Path,default=ROOT/'reproduced_stage13.csv')
    args=a.parse_args();start=time.perf_counter()
    if args.output.exists(): raise FileExistsError('Refuse to overwrite output')
    cfg=load_config(ROOT/'opencode_run/config/stage5.json'); data=args.data_dir.resolve()
    cfg=dataclasses.replace(cfg,data_dir=data,output_dir=args.cache_dir.resolve(),view_file=data/cfg.view_file.name,
        submission_file=data/cfg.submission_file.name,schedule_files={k:data/v.name for k,v in cfg.schedule_files.items()})
    cfg.ensure_dirs()
    print('Reconstructing owned and deduplicated schedule...',flush=True)
    history=build_history_schedule(cfg);target=build_test_candidates(cfg);candidates=target['candidates']
    assert len(candidates)==6632
    if not args.reuse_preprocessing:
        print('Reconstructing user/channel view intervals and unioned coverage...',flush=True)
        views=build_history_views(cfg);build_history_interactions(cfg,history['airings'],views)
    counts=read_count_variants(cfg.out('interactions','history_user_program_counts.csv'))
    ids=json.loads(cfg.out('views','id_maps_history.json').read_text())['history']['user_ids']
    users=read_template_users(cfg);mapping={u:i for i,u in enumerate(ids)}
    pop,stats,_=build_popularity(counts,candidates)
    h=Counter(int(r.show_id) for r in history['airings'].rows);t=Counter(int(r.show_id) for r in target['airings'].rows)
    tw=(cfg.windows['train'].end-cfg.windows['train'].start)/(7*86400)
    vw=(cfg.windows['test'].end-cfg.windows['test'].start)/(7*86400)
    factors={j:math.sqrt(min(4.,max(.25,(t[j]/vw)/(h[j]/tw)))) for j in candidates if h[j]>0}
    assert tw==20 and vw==12
    print('Fitting frozen repeat, AR and weighted EASE components...',flush=True)
    base=fit_predict_hybrid(users,counts,mapping,candidates,pop,stats)
    count=module('frozen_count','outputs/stage12/code/count_input.py');ease=module('frozen_ease','outputs/stage11/code/ease_model.py')
    X,lu,li,input_stats=count.transformed_matrix(counts['primary80'])
    B,fit_stats=ease.fit_from_gram(count.gram_matrix(X),1000)
    _,raw,score_stats=ease.candidate_scores(X,B,lu,li,users,mapping,candidates)
    predictions={}
    for i,u in enumerate(users):
        r=normalize({j:v*factors[j] for j,v in base['repeat_scores'][u].items() if v>0})
        ar=normalize(base['association_scores'][u]);s9={j:.7*v for j,v in r.items()}
        for j,v in ar.items():s9[j]=s9.get(j,0)+.3*v
        es=normalize({j:float(v)*factors[j] for j,v in zip(candidates,raw[i]) if v>0})
        scores={j:.9*v for j,v in s9.items() if v>0}
        for j,v in es.items():scores[j]=scores.get(j,0)+.1*v
        predictions[u]=top_k(scores,candidates,pop,stats)[0]
    write_submission(args.output,users,predictions)
    checks=validate_submission(args.output,users,candidates);assert checks['all_passed']
    actual=digest(args.output);expected='628edd64a9c4ecc26df1f0c2174d7378db4c15ebc4f01615a51f841ff44779d4'
    assert actual==expected,(actual,expected)
    result={'exact_scored_csv_reproduced':True,'sha256':actual,'submission_checks':checks,
        'normalization_scalar':input_stats['normalization_scalar'],'seconds':time.perf_counter()-start,
        'preprocessing_reused':args.reuse_preprocessing,'scope':'raw preprocessing, schedules, full model refit and CSV' if not args.reuse_preprocessing else 'reconstructed schedules, cached joins, full model refit and CSV',
        'implementation':'original OpenCode modules; report-audit packaging wrapper, not a new experiment'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
