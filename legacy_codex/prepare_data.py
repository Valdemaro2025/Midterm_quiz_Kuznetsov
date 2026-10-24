"""Audit source files and join unioned viewing intervals to labelled airings.

Run: python prepare_data.py --data-dir PATH --cache-dir PATH
Inputs are never changed. All timestamps are treated as supplied local wall time;
no unsupported timezone conversion is applied.
"""
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd

SCHEDULES = ['export_arh_11-20-final.csv', 'export_arh_21-30-final.csv', 'export_arh_31-42-final.csv']
FIELDS = ['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']

def seconds(s, fmt):
    return pd.to_datetime(s, format=fmt).to_numpy(dtype='datetime64[s]').astype('int64')

def union_intervals(users, starts, ends):
    """Merge overlapping/adjacent intervals within ONE channel and user."""
    order = np.lexsort((ends, starts, users))
    u,s,e = users[order],starts[order],ends[order]
    ou,os,oe = [],[],[]
    for a,b,c in zip(u,s,e):
        if ou and a == ou[-1] and b <= oe[-1]:
            oe[-1] = max(oe[-1],c)
        else:
            ou.append(a);os.append(b);oe.append(c)
    return np.array(ou,dtype=np.uint64),np.array(os,dtype=np.int64),np.array(oe,dtype=np.int64)

def overlap_pairs(vs,ve,ss,se):
    """All positive interval overlaps; handles nested schedule intervals too."""
    lo=np.searchsorted(np.maximum.accumulate(se),vs,side='right')
    hi=np.searchsorted(ss,ve,side='left')
    count=np.maximum(hi-lo,0)
    ix=np.repeat(np.arange(len(vs)),count)
    if not len(ix):return ix,ix,ix
    starts=np.repeat(np.cumsum(count)-count,count)
    j=np.repeat(lo,count)+np.arange(len(ix))-starts
    duration=np.minimum(ve[ix],se[j])-np.maximum(vs[ix],ss[j])
    ok=duration>0
    return ix[ok],j[ok],duration[ok]

def run(data_dir,cache_dir):
    begin=time.perf_counter();cache_dir.mkdir(parents=True,exist_ok=True)
    audit={'environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__},'files':{}}
    schedules=[]
    for file in SCHEDULES:
        n=labelled=invalid=0;frames=[];allmin=2**63-1;allmax=0;channels=set()
        for c in pd.read_csv(data_dir/file,chunksize=200000,keep_default_na=False,dtype={'tv_show_id':'int64','channel_id':'int32'}):
            n+=len(c);channels.update(c.channel_id);st=seconds(c.start_time,'%d.%m.%Y %H:%M:%S');en=st+c.duration.to_numpy(dtype=np.int64)
            allmin=min(allmin,int(st.min()));allmax=max(allmax,int(en.max()));invalid+=int((en<=st).sum())
            mask=(c.tv_show_id>0).to_numpy()&(en>st)
            labelled+=int(mask.sum());q=c.loc[mask,['channel_id','tv_show_id']+FIELDS].copy();q['start']=st[mask];q['end']=en[mask];frames.append(q)
        z=pd.concat(frames,ignore_index=True);z['source']=file;schedules.append(z)
        audit['files'][file]={'rows':n,'labelled_valid_rows':labelled,'unique_positive_ids':int(z.tv_show_id.nunique()),'channels':len(channels),'nonpositive_duration':invalid,'min_start':str(np.datetime64(allmin,'s')),'max_end':str(np.datetime64(allmax,'s'))}
        print(file,audit['files'][file],flush=True)
    sched=pd.concat(schedules,ignore_index=True)
    may18 = int(np.datetime64('2020-05-18T00:00:00','s').astype('int64'))
    july27 = int(np.datetime64('2020-07-27T00:00:00','s').astype('int64'))
    # Files contain broadcast-day spillover. The later file owns airings whose
    # START falls in its period; do not join two versions of the boundary grid.
    owned = (((sched.source == SCHEDULES[0]) & (sched.start < may18)) |
             ((sched.source == SCHEDULES[1]) & (sched.start >= may18) & (sched.start < july27)) |
             ((sched.source == SCHEDULES[2]) & (sched.start >= july27)))
    audit['schedule_spillover_rows_excluded'] = int((~owned).sum())
    sched=sched[owned].copy()
    # Same channel, timestamp, duration, ID identifies one broadcast, regardless of file.
    before=len(sched);sched=sched.drop_duplicates(['channel_id','start','end','tv_show_id']).sort_values(['channel_id','start','end','tv_show_id']).reset_index(drop=True)
    sched['airing']=np.arange(len(sched),dtype=np.int32)
    audit['duplicate_labelled_airings_removed']=before-len(sched)
    key=sched.groupby(['channel_id','start','end']).tv_show_id.nunique()
    audit['same_interval_multiple_labelled_ids']=int((key>1).sum())
    sched.to_pickle(cache_dir/'schedule.pkl')
    # Metadata for report lookups, retained per source to prevent future-attribute leakage.
    sched[['tv_show_id','source']+FIELDS].drop_duplicates().to_csv(cache_dir/'metadata.csv',index=False)
    with (data_dir/'submission.csv').open(newline='') as f:
        template=list(csv.DictReader(f))
    users_test=[r['user_id'] for r in template]
    assert len(users_test)==len(set(users_test))
    (cache_dir/'test_users.json').write_text(json.dumps(users_test))
    views={};n=bad=0;ud=set();durations=[];time_min=2**63-1;time_max=0
    for c in pd.read_csv(data_dir/'dataset11-30.csv',chunksize=250000,dtype={'user_id':'uint64','vsetv_id':'int32'}):
        st=seconds(c.start_time,'%Y-%m-%d %H:%M:%S');en=seconds(c.stop_time,'%Y-%m-%d %H:%M:%S')
        bad+=int(((en-st)!=c.duraton.to_numpy()).sum());n+=len(c);ud.update(map(int,c.user_id));durations.append(en-st)
        time_min=min(time_min,int(st.min()));time_max=max(time_max,int(en.max()))
        for channel,ind in c.groupby('vsetv_id').indices.items():
            views.setdefault(int(channel),[]).append((c.user_id.to_numpy()[ind],st[ind],en[ind]))
    audit['views']={'rows':n,'users':len(ud),'duration_mismatches':bad,'min_start':str(np.datetime64(time_min,'s')),'max_end':str(np.datetime64(time_max,'s')),'duration_quantiles':dict(zip(['min','p25','p50','p75','p90','p95','p99','max'],np.quantile(np.concatenate(durations),[0,.25,.5,.75,.9,.95,.99,1]).tolist()))}
    del durations
    audit['test_users']=len(users_test);audit['test_users_absent_from_entire_log']=sum(int(u) not in ud for u in users_test)
    parts=[];union_count=0;labelled_union_seconds=0;all_union_seconds=0;pair_count=0
    for channel,chunks in sorted(views.items()):
        arrays=[np.concatenate([x[k] for x in chunks]) for k in range(3)]
        vu,vs,ve=union_intervals(*arrays);union_count+=len(vu);all_union_seconds+=int((ve-vs).sum())
        s=sched[sched.channel_id==channel]
        if s.empty:continue
        ss=s.start.to_numpy();se=s.end.to_numpy();vi,sj,overlap=overlap_pairs(vs,ve,ss,se)
        pair_count+=len(vi);labelled_union_seconds+=int(overlap.sum())
        if len(vi):
            p=pd.DataFrame({'user_id':vu[vi],'airing':s.airing.to_numpy()[sj],'watched_seconds':overlap})
            p=p.groupby(['user_id','airing'],as_index=False).watched_seconds.sum();parts.append(p)
    events=pd.concat(parts,ignore_index=True)
    events=events.merge(sched[['airing','tv_show_id','start','end']],on='airing',how='left',validate='many_to_one')
    events['coverage']=events.watched_seconds/(events.end-events.start)
    audit['join']={'unioned_user_channel_intervals':union_count,'overlap_pairs':pair_count,'user_airing_pairs':len(events),'coverage_above_one':int((events.coverage>1+1e-10).sum()),'high_coverage_airings':int((events.coverage>=.8).sum()),'all_user_channel_union_seconds':all_union_seconds,'sum_labelled_overlap_seconds':labelled_union_seconds,'note':'Labelled overlap sum is not a unique-time coverage ratio if schedule airings overlap.'}
    events.to_csv(cache_dir/'events.csv',index=False)
    for file in ['dataset11-30.csv']+SCHEDULES+['submission.csv']:
        h=hashlib.sha256()
        with (data_dir/file).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        audit['files'].setdefault(file,{})['sha256']=h.hexdigest()
    export_model_inputs(sched, events, cache_dir)
    audit['elapsed_seconds']=time.perf_counter()-begin
    (cache_dir/'data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print(json.dumps(audit,indent=2,ensure_ascii=False),flush=True)


def export_model_inputs(sched, events, cache_dir):
    windows = {
        'validation': ('2020-05-18T00:00:00', '2020-07-27T00:00:00'),
        'test': ('2020-07-27T00:00:00', '2020-10-19T00:00:00'),
    }
    catalog = {}
    for name, (cut_text, end_text) in windows.items():
        cut = int(np.datetime64(cut_text, 's').astype('int64'))
        end = int(np.datetime64(end_text, 's').astype('int64'))
        train = sched[sched.end <= cut]
        future = sched[sched.source == (SCHEDULES[1] if name == 'validation' else SCHEDULES[2])]
        metadata = {}
        for item, rows in train.groupby('tv_show_id'):
            metadata[str(item)] = {field: sorted(set(map(str, rows[field])) - {''}) for field in FIELDS}
        lookup = {str(item): {'title': rows.tv_show_title.mode().iloc[0],
                              'category': rows.tv_show_category.mode().iloc[0]}
                  for item, rows in future.groupby('tv_show_id')}
        catalog[name] = {'cutoff': cut, 'end': end,
                         'candidates': sorted(map(int, future.tv_show_id.unique())),
                         'training_metadata': metadata, 'lookup': lookup,
                         'training_airings': {str(i): int(n) for i,n in train.groupby('tv_show_id').size().items()},
                         'future_airings': {str(i): int(n) for i,n in future.groupby('tv_show_id').size().items()}}
    (cache_dir / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False))
    np.savez_compressed(cache_dir / 'events.npz',
        user_id=events.user_id.to_numpy(), item=events.tv_show_id.to_numpy(),
        start=events.start.to_numpy(), end=events.end.to_numpy(),
        coverage=events.coverage.to_numpy(), seconds=events.watched_seconds.to_numpy(),
        airing=events.airing.to_numpy())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data-dir',type=Path,required=True);a.add_argument('--cache-dir',type=Path,required=True);args=a.parse_args();run(args.data_dir,args.cache_dir)
