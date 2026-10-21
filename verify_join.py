"""Independent raw-CSV audit of 64 joined user-airing examples.
Uses a straightforward clipping-and-union oracle, not the vectorized join code.
"""
import argparse,csv,json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def verify(data,cache,out):
    e=pd.read_csv(cache/'events.csv',dtype={'user_id':'uint64'})
    s=pd.read_pickle(cache/'schedule.pkl').set_index('airing')
    sample=e.sample(n=64,random_state=2026).copy()
    sample['channel_id']=[int(s.loc[a,'channel_id']) for a in sample.airing]
    keys={(str(row.user_id),str(row.channel_id)) for row in sample.itertuples()}
    logs={k:[] for k in keys};mismatch_count=0;max_difference=0
    with (data/'dataset11-30.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            key=row['user_id'],row['vsetv_id']
            if key in logs:
                start=datetime.fromisoformat(row['start_time']);end=datetime.fromisoformat(row['stop_time'])
                # Calendar arithmetic, avoiding host-timezone timestamps.
                epoch=datetime(1970,1,1)
                logs[key].append((int((start-epoch).total_seconds()),int((end-epoch).total_seconds())))
    actual=[]
    for row in sample.itertuples():
        intervals=[]
        for a,b in logs[(str(row.user_id),str(row.channel_id))]:
            left=max(a,row.start);right=min(b,row.end)
            if right>left:intervals.append((left,right))
        intervals.sort();merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
            else:merged.append((a,b))
        seconds=sum(b-a for a,b in merged)
        assert seconds==row.watched_seconds,(row.user_id,seconds,row.watched_seconds)
        actual.append({'user_id':str(row.user_id),'airing':int(row.airing),'program_id':int(row.tv_show_id),'channel_id':int(row.channel_id),'start':str(np.datetime64(int(row.start),'s')),'duration':int(row.end-row.start),'oracle_seconds':seconds,'pipeline_seconds':int(row.watched_seconds),'coverage':row.coverage})
    report={'random_seed':2026,'sample_size':64,'mismatches':0,'oracle':'read original viewing CSV; clip all intervals to each sampled airing; independently union and sum','cases':actual}
    out.write_text(json.dumps(report,indent=2));print('64 raw-CSV overlap checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.data_dir,a.cache_dir,a.output)
