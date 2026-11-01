"""Evaluate fixed candidates; do not tune on diagnostic or new-week outcomes."""
import argparse, importlib.util, json, pickle, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *
from asg2.run import predict

def run(data, out):
    data, out = Path(data), Path(out)
    tables=load_tables(data); events=build_events(tables)
    protocol=json.loads((out/'protocol.json').read_text())
    module=data/'evaluation/evaluate.py'
    spec=importlib.util.spec_from_file_location('official',module)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    candidates={}
    for variant in ['static','temporal','relational','full']:
        with (out/f'model_{variant}.pkl').open('rb') as f:candidates[variant]=pickle.load(f)
    design=dict(origins_taiwan=['2026-10-08','2026-10-15','2026-10-22','2026-10-29'],
        models=['all_zero','all_one','last_week_rule','frequency90_at_0.5',*candidates],
        populations=['recent90_proxy','selection_lag14d'],
        no_tuning=True, thresholds_from='2026-09-24 calibration only',
        note='Latest snapshot refits, not preserved October checkpoints. New-week data coverage may be incomplete.')
    # Save protocol before exposing any new-week scores.
    (out/'recheck_design.json').write_text(json.dumps(design,indent=2))
    rows=[];pending=[]; coverage=[]
    for date in design['origins_taiwan']:
        origin=stamp(date+'T00:00:00+08:00'); end=origin+pd.Timedelta(days=7)
        if end>events.available_at.max():
            pending.append(dict(origin_taiwan=date,status='PENDING: seven-day outcome not complete'))
            continue
        start=origin-pd.Timedelta(hours=4)
        # The official roster is selected 14d4h before its forecast origin.
        for population in design['populations']:
            ids=None
            if population=='selection_lag14d':
                ids=features(tables,events,origin-pd.Timedelta(days=14),gap_hours=4).UserID
            X=features(tables,events,origin,user_ids=ids,gap_hours=4)
            y=labels(tables,events,origin,X.UserID)
            observed=asof_events(events,start)
            recent=observed[observed.event_time>=start-pd.Timedelta(days=90)].copy()
            recent['week']=((start-recent.event_time).dt.total_seconds()//(7*86400)).astype(int)
            recent=recent[recent.week<12]
            freq=recent.drop_duplicates(['user_id','kind','week']).groupby(['user_id','kind']).size().unstack(fill_value=0)
            user_index=tables['Users'].set_index('github_user_id').loc[X.UserID].user_id
            freq=freq.reindex(index=user_index,columns=TARGETS,fill_value=0).fillna(0).to_numpy()/12
            predictions={'all_zero':np.zeros_like(y),'all_one':np.ones_like(y),
                'last_week_rule':np.column_stack([X[f't_count7_{k}'].to_numpy()>0 for k in TARGETS]),
                'frequency90_at_0.5':freq>=.5}
            for name,c in candidates.items():
                predictions[name]=predict(c['models'],X,c['columns'])>=c['thresholds']
            truth=pd.DataFrame(y,columns=TARGETS);truth.insert(0,'UserID',X.UserID.to_numpy())
            for name,p in predictions.items():
                df=pd.DataFrame(p.astype(int),columns=TARGETS);df.insert(0,'UserID',X.UserID.to_numpy())
                dest=out/f'holdout_predictions_{date}_{population}_{name}.csv';df.to_csv(dest,index=False)
                official.read_predictions(dest,expected_users=X.UserID)
                ours=score(y,p);theirs=official.scores(df,truth)
                for metric in theirs:assert abs(ours[metric]-theirs[metric])<1e-12
                rows.append(dict(origin_taiwan=date,population=population,model=name,n_users=len(X),
                    status='PROVISIONAL: supplied records only; inspect daily_event_counts.csv',**ours))
            for j,k in enumerate(TARGETS):coverage.append(dict(origin_taiwan=date,population=population,
                target=k,positive_users=int(y[:,j].sum()),positive_rate=float(y[:,j].mean())))
    pd.DataFrame(rows).to_csv(out/'recheck_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'label_prevalence.csv',index=False)
    (out/'pending.json').write_text(json.dumps(pending,indent=2))
    print(pd.DataFrame(rows)[['origin_taiwan','population','model','score']].to_string(index=False))
    print('Pending:',pending)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.data,a.out)
