"""Evaluate fixed hybrids only when a new outcome week passes coverage screening."""
import argparse
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from asg2.core import load_tables, build_events, stamp, TARGETS, features, labels, score
from asg2.dynamic_relations import dynamic_features
from asg2.hybrid import apply_hybrid, last_week
from asg2.run import predict, fingerprint


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    target=stamp(a.origin);end=target+pd.Timedelta(days=7)
    tables=load_tables(a.data);events=build_events(tables)
    evidence=dict(origin_utc=str(target),outcome_end_utc=str(end),
                  latest_available_utc=str(events.available_at.max()),data_manifest=fingerprint(a.data))
    # Do not report an optimistic score from incomplete outcome reconstruction.
    daily=events[(events.event_time>=target)&(events.event_time<end)&(events.available_at<end)].assign(
        day=lambda d:(d.event_time-target).dt.days).groupby(['day','kind']).size().unstack(fill_value=0)
    daily=daily.reindex(index=range(7),columns=TARGETS,fill_value=0).fillna(0)
    evidence['zero_record_days']={k:int((daily[k]==0).sum()) for k in TARGETS}
    if events.available_at.max()<end or (daily==0).any().any():
        evidence.update(status='PENDING',reason='Outcome week unfinished or target coverage screen fails; no score generated')
    else:
        configs=json.loads(Path(a.selection).read_text())['configs']
        X=features(tables,events,target);X=dynamic_features(tables,events,X,target)
        y=labels(tables,events,target,X.UserID);rows=[]
        for name,config in configs.items():
            with (Path(a.source)/config['checkpoint']).open('rb') as f:b=pickle.load(f)
            p=apply_hybrid(predict(b['models'],X,b['columns']),last_week(X),config)
            rows.append(dict(candidate=name,**score(y,p)))
        rows.append(dict(candidate='last_week_rule',**score(y,last_week(X))))
        with (Path(a.source)/'model_temporal.pkl').open('rb') as f:b=pickle.load(f)
        rows.append(dict(candidate='original_temporal',**score(y,predict(b['models'],X,b['columns'])>=b['thresholds'])))
        evidence.update(status='SCORED',n_users=len(X),scores=rows,
            warning='A zero-day screen does not establish complete collection. Fixed model weights and decision policies were not tuned on this week.')
    (out/'forward_status.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True)
    p.add_argument('--origin',default='2026-10-29T00:00:00+08:00')
    p.add_argument('--source',default='artifacts_coverage_20261102')
    p.add_argument('--selection',default='artifacts_hybrid_20261102/selection.json')
    p.add_argument('--out',default='artifacts_hybrid_20261102')
    run(p.parse_args())
