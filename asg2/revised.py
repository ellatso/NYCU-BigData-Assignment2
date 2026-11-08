"""November 7 revision: new official provenance, reproducible ablations and Nov22 inference."""
import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMClassifier
from . import core
from .core import TARGETS,stamp,load_tables,build_events,labels,score
from .dynamic_relations import dynamic_features
from .coverage import tune_mean_thresholds,score_subset,complete_training_origins
from .provenance import ROOT,VENDOR,verify_data,verify_vendor,sha256,integrity
from .run import Constant

VARIANTS=['observed_history','recent7','temporal30','no_recency','no_trend',
          'static_relations','temporal_static_relations','received_relations',
          'dynamic_relations','no_peer_relations','availability_clock']

def read_data(path,fixture=False):
    if fixture:
        meta=json.loads((Path(path)/'provenance.json').read_text())
        if meta.get('dataset_kind')!='synthetic_fixture':raise ValueError('Fixture mode accepts only labeled fixtures')
    else:meta=verify_data(path)
    return load_tables(path),meta

def make_frame(tables,events,origin,ids=None,gap=4,availability_clock=False):
    """Unknown or not-yet-seen IDs get a separate cold-start prior at inference."""
    origin=stamp(origin);cutoff=origin-pd.Timedelta(hours=gap)
    u=tables['Users'];known=u.loc[u.is_cohort & u.first_seen_at.lt(cutoff),'github_user_id'].to_numpy(dtype='int64')
    requested=known if ids is None else np.asarray(ids,dtype='int64')
    if len(np.unique(requested))!=len(requested):raise ValueError('Duplicate requested ID')
    valid=requested[np.isin(requested,known)]
    if not len(valid):return pd.DataFrame({'UserID':valid,'origin':pd.Series([],dtype='datetime64[ns]')})
    e=events.copy() if availability_clock else events
    if availability_clock:e['event_time']=e.available_at
    X=core.features(tables,e,origin,user_ids=valid,gap_hours=gap)
    # Short relationship windows and all FK endpoints remain visibility-gated.
    return dynamic_features(tables,e,X,origin-pd.Timedelta(hours=gap-4))

def columns(X,variant):
    user=[c for c in X if c.startswith('u_')]
    temporal=[c for c in X if c.startswith('t_') and not ('count90_' in c or c.endswith('days90') or c.endswith('repos90'))]
    if variant=='observed_history':return user
    if variant=='recent7':return user+[c for c in temporal if c.startswith('t_count7_') or c in ['t_active_days7','t_repos7']]
    if variant=='no_recency':temporal=[c for c in temporal if not c.startswith('t_recency_')]
    if variant=='no_trend':temporal=[c for c in temporal if not c.startswith('t_trend_')]
    if variant=='static_relations':return user+[c for c in X if c.startswith('r_static_')]
    result=user+temporal
    if variant=='temporal_static_relations':result += [c for c in X if c.startswith('r_static_')]
    if variant=='received_relations':result += [c for c in X if c.startswith('d_received_')]
    if variant in ['dynamic_relations','no_peer_relations']:
        result += [c for c in X if c.startswith('d_') and (variant!='no_peer_relations' or not c.startswith(('d_peer','d_cohort')))]
    return result

def fit(X,y,cols,trees=160,seed=42):
    heads=[]
    for j in range(6):
        yy=y[:,j]
        if len(np.unique(yy))<2:heads.append(Constant(float(yy.mean())));continue
        model=LGBMClassifier(n_estimators=trees,learning_rate=.045,num_leaves=23,min_child_samples=30,
            reg_lambda=4,random_state=seed,n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(X.reindex(columns=cols,fill_value=0),yy);heads.append(model)
    return heads

def probability(heads,X,cols):
    return np.column_stack([h.predict_proba(X.reindex(columns=cols,fill_value=0))[:,1] for h in heads])

def local_dates(values):return [stamp(v+'T00:00:00+08:00') for v in values]

def raw_bounds(meta):
    files=meta['raw_manifest']['files'];valid=[stamp(r['hour_utc']) for r in files.values() if r.get('status')=='ok']
    if not valid:raise ValueError('No source hours')
    return min(valid),max(valid)+pd.Timedelta(hours=1)

def hour_coverage(meta,start,end):
    expected=pd.date_range(stamp(start).floor('h'),stamp(end).ceil('h'),freq='h',inclusive='left')
    valid={stamp(v['hour_utc']) for v in meta['raw_manifest']['files'].values() if v.get('status')=='ok'}
    return dict(expected_hours=len(expected),available_hours=sum(t in valid for t in expected))

def save_json(path,data):Path(path).write_text(json.dumps(data,indent=2,default=str)+'\n')

def experiment(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(a.config).read_text());tables,meta=read_data(a.data,a.fixture_only);events=build_events(tables)
    raw_start,raw_end=raw_bounds(meta)
    groups={k:local_dates(config[k+'_origins_taiwan']) for k in ['train','calibration','evaluation']}
    assert max(groups['train'])+pd.Timedelta(days=7)<=min(groups['calibration'])-pd.Timedelta(hours=config['observation_gap_hours'])
    assert max(groups['calibration'])+pd.Timedelta(days=7)<=min(groups['evaluation'])-pd.Timedelta(hours=config['observation_gap_hours'])
    for origin in sum(groups.values(),[]):
        if origin+pd.Timedelta(days=7)>raw_end:raise ValueError('An outcome week is later than the downloaded source hours')
    scope='synthetic_fixture_only' if a.fixture_only else 'observed_official_gharchive'
    design=dict(scope=scope,config=config,source_period_utc=[str(raw_start),str(raw_end)],
        variants=VARIANTS,labels='Seven-day observed occurrences, available by outcome end; incomplete GH Archive is not complete truth',
        selection='Primary relational recipe fixed in config before real execution; thresholds use mean weekly calibration Score. Best calibration reference is reported separately.',
        grading='F1 has zero grade weight in the Nov7 revision. Ablations are explanatory evidence.')
    save_json(out/'design.json',design)
    cache={};coverage=[]
    for origin in sum(groups.values(),[]):
        X=make_frame(tables,events,origin,gap=config['observation_gap_hours'])
        if not len(X):raise ValueError('No observed cohort users at origin '+str(origin))
        y=labels(tables,events,origin,X.UserID);cache[origin]=(X,y)
        coverage.append(dict(origin_utc=str(origin),users=len(X),**hour_coverage(meta,origin,origin+pd.Timedelta(days=7)),
            prior_observed_days=float((origin-pd.Timedelta(hours=config['observation_gap_hours'])-raw_start).total_seconds()/86400)))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    train=[cache[t] for t in groups['train']];cal=[cache[t] for t in groups['calibration']]
    X=pd.concat([z[0] for z in train],ignore_index=True);y=np.concatenate([z[1] for z in train])
    C=pd.concat([z[0] for z in cal],ignore_index=True);cy=np.concatenate([z[1] for z in cal])
    folds=np.concatenate([np.repeat(i,len(z[0])) for i,z in enumerate(cal)])
    metrics=[];models={};best=None
    for variant in VARIANTS:
        A,B=X,C
        if variant=='availability_clock':
            aa=[make_frame(tables,events,t,gap=config['observation_gap_hours'],availability_clock=True) for t in groups['train']]
            bb=[make_frame(tables,events,t,gap=config['observation_gap_hours'],availability_clock=True) for t in groups['calibration']]
            A=pd.concat(aa,ignore_index=True);B=pd.concat(bb,ignore_index=True)
        cols=columns(A,variant);heads=fit(A,y,cols,config['trees'],config['seed']);p=probability(heads,B,cols)
        threshold=tune_mean_thresholds(cy,p,folds)
        cal_score=float(np.mean([score(cy[folds==i],p[folds==i]>=threshold)['score'] for i in np.unique(folds)]))
        bundle=dict(models=heads,columns=cols,thresholds=threshold,variant=variant,
                    calibration_score=cal_score,prior=cy.mean(0),config=config,scope=scope)
        models[variant]=bundle;metrics.append(dict(model=variant,split='calibration_mean',score=cal_score,n_features=len(cols)))
        if best is None or (cal_score,-len(cols))>(models[best]['calibration_score'],-len(models[best]['columns'])):best=variant
        print('CAL',variant,cal_score,flush=True)
    chosen=config.get('submission_variant','dynamic_relations')
    if chosen not in models:raise ValueError('Unknown configured submission variant')
    save_json(out/'selection.json',dict(selected=chosen,recipe_fixed_before_real_execution=True,
        best_calibration_reference=best,scope=scope,calibration_scores={k:v['calibration_score'] for k,v in models.items()}))
    for name,b in models.items():
        yy=[];pp=[]
        for t in groups['evaluation']:
            D,dy=cache[t]
            if name=='availability_clock':D=make_frame(tables,events,t,gap=config['observation_gap_hours'],availability_clock=True)
            pred=probability(b['models'],D,b['columns'])>=b['thresholds'];yy.append(dy);pp.append(pred)
            metrics.append(dict(model=name,split=str(t),**score(dy,pred)))
        metrics.append(dict(model=name,split='pooled_observed_development',n_features=len(b['columns']),**score(np.concatenate(yy),np.concatenate(pp))))
    yy=[];pp=[]
    for t in groups['evaluation']:
        D,dy=cache[t];pred=np.column_stack([D['t_count7_'+k].to_numpy()>0 for k in TARGETS]);yy.append(dy);pp.append(pred)
        metrics.append(dict(model='last_week_rule',split=str(t),**score(dy,pred)))
    metrics.append(dict(model='last_week_rule',split='pooled_observed_development',**score(np.concatenate(yy),np.concatenate(pp))))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    # Coverage and delay distributions make the temporal assumptions inspectable.
    delay=(events.available_at-events.event_time).dt.total_seconds()/3600
    save_json(out/'time_diagnostics.json',dict(visible_event_rows=len(events),delayed_over_24h=int((delay>24).sum()),
        max_delay_hours=float(delay.max()) if len(delay) else None,warning='Observed archive delays and missing source hours are different limitations.'))
    save_json(out/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lightgbm.__version__,seed=config['seed'],threads=1))
    with (out/'model.pkl').open('wb') as f:pickle.dump(models[chosen],f)
    save_json(out/'run_manifest.json',dict(scope=scope,data_provenance_sha256=sha256(Path(a.data)/'provenance.json'),config=config,
        source_sha256={p:sha256(ROOT/p) for p in ['asg2/core.py','asg2/dynamic_relations.py','asg2/revised.py','asg2/coverage.py','asg2/run.py','asg2/provenance.py']},official_pin=verify_vendor()))
    print('Fixed primary',chosen,'calibration reference',best,'scope',scope,flush=True)

def predict_file(a):
    out=Path(a.out);tables,meta=read_data(a.data,a.fixture_only);events=build_events(tables)
    with (out/'model.pkl').open('rb') as f:b=pickle.load(f)
    manifest=json.loads((out/'run_manifest.json').read_text())
    if manifest['data_provenance_sha256']!=sha256(Path(a.data)/'provenance.json'):
        raise ValueError('Prediction data differ from this experiment snapshot: rerun experiment on the new prepared data first')
    if b['scope']!=('synthetic_fixture_only' if a.fixture_only else 'observed_official_gharchive'):raise ValueError('Checkpoint scope mismatch')
    for p,want in manifest['source_sha256'].items():
        if sha256(ROOT/p)!=want:raise ValueError('Model feature implementation changed: retrain before prediction')
    roster=Path(a.input) if a.input else VENDOR/'submission/test_input.csv'
    ids=pd.read_csv(roster).UserID.to_numpy(dtype='int64')
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate input IDs')
    if not a.fixture_only:
        official_ids=set(pd.read_csv(VENDOR/'submission/test_input.csv').UserID)
        if set(ids)!=official_ids:raise ValueError('Input differs from the pinned official test roster; review and update the official source pin before changing test users')
    target=stamp(a.target_start);_,raw_end=raw_bounds(meta)
    cutoff=min(stamp(a.cutoff),target,raw_end) if a.cutoff else min(target,raw_end)
    if max(local_dates(b['config']['calibration_origins_taiwan']))+pd.Timedelta(days=7)>cutoff:
        raise ValueError('The checkpoint calibration outcomes finish after the requested prediction cutoff')
    # A user-specified forecast cutoff is a real observed boundary, not a future
    # freshness fiction. Future archive rows are removed by make_frame.
    inference_origin=cutoff+pd.Timedelta(hours=4)
    cols=b['columns'];heads=b['models'];refit_origins=[]
    if a.refit:
        start=local_dates([b['config']['train_origins_taiwan'][0]])[0]
        refit_origins=complete_training_origins(start,cutoff)
        frames=[];ys=[]
        for t in refit_origins:
            X=make_frame(tables,events,t,availability_clock=b['variant']=='availability_clock')
            if len(X):frames.append(X);ys.append(labels(tables,events,t,X.UserID))
        if not frames:raise ValueError('No completed refit windows')
        heads=fit(pd.concat(frames,ignore_index=True),np.concatenate(ys),cols,b['config']['trees'],b['config']['seed'])
    fitted_model=out/'refit_model.pkl' if a.refit else out/'model.pkl'
    if a.refit:
        with fitted_model.open('wb') as f:pickle.dump({**b,'models':heads,'refit_origins':list(map(str,refit_origins))},f)
    D=make_frame(tables,events,inference_origin,ids=ids,availability_clock=b['variant']=='availability_clock')
    prob=np.tile(np.asarray(b['prior']),(len(ids),1))
    if len(D):
        loc=pd.Index(ids).get_indexer(D.UserID);prob[loc]=probability(heads,D,cols)
    prediction=(prob>=b['thresholds']).astype('int8')
    df=pd.DataFrame(prediction,columns=TARGETS);df.insert(0,'UserID',ids)
    filename='fixture_predictions.csv' if a.fixture_only else f'output_{a.student_id}.csv'
    destination=out/filename
    fd,tmp=tempfile.mkstemp(dir=out,prefix='.checked_',suffix='.csv');os.close(fd)
    try:
        df.to_csv(tmp,index=False)
        subprocess.run([sys.executable,str(VENDOR/'scripts/evaluate.py'),'--pred',tmp,'--input',str(roster)],check=True)
        if not np.array_equal(pd.read_csv(tmp).UserID.to_numpy(),ids):raise ValueError('Input order was not preserved')
        os.replace(tmp,destination)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    save_json(out/('fixture_prediction_protocol.json' if a.fixture_only else 'prediction_protocol.json'),dict(
        scope=b['scope'],target_start_utc=str(target),outcome_end_utc=str(target+pd.Timedelta(days=7)),feature_cutoff_utc=str(cutoff),
        forecast_lead_hours=float((target-cutoff).total_seconds()/3600),input_sha256=sha256(roster),output_sha256=sha256(destination),
        users=len(ids),observed_users=len(D),unseen_users=len(ids)-len(D),cold_start_policy='Calibration per-target observed prevalence, followed by the fixed threshold',
        data_provenance_sha256=sha256(Path(a.data)/'provenance.json'),model= b['variant'],model_sha256=sha256(fitted_model),
        refitted=a.refit,refit_origins=list(map(str,refit_origins)),note='An early prediction can have a long lead; no official accuracy or upload is claimed.'))
    print(destination,flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('experiment');e.add_argument('--data',required=True);e.add_argument('--out',default='artifacts_revised');e.add_argument('--config',default=str(ROOT/'config/revised.json'));e.add_argument('--fixture-only',action='store_true')
    d=sub.add_parser('predict');d.add_argument('--data',required=True);d.add_argument('--out',default='artifacts_revised');d.add_argument('--input');d.add_argument('--target-start',default='2026-11-22T00:00:00+08:00');d.add_argument('--cutoff');d.add_argument('--student-id',default='113700058');d.add_argument('--refit',action='store_true');d.add_argument('--fixture-only',action='store_true')
    a=p.parse_args();experiment(a) if a.action=='experiment' else predict_file(a)

if __name__=='__main__':main()
