"""Train, calibrate and hold out time-separated weeks. Never use holdout to tune thresholds."""
import argparse,csv,json,pickle,subprocess,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from .core import *

class Constant:
    def __init__(self,p):self.p=p
    def predict_proba(self,X):return np.tile([1-self.p,self.p],(len(X),1))

def fit_models(X,y,columns,trees=160):
    models=[]
    for j,kind in enumerate(TARGETS):
        if len(np.unique(y[:,j]))<2:m=Constant(float(y[:,j].mean()))
        else:
            m=LGBMClassifier(n_estimators=trees,learning_rate=.045,num_leaves=23,max_depth=-1,min_child_samples=70,reg_lambda=4,subsample=.85,subsample_freq=1,colsample_bytree=.85,random_state=42,n_jobs=4,verbosity=-1)
            m.fit(X[columns],y[:,j])
        models.append(m)
    return models

def predict(models,X,columns):return np.column_stack([m.predict_proba(X[columns])[:,1] for m in models])

def fingerprint(data):
    base=Path(data)/'data/db'
    if not base.exists():base=Path(data)
    out={}
    for name in TABLES:
        p=base/f'{name}.parquet';h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        out[name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    return out

def make_samples(tables,events,origins,cache,gap):
    frames=[];ys=[]
    coverage=events.available_at.max()
    for t in origins:
        t=stamp(t)
        if t+pd.Timedelta(days=7)>coverage:raise ValueError(f'Outcome week not yet fully observed: {t}')
        key=t.strftime('%Y%m%dT%H%M')+f'_gap{gap}'
        path=cache/f'{key}.pkl'
        if path.exists():
            with path.open('rb') as f:X,y=pickle.load(f)
            print('Cached',key,len(X),flush=True)
        else:
            X=features(tables,events,t,gap_hours=gap);y=labels(tables,events,t,X.UserID)
            with path.open('wb') as f:pickle.dump((X,y),f)
            print('Built',key,len(X),flush=True)
        if not len(X):raise ValueError('No eligible users')
        frames.append(X);ys.append(y)
    return pd.concat(frames,ignore_index=True),np.concatenate(ys)

def dates(start,end):return pd.date_range(start,end,freq='7D',tz='Asia/Taipei').tz_convert('UTC').tz_localize(None)

def benchmark(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tables=load_tables(args.data);events=build_events(tables)
    manifest=fingerprint(args.data)
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    cache=out/'cache'/digest;cache.mkdir(parents=True,exist_ok=True)
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    train_dates=dates(args.train_start,'2026-09-17');cal_dates=dates('2026-09-24','2026-10-01');test_dates=dates('2026-10-08','2026-10-15')
    # Outcomes of every preceding split finish before the next split's feature cutoff.
    # Last training origin Sep17 has outcomes through Sep24; calibration's earliest
    # feature cutoff is Sep23 20:00, so exclude Sep17 to preserve the 4-hour gap.
    # Likewise keep a full calendar week between calibration and holdout origins.
    train_dates=train_dates[train_dates+pd.Timedelta(days=7)<=cal_dates.min()-pd.Timedelta(hours=args.gap_hours)]
    cal_dates=cal_dates[cal_dates+pd.Timedelta(days=7)<=test_dates.min()-pd.Timedelta(hours=args.gap_hours)]
    Xtr,ytr=make_samples(tables,events,train_dates,cache,args.gap_hours)
    Xcal,ycal=make_samples(tables,events,cal_dates,cache,args.gap_hours)
    Xtest,ytest=make_samples(tables,events,test_dates,cache,args.gap_hours)
    results=[]
    # A rule baseline uses observed activity in the last 7 days, no fitting.
    def lastweek(X):return np.column_stack([X[f't_count7_{k}'].to_numpy()>0 for k in TARGETS])
    rule=score(ytest,lastweek(Xtest));rule.update(model='last_week_rule',split='holdout');results.append(rule)
    best=None;best_cal=-1
    probabilities={}
    for variant in ['static','temporal','relational','full']:
        cols=feature_columns(Xtr,variant);t0=time.perf_counter()
        models=fit_models(Xtr,ytr,cols,args.trees)
        probcal=predict(models,Xcal,cols)
        thresholds=tune_thresholds(ycal,probcal)
        probtest=predict(models,Xtest,cols);probabilities[variant]=probtest
        calscore=score(ycal,probcal>=thresholds)
        if calscore['score']>best_cal:best_cal=calscore['score'];best=variant
        row=score(ytest,probtest>=thresholds)
        row.update(model=variant,split='holdout',n_features=len(cols),fit_seconds=time.perf_counter()-t0,calibration_score=calscore['score'])
        results.append(row)
        for origin in sorted(Xtest.origin.unique()):
            mask=Xtest.origin.eq(origin).to_numpy()
            results.append({**score(ytest[mask],probtest[mask]>=thresholds),'model':variant,'split':str(origin),'n_users':int(mask.sum())})
        with (out/f'model_{variant}.pkl').open('wb') as f:pickle.dump({'models':models,'columns':cols,'thresholds':thresholds,'variant':variant},f)
        (out/f'thresholds_{variant}.json').write_text(json.dumps(dict(zip(TARGETS,map(float,thresholds))),indent=2))
        pred=pd.DataFrame((probtest>=thresholds).astype(int),columns=TARGETS);pred.insert(0,'UserID',Xtest.UserID.to_numpy());pred.insert(1,'origin',Xtest.origin)
        pred.to_csv(out/f'holdout_predictions_{variant}.csv',index=False)
        print('RESULT',row,flush=True)
    df=pd.DataFrame(results);df.to_csv(out/'metrics.csv',index=False)
    protocol={'train_origins':[str(x) for x in train_dates],'calibration_origins':[str(x) for x in cal_dates],'holdout_origins':[str(x) for x in test_dates],'train_rows':len(Xtr),'calibration_rows':len(Xcal),'holdout_rows':len(Xtest),'observation_gap_hours':args.gap_hours,'data_latest_available_timestamp':str(events.available_at.max()),'selected_by_calibration_only':best,'selected_calibration_score':best_cal,'seed':42,'threshold_objective':'(macro F1 + micro F1)/2, official empty-class convention','historical_eligibility':'cohort users with >=5 observed target events in past90 days (proxy for test selection)','warning':'Temporal holdout is historical, not official November test performance.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('\nComplete. Selected using calibration ONLY:',best)
    print(df[df.split=='holdout'].to_string(index=False))

def final_predict(args):
    out=Path(args.out);protocol=json.loads((out/'protocol.json').read_text());variant=protocol['selected_by_calibration_only']
    with (out/f'model_{variant}.pkl').open('rb') as f:chosen=pickle.load(f)
    tables=load_tables(args.data);events=build_events(tables)
    ids=pd.read_csv(args.input).UserID.to_numpy()
    target=stamp('2026-11-12 00:00:00+08:00');observation_end=target-pd.Timedelta(hours=4)
    if events.available_at.max()<pd.Timestamp('2026-11-11 11:00:00'):
        raise ValueError('Official final predictions require the final Nov11 release. Do not submit an October dry run.')
    # Refitting includes only complete outcome windows before the observation cutoff.
    start=protocol['train_origins'][0]
    origins=pd.date_range(start,target-pd.Timedelta(days=7),freq='7D')
    origins=origins[origins+pd.Timedelta(days=7)<=observation_end]
    manifest=fingerprint(args.data);digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    cache=out/'cache'/digest;cache.mkdir(parents=True,exist_ok=True)
    X,y=make_samples(tables,events,origins,cache,4)
    cols=chosen['columns'];models=fit_models(X,y,cols,args.trees)
    Xtest=features(tables,events,target,user_ids=ids,gap_hours=4)
    pred=predict(models,Xtest,cols)>=chosen['thresholds']
    df=pd.DataFrame(pred.astype('int8'),columns=TARGETS);df.insert(0,'UserID',Xtest.UserID.to_numpy())
    destination=out/f'output_{args.student_id}.csv';df.to_csv(destination,index=False)
    # Run the OFFICIAL evaluator, which checks both header and exact user set.
    cmd=['python',str(Path(args.data)/'evaluation/evaluate.py'),'--pred',str(destination),'--input',args.input]
    subprocess.run(cmd,check=True)
    (out/'final_protocol.json').write_text(json.dumps({'model':variant,'target_start_utc':str(target),'feature_cutoff_utc':str(observation_end),'refit_origins':[str(t) for t in origins],'thresholds_from_historical_calibration':chosen['thresholds'].tolist(),'data_manifest':manifest},indent=2))
    print('Official-format prediction file:',destination)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['benchmark','predict']);p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts');p.add_argument('--train-start',default='2026-05-07');p.add_argument('--trees',type=int,default=160);p.add_argument('--gap-hours',type=int,default=4);p.add_argument('--input');p.add_argument('--student-id',default='113700058')
    a=p.parse_args()
    if a.action=='predict' and not a.input:p.error('--input is required for predict')
    benchmark(a) if a.action=='benchmark' else final_predict(a)

if __name__=='__main__':main()
