"""Rebuild one common screened evaluation protocol and test dynamic FK features."""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *
from asg2.run import fit_models,predict,fingerprint
from asg2.coverage import score_subset,tune_mean_thresholds,rank_predictions,push_frequency,push_status
from asg2.dynamic_relations import dynamic_features

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tables=load_tables(a.data);events=build_events(tables);manifest=fingerprint(a.data)
    base=Path('artifacts_20261102/cache');digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    cache=out/'cache'/digest;cache.mkdir(parents=True,exist_ok=True)
    ts=lambda x:stamp(x+'T00:00:00+08:00')
    train_dates=list(pd.date_range(ts('2026-05-07'),ts('2026-08-06'),freq='7D'))
    cal_dates=list(map(ts,['2026-08-20','2026-09-10']))
    eval_dates=list(map(ts,['2026-09-24','2026-10-01']))
    design=dict(training_origins=list(map(str,train_dates)),calibration_origins=list(map(str,cal_dates)),
        evaluation_origins=list(map(str,eval_dates)),screen='No zero-record day in any of six targets during these calibration/evaluation outcome weeks; NOT proof of complete collection',
        threshold_objective='Mean of the two per-week official combined Scores',ranking='Per-target top-k, k from mean calibration-week positive rate; stable-ID tie break',
        candidate_features=['temporal','previous_relational','dynamic_relational'],decision_rules=['mean_threshold','rank_prevalence'],
        selected_by='calibration mean Score only before evaluation outcomes are loaded',
        observation_gap_hours=4,reserved_outcomes_scored=False,
        note='New uniform historical development protocol. Sep24 was used in earlier experiments and is not globally untouched.',
        data_manifest=manifest)
    assert max(train_dates)+pd.Timedelta(days=7)<=min(cal_dates)-pd.Timedelta(hours=4)
    assert max(cal_dates)+pd.Timedelta(days=7)<=min(eval_dates)-pd.Timedelta(hours=4)
    (out/'design.json').write_text(json.dumps(design,indent=2))
    # Document target-specific zero days for all calendar-week candidate origins.
    daily=events.assign(day=(events.event_time+pd.Timedelta(hours=8)).dt.floor('D')).groupby(['day','kind']).size().unstack(fill_value=0).reindex(columns=TARGETS,fill_value=0)
    coverage=[]
    for date in pd.date_range('2026-08-20','2026-10-29',freq='7D'):
        z=daily.reindex(pd.date_range(date,periods=7),fill_value=0)
        coverage.append(dict(origin_taiwan=str(date.date()),**{k:int((z[k]==0).sum()) for k in TARGETS}))
    pd.DataFrame(coverage).to_csv(out/'weekly_zero_days.csv',index=False)
    for t in cal_dates+eval_dates:
        date=t+pd.Timedelta(hours=8);z=daily.reindex(pd.date_range(date,periods=7),fill_value=0)
        if (z==0).any().any():raise ValueError('Chosen week fails the stated zero-day screen')
    def sample(t):
        key=t.strftime('%Y%m%dT%H%M')+'_gap4.pkl'
        if (cache/key).exists():
            with (cache/key).open('rb') as f:return pickle.load(f)
        if (base/digest/key).exists():
            with (base/digest/key).open('rb') as f:X,y=pickle.load(f)
        else:X=features(tables,events,t);y=labels(tables,events,t,X.UserID)
        X=dynamic_features(tables,events,X,t)
        with (cache/key).open('wb') as f:pickle.dump((X,y),f)
        print('FEATURES',str(t),len(X),flush=True);return X,y
    training=[sample(t) for t in train_dates];cal=[sample(t) for t in cal_dates]
    X=pd.concat([z[0] for z in training],ignore_index=True);y=np.concatenate([z[1] for z in training])
    C=pd.concat([z[0] for z in cal],ignore_index=True);cy=np.concatenate([z[1] for z in cal])
    folds=np.concatenate([np.repeat(i,len(z[0])) for i,z in enumerate(cal)])
    rows=[];bundles={};support=[]
    rule_calibration_mean=float(np.mean([score(z[1],np.column_stack([z[0][f't_count7_{k}'].to_numpy()>0 for k in TARGETS]))['score'] for z in cal]))
    rows.append(dict(candidate='last_week_rule',split='calibration_mean',score=rule_calibration_mean))
    for c in [c for c in X if c.startswith('d_')]:support.append(dict(feature=c,nonzero_training_rows=int((X[c]>0).sum()),training_rows=len(X)))
    pd.DataFrame(support).to_csv(out/'dynamic_support.csv',index=False)
    rates=np.mean([z[1].mean(axis=0) for z in cal],axis=0)
    for variant in design['candidate_features']:
        cols=feature_columns(X,'full' if variant=='previous_relational' else 'temporal')
        if variant=='dynamic_relational':cols += [c for c in X if c.startswith('d_')]
        models=fit_models(X,y,cols);pc=predict(models,C,cols);thresholds=tune_mean_thresholds(cy,pc,folds)
        for rule in design['decision_rules']:
            pred=pc>=thresholds if rule=='mean_threshold' else np.concatenate([rank_predictions(pc[folds==i],rates,z[0].UserID) for i,z in enumerate(cal)])
            mean=float(np.mean([score(cy[folds==i],pred[folds==i])['score'] for i in range(len(cal))]))
            name=variant+'_'+rule;bundle=dict(models=models,columns=cols,thresholds=thresholds,rates=rates,variant=variant,rule=rule,calibration_mean_score=mean)
            bundles[name]=bundle;rows.append(dict(candidate=name,split='calibration_mean',score=mean))
            print('CALIBRATION',name,mean,flush=True)
    selected=max(bundles,key=lambda n:bundles[n]['calibration_mean_score'])
    if rule_calibration_mean>bundles[selected]['calibration_mean_score']:
        raise ValueError('Rule wins calibration on this snapshot; do not export a learner as the selected model')
    selection=dict(selected_by_calibration=selected,calibration_scores={**{n:b['calibration_mean_score'] for n,b in bundles.items()},'last_week_rule':rule_calibration_mean},
        historical_training_rows=len(X),policy='No evaluation-based switching',status='Historical method comparison; final refit required')
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    for n,b in bundles.items():
        with (out/f'model_{n}.pkl').open('wb') as f:pickle.dump(b,f)
    # The retained candidate uses the existing production feature extractor.
    if selected=='temporal_mean_threshold':
        with (out/'model_temporal.pkl').open('wb') as f:pickle.dump(bundles[selected],f)
        protocol=dict(train_origins=list(map(str,train_dates)),calibration_origins=list(map(str,cal_dates)),
            holdout_origins=list(map(str,eval_dates)),observation_gap_hours=4,selected_by_calibration_only='temporal',
            threshold_objective=design['threshold_objective'],selected_candidate=selected,
            warning='Common screened historical development weeks, not confirmed complete collection or official test performance')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluation=[sample(t) for t in eval_dates]
    for name,b in bundles.items():
        combined_y=[];combined_p=[]
        for t,(D,dy) in zip(eval_dates,evaluation):
            prob=predict(b['models'],D,b['columns']);p=prob>=b['thresholds'] if b['rule']=='mean_threshold' else rank_predictions(prob,b['rates'],D.UserID)
            combined_y.append(dy);combined_p.append(p)
            rows.append(dict(candidate=name,split=str(t),n_users=len(D),**score(dy,p)))
        rows.append(dict(candidate=name,split='common_two_weeks_pooled',**score(np.concatenate(combined_y),np.concatenate(combined_p))))
        rows.append(dict(candidate=name,split='common_two_weeks_mean',score=float(np.mean([score(yy,pp)['score'] for yy,pp in zip(combined_y,combined_p)]))))
        print('EVAL',name,rows[-2]['score'],flush=True)
    # Report last-week rule on the SAME population and dates.
    yy=[];pp=[]
    for t,(D,dy) in zip(eval_dates,evaluation):
        p=np.column_stack([D[f't_count7_{k}'].to_numpy()>0 for k in TARGETS]);yy.append(dy);pp.append(p)
        rows.append(dict(candidate='last_week_rule',split=str(t),**score(dy,p)))
    rows.append(dict(candidate='last_week_rule',split='common_two_weeks_pooled',**score(np.concatenate(yy),np.concatenate(pp))))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    # Push frequency threshold uses only screened calibration labels and mean per-week push F1.
    freq=[push_frequency(events,tables,t-pd.Timedelta(hours=4),z[0].UserID) for t,z in zip(cal_dates,cal)]
    grid=np.linspace(.05,.95,37);f1=[np.mean([score_subset(z[1][:,-1:],(f>=v)[:,None])['score'] for z,f in zip(cal,freq)]) for v in grid]
    threshold=float(grid[int(np.argmax(f1))])
    fallback=dict(method='12-observation-week push frequency, clock ending at last observed push if stale',threshold=threshold,
        stale_hours=48,calibration_origins=list(map(str,cal_dates)),calibration_mean_push_f1=float(max(f1)),
        requires_stress_test=True,warning='Last observed push is NOT a verified complete-data boundary')
    (out/'fallback_config.json').write_text(json.dumps(fallback,indent=2))
    # Fixed simulated dropouts retain original truth and evaluation users.
    b=bundles[selected];stress=[]
    for t,(D,dy) in zip(eval_dates,evaluation):
        for days in [0,2,7,14,21]:
            cutoff=t-pd.Timedelta(hours=4);damaged=events[~((events.kind=='push')&(events.event_time>=cutoff-pd.Timedelta(days=days)))] if days else events
            Z=features(tables,damaged,t,user_ids=D.UserID);Z=dynamic_features(tables,damaged,Z,t)
            prob=predict(b['models'],Z,b['columns']);pred=prob>=b['thresholds'] if b['rule']=='mean_threshold' else rank_predictions(prob,b['rates'],Z.UserID)
            for mode in ['naive','frequency','last_observed_frequency']:
                p=pred.copy()
                if mode!='naive':p[:,-1]=push_frequency(damaged,tables,cutoff,Z.UserID,freeze_clock=mode=='last_observed_frequency')>=threshold
                stress.append(dict(origin_utc=str(t),missing_push_feature_days=days,policy=mode,
                    predicted_push_positive_rate=float(p[:,-1].mean()),push_f1=score_subset(dy[:,-1:],p[:,-1:])['score'],**score(dy,p)))
    pd.DataFrame(stress).to_csv(out/'push_dropout_stress.csv',index=False)
    sf=pd.DataFrame(stress);required=sf[sf.missing_push_feature_days.isin([7,14,21])]
    compare=required.pivot(index=['origin_utc','missing_push_feature_days'],columns='policy',values='push_f1')
    fallback.update(requires_stress_test=False,validated_for_use=bool((compare.last_observed_frequency>compare.naive).all()),
        stress_test_scope='Simulated trailing push-feature loss of 7/14/21 days; original labels retained',
        stress_test_mean_push_f1_naive=float(compare.naive.mean()),
        stress_test_mean_push_f1_fallback=float(compare.last_observed_frequency.mean()),
        stale_hours=168,warning='Fallback activates after seven days without push records. Last observed push is not a verified complete boundary; historical internal gaps remain a limitation.')
    (out/'fallback_config.json').write_text(json.dumps(fallback,indent=2))
    (out/'push_status.json').write_text(json.dumps(push_status(events,stamp('2026-11-02T00:00:00+08:00')-pd.Timedelta(hours=4)),indent=2))
    print('DONE',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts_coverage_20261102');run(p.parse_args())
