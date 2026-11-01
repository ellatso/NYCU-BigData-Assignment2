"""Four chronological calibration origins and a fixed nine-candidate comparison."""
import argparse,hashlib,importlib.util,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *
from asg2.run import fingerprint,predict
from asg2.calibration import RECIPES,eligible_origins,frequency12,train,blend

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    baseline=Path(a.baseline);old=json.loads((baseline/'protocol.json').read_text())
    manifest=fingerprint(a.data)
    if manifest!=json.loads((baseline/'data_manifest.json').read_text()):raise ValueError('Snapshot mismatch')
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    source={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['asg2/calibration.py','scripts/experiment_v3.py']}
    cache=out/'cache'/hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:16]
    cache.mkdir(parents=True,exist_ok=True);basecache=baseline/'cache'/digest
    tables=load_tables(a.data);events=build_events(tables)
    calibration=[stamp(d+'T00:00:00+08:00') for d in ['2026-08-13','2026-08-27','2026-09-10','2026-09-24']]
    development=[stamp(d+'T00:00:00+08:00') for d in ['2026-10-08','2026-10-15','2026-10-22']]
    all_train=list(map(stamp,old['train_origins']))
    design=dict(train_origin_pool=list(map(str,all_train)),calibration_origins=list(map(str,calibration)),
        development_origins=list(map(str,development)),recipes=RECIPES,frequency_weights=[0,.25,.5],
        recent_window='last 12 eligible origin weeks',weighting='6-week half-life, normalized mean sample weight 1',
        observation_gap_hours=4,feature_set='original temporal 51 columns',
        threshold_objective='official pooled combined-F1 across four chronological OOF weeks',
        selection='maximum pooled calibration score only, fixed recipe/alpha order breaks ties',
        reserved_origin='2026-10-29T00:00:00+08:00',reserved_outcomes_scored=False,
        data_manifest=manifest,source_sha256=source,
        caution='Calibration score fits thresholds and is optimistic. October development weeks have been inspected before.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    def sample(t):
        t=stamp(t);key=t.strftime('%Y%m%dT%H%M')+'_gap4.pkl';path=basecache/key
        if path.exists():
            with path.open('rb') as f:X,y=pickle.load(f)
        else:
            X=features(tables,events,t);y=labels(tables,events,t,X.UserID)
        return X,y
    loaded={t:sample(t) for t in sorted(set(all_train+calibration))}
    def training(origins):
        return pd.concat([loaded[t][0] for t in origins],ignore_index=True),np.concatenate([loaded[t][1] for t in origins])
    cols=feature_columns(loaded[all_train[0]][0],'temporal')
    oof={r:[] for r in RECIPES};all_y=[];all_freq=[];fold_ids=[];fold_rows=[]
    for fold,t in enumerate(calibration):
        C,cy=loaded[t];freq=frequency12(tables,events,t,C.UserID)
        all_y.append(cy);all_freq.append(freq);fold_ids.extend([fold]*len(C))
        for recipe in RECIPES:
            origins=eligible_origins(all_train,t,recipe)
            assert max(origins)+pd.Timedelta(days=7)<=t-pd.Timedelta(hours=4)
            X,y=training(origins);checkpoint=cache/f'fold{fold}_{recipe}.pkl'
            if checkpoint.exists():
                with checkpoint.open('rb') as f:models=pickle.load(f)
            else:
                models=train(X,y,cols,recipe)
                with checkpoint.open('wb') as f:pickle.dump(models,f)
            oof[recipe].append(predict(models,C,cols))
            fold_rows.append(dict(fold=fold,calibration_origin_utc=str(t),recipe=recipe,training_rows=len(X),
                training_origins=list(map(str,origins)),last_label_end_utc=str(max(origins)+pd.Timedelta(days=7)),
                feature_cutoff_utc=str(t-pd.Timedelta(hours=4))))
            print('FOLD',str(t),recipe,len(origins),len(X),flush=True)
    ycal=np.concatenate(all_y);freqcal=np.concatenate(all_freq);fold_ids=np.array(fold_ids)
    rows=[];threshold_rows=[];configs={}
    for recipe in RECIPES:
        pc=np.concatenate(oof[recipe])
        for alpha in design['frequency_weights']:
            name=f'{recipe}_freq{int(alpha*100)}';prob=blend(pc,freqcal,alpha)
            thresholds=tune_thresholds(ycal,prob);configs[name]=dict(recipe=recipe,frequency_weight=alpha,thresholds=thresholds.tolist())
            rows.append(dict(candidate=name,split='pooled_calibration_fit',**score(ycal,prob>=thresholds)))
            for f,t in enumerate(calibration):
                sel=fold_ids==f
                rows.append(dict(candidate=name,split='calibration_'+str(t),**score(ycal[sel],prob[sel]>=thresholds)))
                # Stability diagnostic only: each omitted-week fit is not a prospective backtest.
                th=tune_thresholds(ycal[~sel],prob[~sel])
                threshold_rows.append(dict(candidate=name,omitted_calibration_week=str(t),**dict(zip(TARGETS,map(float,th)))))
            print('CALIBRATION',name,score(ycal,prob>=thresholds)['score'],flush=True)
    winner=max([r for r in rows if r['split']=='pooled_calibration_fit'],key=lambda r:r['score'])['candidate']
    (out/'selection.json').write_text(json.dumps(dict(selected_by_calibration=winner,candidates=configs,
        status='experimental frozen comparison; final submission route unchanged'),indent=2))
    (out/'fold_protocol.json').write_text(json.dumps(fold_rows,indent=2))
    pd.DataFrame(threshold_rows).to_csv(out/'threshold_stability.csv',index=False)
    # Freeze selection BEFORE loading development outcomes.
    dev_samples=[sample(t) for t in development]
    D=pd.concat([z[0] for z in dev_samples],ignore_index=True);dy=np.concatenate([z[1] for z in dev_samples])
    freqdev=np.concatenate([frequency12(tables,events,t,z[0].UserID) for t,z in zip(development,dev_samples)])
    spec=importlib.util.spec_from_file_location('official',Path(a.data)/'evaluation/evaluate.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    dev_predictions={}
    final_train_protocol={}
    for recipe in RECIPES:
        origins=eligible_origins(all_train,development[0],recipe);X,y=training(origins)
        models=train(X,y,cols,recipe);prob=predict(models,D,cols)
        final_train_protocol[recipe]=dict(origins=list(map(str,origins)),rows=len(X))
        with (out/f'model_{recipe}.pkl').open('wb') as f:pickle.dump(dict(models=models,columns=cols,recipe=recipe),f)
        for name,c in configs.items():
            if c['recipe']!=recipe:continue
            dev_predictions[name]=blend(prob,freqdev,c['frequency_weight'])>=np.array(c['thresholds'])
    with (baseline/'model_temporal.pkl').open('rb') as f:control=pickle.load(f)
    dev_predictions['original_singleweek']=predict(control['models'],D,control['columns'])>=control['thresholds']
    dev_predictions['frequency_only_0.5']=freqdev>=.5
    pd.DataFrame(dict(UserID=D.UserID,origin=D.origin)).to_csv(out/'holdout_predictions_development_ids.csv',index=False)
    for name,prediction in dev_predictions.items():
        np.savez_compressed(out/f'predictions_{name}.npz',pred=prediction,y=dy)
        for split,sel in [('old_diagnostics_2weeks',D.origin<development[-1]),('all_development_3weeks',np.ones(len(D),bool)),
                          *[(str(t),D.origin.eq(t)) for t in development]]:
            sel=np.asarray(sel);r=dict(candidate=name,split=split,n_users=int(sel.sum()),**score(dy[sel],prediction[sel]))
            truth=pd.DataFrame(dy[sel],columns=TARGETS);pred=pd.DataFrame(prediction[sel].astype(int),columns=TARGETS)
            for z in [truth,pred]:z.insert(0,'UserID',np.arange(len(z)))
            for k,v in official.scores(pred,truth).items():assert abs(v-r[k])<1e-12
            rows.append(r)
        print('DEVELOPMENT',name,score(dy,prediction)['score'],flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'final_training.json').write_text(json.dumps(final_train_protocol,indent=2))
    print('DONE calibrated choice:',winner,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline',default='artifacts_20261102');p.add_argument('--out',default='artifacts_v3_20261102')
    main(p.parse_args())
