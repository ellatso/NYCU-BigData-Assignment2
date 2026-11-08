"""Fixed second-round development experiment. Never score Oct29 in this script."""
import argparse, hashlib, importlib.util, json, pickle, sys, warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *
from asg2.run import predict,fingerprint,Constant,LGBMClassifier
from asg2.enhanced import enrich,suspect_push_days,push_training_mask

def main(a):
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    prior=Path(a.baseline);protocol=json.loads((prior/'protocol.json').read_text())
    manifest=fingerprint(a.data)
    if manifest!=json.loads((prior/'data_manifest.json').read_text()):raise ValueError('Snapshot differs from baseline; rerun audit first')
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    basecache=prior/'cache'/digest
    source_digest=hashlib.sha256(Path('asg2/enhanced.py').read_bytes()).hexdigest()
    cache=out/'cache'/f'{digest}_{source_digest[:12]}';cache.mkdir(parents=True,exist_ok=True)
    tables=load_tables(a.data);events=build_events(tables)
    designs=[('baseline',False,False,False),('baseline_pushmask',False,False,True),
             ('short_temporal',True,False,False),('short_temporal_pushmask',True,False,True),
             ('short_threads',True,True,False),('short_threads_pushmask',True,True,True)]
    design=dict(candidates=[x[0] for x in designs],train_origins=protocol['train_origins'],
        calibration_origins=protocol['calibration_origins'],
        development_origins=[str(stamp(x+'T00:00:00+08:00')) for x in ['2026-10-08','2026-10-15','2026-10-22']],
        reserved_origin='2026-10-29T00:00:00+08:00',reserved_outcomes_scored=False,
        coverage_heuristic='Taiwan day: push=0, other activity>0, prior28d median pushes>=100; push model only excludes affected outcome weeks',
        thresholds='Sep24 only, same combined-F1 coordinate search as baseline',
        selection='calibration only; development results are not an unbiased estimate',
        source_sha256=source_digest,data_manifest=manifest)
    (out/'design.json').write_text(json.dumps(design,indent=2))
    suspects=suspect_push_days(events)
    def samples(origins):
        frames=[];ys=[]
        for origin in origins:
            t=stamp(origin);key=t.strftime('%Y%m%dT%H%M')+'_gap4.pkl'
            if (cache/key).exists():
                with (cache/key).open('rb') as f:X,y=pickle.load(f)
            else:
                if (basecache/key).exists():
                    with (basecache/key).open('rb') as f:base,y=pickle.load(f)
                else:
                    base=features(tables,events,t);y=labels(tables,events,t,base.UserID)
                X=enrich(tables,events,base,t)
                with (cache/key).open('wb') as f:pickle.dump((X,y),f)
            frames.append(X);ys.append(y);print('FEATURES',str(t),len(X),flush=True)
        return pd.concat(frames,ignore_index=True),np.concatenate(ys)
    X,y=samples(design['train_origins']);C,cy=samples(design['calibration_origins'])
    # Training-row exclusion is fixed before seeing any development scores.
    keep=push_training_mask(X.origin,suspects)
    affected=sorted(str(x) for x in X.loc[~keep,'origin'].unique())
    design.update(push_train_rows_before=len(X),push_train_rows_after=int(keep.sum()),excluded_train_origins=affected)
    (out/'design.json').write_text(json.dumps(design,indent=2))
    (out/'suspect_push_days.json').write_text(json.dumps([str(x.date()) for x in suspects],indent=2))
    fits={};rows=[]
    for name,short,rel,mask in designs:
        cols=feature_columns(X,'temporal')
        if short:cols += [c for c in X if c.startswith('v2_')]
        if rel:cols += [c for c in X if c.startswith('v2r_')]
        models=[]
        for j in range(6):
            use=keep if mask and j==5 else np.ones(len(X),bool)
            if not use.any():raise ValueError('Mask removes every training row')
            # Reuse the exact baseline classifier configuration and target-specific loss.
            yy=y[use,j]
            if len(np.unique(yy))<2:m=Constant(float(yy.mean()))
            else:
                m=LGBMClassifier(n_estimators=160,learning_rate=.045,num_leaves=23,max_depth=-1,
                    min_child_samples=70,reg_lambda=4,subsample=.85,subsample_freq=1,
                    colsample_bytree=.85,random_state=42,n_jobs=4,verbosity=-1)
                m.fit(X.loc[use,cols],yy)
            models.append(m)
        pc=predict(models,C,cols);thresholds=tune_thresholds(cy,pc)
        fits[name]={'models':models,'columns':cols,'thresholds':thresholds}
        rows.append(dict(candidate=name,split='calibration',**score(cy,pc>=thresholds)))
        with (out/f'model_{name}.pkl').open('wb') as f:pickle.dump(fits[name],f)
        print('CALIBRATION',name,rows[-1]['score'],flush=True)
    selected=max(rows,key=lambda r:r['score'])['candidate']
    (out/'selection.json').write_text(json.dumps(dict(selected_by_calibration=selected,
        note='Experimental candidate only; not wired to final submission',thresholds={n:v['thresholds'].tolist() for n,v in fits.items()}),indent=2))
    # Only now expose already-inspected development outcomes; Oct29 is never requested.
    D,dy=samples(design['development_origins'])
    spec=importlib.util.spec_from_file_location('official',Path(a.data)/'evaluation/evaluate.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    for name,c in fits.items():
        probabilities=predict(c['models'],D,c['columns']);p=probabilities>=c['thresholds']
        np.savez_compressed(out/f'probabilities_{name}.npz',prob=probabilities,y=dy,ids=D.UserID.to_numpy(),origins=D.origin.to_numpy())
        for split,sel in [('old_diagnostics_2weeks',D.origin<stamp('2026-10-22T00:00:00+08:00')),
                          ('all_development_3weeks',np.ones(len(D),bool)),
                          *[(str(t),D.origin.eq(t)) for t in sorted(D.origin.unique())]]:
            sel=np.asarray(sel)
            row=dict(candidate=name,split=split,n_users=int(sel.sum()),**score(dy[sel],p[sel]))
            # Verify aggregate scoring against the supplied evaluator, with unique synthetic row IDs.
            truth=pd.DataFrame(dy[sel],columns=TARGETS);pred=pd.DataFrame(p[sel].astype(int),columns=TARGETS)
            for frame in [truth,pred]:frame.insert(0,'UserID',np.arange(len(frame)))
            for k,v in official.scores(pred,truth).items():assert abs(v-row[k])<1e-12
            rows.append(row)
        print('DEV',name,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    print('DONE selected_by_calibration',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline',default='artifacts_20261102');p.add_argument('--out',default='artifacts_v2_20261102')
    main(p.parse_args())
