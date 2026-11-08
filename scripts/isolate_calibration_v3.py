"""Post-inspection diagnostic: isolate training recipe from calibration horizon.

This fixed factorial follow-up was added AFTER viewing initial v3 diagnostics.
All additional thresholds still use Sep24 labels only. No fresh-test claim.
"""
import argparse,importlib.util,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *
from asg2.run import predict,fingerprint
from asg2.calibration import RECIPES,frequency12,blend

def run(a):
    out=Path(a.out);base=Path(a.baseline)
    if fingerprint(a.data)!=json.loads((out/'design.json').read_text())['data_manifest']:raise ValueError('Snapshot mismatch')
    origins=['2026-10-08','2026-10-15','2026-10-22']
    amendment=dict(reason='Isolate training changes after observing multiweek threshold failure',
        timing='Added after initial v3 development scores were inspected',
        candidates=[f'{r}_single_freq{int(w*100)}' for r in RECIPES for w in [0,.25,.5]],
        calibration_origin='2026-09-24T00:00:00+08:00',development_origins=origins,
        reserved_outcomes_scored=False,threshold_selection='Sep24 only',new_training=False)
    (out/'diagnostic_amendment.json').write_text(json.dumps(amendment,indent=2))
    tables=load_tables(a.data);events=build_events(tables)
    def cached(origin):
        key=stamp(origin).strftime('%Y%m%dT%H%M')+'_gap4.pkl'
        paths=list((base/'cache').rglob(key)) or list(Path('artifacts_v2_20261102/cache').rglob(key))
        if paths:
            with paths[0].open('rb') as f:return pickle.load(f)
        X=features(tables,events,origin)
        return X,labels(tables,events,origin,X.UserID)
    C,cy=cached(amendment['calibration_origin']);dev=[cached(t+'T00:00:00+08:00') for t in origins]
    D=pd.concat([x[0] for x in dev],ignore_index=True);dy=np.concatenate([x[1] for x in dev])
    fc=frequency12(tables,events,amendment['calibration_origin'],C.UserID)
    fd=np.concatenate([frequency12(tables,events,t+'T00:00:00+08:00',x[0].UserID) for t,x in zip(origins,dev)])
    configs={};cal=[];probs={}
    for recipe in RECIPES:
        with (out/f'model_{recipe}.pkl').open('rb') as f:model=pickle.load(f)
        pc=predict(model['models'],C,model['columns']);probs[recipe]=predict(model['models'],D,model['columns'])
        for weight in [0,.25,.5]:
            name=f'{recipe}_single_freq{int(weight*100)}';p=blend(pc,fc,weight)
            threshold=tune_thresholds(cy,p)
            configs[name]=dict(recipe=recipe,frequency_weight=weight,thresholds=threshold.tolist())
            cal.append(dict(candidate=name,split='single_calibration_fit',**score(cy,p>=threshold)))
    selected=max(cal,key=lambda r:r['score'])['candidate']
    (out/'singleweek_selection.json').write_text(json.dumps(dict(selected_by_calibration=selected,candidates=configs,
        status='Post-inspection diagnostic candidates; no fresh-test validation'),indent=2))
    spec=importlib.util.spec_from_file_location('official',Path(a.data)/'evaluation/evaluate.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    rows=list(cal)
    for name,c in configs.items():
        pred=blend(probs[c['recipe']],fd,c['frequency_weight'])>=np.array(c['thresholds'])
        for split,sel in [('old_diagnostics_2weeks',D.origin<stamp('2026-10-22T00:00:00+08:00')),
                          ('all_development_3weeks',np.ones(len(D),bool)),
                          *[(str(stamp(t+'T00:00:00+08:00')),D.origin.eq(stamp(t+'T00:00:00+08:00'))) for t in origins]]:
            sel=np.asarray(sel);r=dict(candidate=name,split=split,**score(dy[sel],pred[sel]))
            truth=pd.DataFrame(dy[sel],columns=TARGETS);p=pd.DataFrame(pred[sel].astype(int),columns=TARGETS)
            for frame in [truth,p]:frame.insert(0,'UserID',np.arange(len(frame)))
            for k,v in official.scores(p,truth).items():assert abs(v-r[k])<1e-12
            rows.append(r)
        print('DIAGNOSTIC',name,score(dy,pred)['score'],flush=True)
    pd.DataFrame(rows).to_csv(out/'singleweek_diagnostic_metrics.csv',index=False)
    print('SINGLEWEEK SELECTED',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline',default='artifacts_20261102');p.add_argument('--out',default='artifacts_v3_20261102')
    run(p.parse_args())
