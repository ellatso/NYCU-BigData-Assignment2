"""Focused checks for leakage, split timing, identifiers and official scoring."""
import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import *

def main(data,out):
    tables=load_tables(data);events=build_events(tables);t=stamp('2026-09-24T00:00:00+08:00')
    x=features(tables,events,t)
    future=events.iloc[:100].copy()
    future['available_at']=t+pd.Timedelta(days=100)
    future['event_time']=t-pd.Timedelta(days=1)
    x2=features(tables,pd.concat([events,future],ignore_index=True),t)
    pd.testing.assert_frame_equal(x,x2)
    future['event_time']=t+pd.Timedelta(days=100)
    future['available_at']=t-pd.Timedelta(days=1)
    x3=features(tables,pd.concat([events,future],ignore_index=True),t)
    pd.testing.assert_frame_equal(x,x3)
    assert x.UserID.is_unique
    assert set(x.UserID).issubset(set(tables['Users'].github_user_id))
    # Official empty-class F1 convention differs from sklearn's usual default.
    module=Path(data)/'evaluation/evaluate.py'
    spec=importlib.util.spec_from_file_location('official_evaluation',module)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    rng=np.random.default_rng(42)
    for y,p in [(np.zeros((8,6),int),np.zeros((8,6),int)),(rng.integers(0,2,(100,6)),rng.integers(0,2,(100,6)))]:
        truth=pd.DataFrame(y,columns=TARGETS);truth.insert(0,'UserID',np.arange(len(y)))
        pred=pd.DataFrame(p,columns=TARGETS);pred.insert(0,'UserID',np.arange(len(y)))
        expected=official.scores(pred,truth);got=score(y,p)
        for k in ['macro_f1','micro_f1','score']:assert abs(got[k]-expected[k])<1e-12
    protocol=json.loads((Path(out)/'protocol.json').read_text())
    gap=pd.Timedelta(hours=protocol['observation_gap_hours'])
    assert max(map(stamp,protocol['train_origins']))+pd.Timedelta(days=7)<=min(map(stamp,protocol['calibration_origins']))-gap
    assert max(map(stamp,protocol['calibration_origins']))+pd.Timedelta(days=7)<=min(map(stamp,protocol['holdout_origins']))-gap
    result={'future_availability_excluded':True,'future_occurrence_excluded':True,'stable_github_output_ids':True,'official_score_equivalence_including_empty_classes':True,'nonoverlapping_train_calibration_holdout':True}
    (Path(out)/'checks.json').write_text(json.dumps(result,indent=2))
    print('CHECKS PASSED:',result)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts');a=p.parse_args();main(a.data,a.out)
