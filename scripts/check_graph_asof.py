"""Real-snapshot as-of invariance and graph checkpoint/forward checks."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from asg2.core import load_tables,build_events,stamp
from asg2.graph import graph_snapshot

def equal_graph(a,b):
    for key in ['xu','xr','ids','selected']:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    for key in ['ur','ru','peers']:
        for x,y in zip(a[key],b[key]):
            torch.testing.assert_close(x.indices(),y.indices(),rtol=0,atol=0)
            torch.testing.assert_close(x.values(),y.values(),rtol=0,atol=0)

def main(data,out):
    torch.set_num_threads(4)
    tables=load_tables(data);events=build_events(tables);t=stamp('2026-09-24T00:00:00+08:00')
    original=graph_snapshot(tables,events,t,with_labels=False)
    future=events.iloc[:50].copy();future['available_at']=t+pd.Timedelta(days=100);future['event_time']=t-pd.Timedelta(days=1)
    equal_graph(original,graph_snapshot(tables,pd.concat([events,future],ignore_index=True),t,with_labels=False))
    future['event_time']=t+pd.Timedelta(days=100);future['available_at']=t-pd.Timedelta(days=1)
    equal_graph(original,graph_snapshot(tables,pd.concat([events,future],ignore_index=True),t,with_labels=False))
    protocol=json.loads((Path(out)/'protocol.json').read_text())
    actual=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('asg2/graph.py').read_bytes()).hexdigest()
    assert protocol['implementation_sha256']==actual
    assert protocol['epochs_fixed_before_diagnostic']==8
    state=torch.load(Path(out)/'relational_gnn.pt',map_location='cpu',weights_only=True)
    assert state['implementation_sha256']==actual
    status={'graph_future_availability_excluded':True,'graph_future_occurrence_excluded':True,'checkpoint_matches_graph_implementation':True,'fixed_eight_epoch_experiment':True,'n_selected_users':len(original['selected'])}
    (Path(out)/'checks.json').write_text(json.dumps(status,indent=2));print('Graph as-of checks PASS',status)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts_step2');a=p.parse_args();main(a.data,a.out)
