"""Fixed neural comparison; previously inspected October weeks are diagnostics."""
import argparse,hashlib,json,time,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import load_tables,build_events,score,tune_thresholds,stamp
from .run import dates,fingerprint
from .graph import graph_snapshot,RelationalNet,to_device,GRAPH_VERSION

def graph_cache(tables,events,origin,cache):
    p=cache/(stamp(origin).strftime('%Y%m%dT%H%M')+'.pt')
    if p.exists():return p
    if stamp(origin)+pd.Timedelta(days=7)>events.available_at.max():raise ValueError(f'Incomplete outcome window: {origin}')
    g=graph_snapshot(tables,events,origin);torch.save(g,p)
    print('Built graph',origin,'labeled users',len(g['selected']),flush=True)
    return p

def load_graph(path,device):return to_device(torch.load(path,map_location='cpu',weights_only=True),device)

def predict(model,paths,device):
    probs=[];ys=[];model.eval()
    with torch.no_grad():
        for p in paths:
            g=load_graph(p,device);probs.append(torch.sigmoid(model(g)[g['selected']]).cpu().numpy());ys.append(g['y'].cpu().numpy())
    return np.concatenate(probs),np.concatenate(ys)

def setup(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    device=torch.device(args.device);tables=load_tables(args.data);events=build_events(tables)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest=fingerprint(args.data)
    implementation=hashlib.sha256(Path(__file__).with_name('graph.py').read_bytes()).hexdigest()
    digest=hashlib.sha256(json.dumps({'data':manifest,'version':GRAPH_VERSION,'code':implementation},sort_keys=True).encode()).hexdigest()[:16]
    cache=out/'cache'/digest;cache.mkdir(parents=True,exist_ok=True)
    (out/('data_manifest.json' if args.action=='train' else 'forward_data_manifest.json')).write_text(json.dumps(manifest,indent=2))
    return tables,events,out,cache,device,implementation

def train(args):
    tables,events,out,cache,device,implementation=setup(args)
    train_origins=dates('2026-05-07','2026-09-10');cal_origins=dates('2026-09-24','2026-09-24');diag_origins=dates('2026-10-08','2026-10-15')
    assert train_origins.max()+pd.Timedelta(days=7)<=cal_origins.min()-pd.Timedelta(hours=4)
    tr=[graph_cache(tables,events,t,cache) for t in train_origins];cal=[graph_cache(tables,events,t,cache) for t in cal_origins];diag=[graph_cache(tables,events,t,cache) for t in diag_origins]
    first=load_graph(tr[0],device);dims=(first['xu'].shape[1],first['xr'].shape[1]);del first
    train_y=torch.cat([load_graph(p,'cpu')['y'] for p in tr]);positive=train_y.sum(0)
    pos_weight=((len(train_y)-positive)/(positive+1)).clamp(1,10).to(device);rows=[]
    for name,enabled in [('user_mlp',False),('relational_gnn',True)]:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=RelationalNet(*dims,hidden=args.hidden,use_graph=enabled).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        t0=time.perf_counter();history=[]
        for epoch in range(args.epochs):
            model.train();running=0
            # Shuffle training-week optimization order; samples/splits stay temporal.
            for j in np.random.default_rng(42+epoch).permutation(len(tr)):
                g=load_graph(tr[int(j)],device);optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(g)[g['selected']],g['y']);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();running+=loss.item()
            history.append({'epoch':epoch+1,'mean_graph_loss':running/len(tr)})
            print(name,'epoch',epoch+1,'loss',history[-1]['mean_graph_loss'],flush=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-t0;pcal,ycal=predict(model,cal,device);thresholds=tune_thresholds(ycal,pcal);pdiag,ydiag=predict(model,diag,device)
        row={**score(ydiag,pdiag>=thresholds),'model':name,'split':'development_diagnostic','training_seconds':elapsed,'epochs':args.epochs,'hidden':args.hidden,'calibration_score':score(ycal,pcal>=thresholds)['score']};rows.append(row)
        for origin,path in zip(diag_origins,diag):
            p,y=predict(model,[path],device);rows.append({**score(y,p>=thresholds),'model':name,'split':str(origin),'n_users':len(y)})
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'user_dim':dims[0],'repo_dim':dims[1],'hidden':args.hidden,'use_graph':enabled,'thresholds':thresholds.tolist(),'graph_version':GRAPH_VERSION,'implementation_sha256':implementation},out/f'{name}.pt')
        pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
        (out/f'{name}_thresholds.json').write_text(json.dumps(thresholds.tolist()))
        print('RESULT',row,flush=True);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    protocol={'graph_version':GRAPH_VERSION,'implementation_sha256':implementation,'train_origins':[str(x) for x in train_origins],'calibration_origins':[str(x) for x in cal_origins],'diagnostic_origins':[str(x) for x in diag_origins],'planned_fresh_origins':['2026-10-22 00:00:00+08:00','2026-10-29 00:00:00+08:00'],'epochs_fixed_before_diagnostic':args.epochs,'hidden_fixed_before_diagnostic':args.hidden,'seed':42,'device':str(device),'torch_version':str(torch.__version__),'observation_gap_hours':4,'diagnostic_not_used_for_epoch_selection':True,'notes':'October8/15 were already inspected in Step1, so scores are development diagnostics. Freeze candidates before evaluating new forward weeks.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(pd.DataFrame(rows).to_string(index=False))

def forward(args):
    tables,events,out,cache,device,implementation=setup(args);rows=[];pending=[]
    for raw in args.origins:
        t=stamp(raw)
        if t+pd.Timedelta(days=7)>events.available_at.max():pending.append(raw);print('PENDING: outcome week not fully observed:',raw);continue
        path=graph_cache(tables,events,t,cache)
        for name in ['user_mlp','relational_gnn']:
            state=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True)
            if state['implementation_sha256']!=implementation:raise ValueError('Graph implementation differs from trained checkpoint')
            model=RelationalNet(state['user_dim'],state['repo_dim'],state['hidden'],state['use_graph']).to(device);model.load_state_dict(state['state_dict'])
            p,y=predict(model,[path],device);rows.append({**score(y,p>=np.asarray(state['thresholds'])),'model':name,'origin':str(t),'split':'forward_frozen_comparison','n_users':len(y)})
    # Step 1's checkpoint must be fitted on the original development snapshot,
    # not retrained after reading these forward-week labels.
    baseline_path=out.parent/'artifacts/model_temporal.pkl'
    if baseline_path.exists():
        baseline_manifest=json.loads((baseline_path.parent/'data_manifest.json').read_text())
        training_manifest=json.loads((out/'data_manifest.json').read_text())
        if baseline_manifest!=training_manifest:
            raise ValueError('Frozen temporal baseline must use the same training data snapshot as the graph candidates.')
        import pickle
        from .core import features
        from .run import predict as tree_predict
        with baseline_path.open('rb') as f:baseline=pickle.load(f)
        for raw in args.origins:
            t=stamp(raw)
            if t+pd.Timedelta(days=7)>events.available_at.max():continue
            g=load_graph(graph_cache(tables,events,t,cache),'cpu')
            X=features(tables,events,t)
            p=tree_predict(baseline['models'],X,baseline['columns'])
            from .core import labels
            y=labels(tables,events,t,X.UserID)
            rows.append({**score(y,p>=baseline['thresholds']),'model':'temporal_lightgbm','origin':str(t),'split':'forward_frozen_comparison','n_users':len(y)})
    (out/'forward_status.json').write_text(json.dumps({'pending':pending,'rows':rows,'temporal_baseline_checkpoint_available':baseline_path.exists()},indent=2))
    if rows:pd.DataFrame(rows).to_csv(out/'forward_metrics.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','forward']);p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts_step2');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=8);p.add_argument('--hidden',type=int,default=32);p.add_argument('--origins',nargs='+',default=['2026-10-22 00:00:00+08:00','2026-10-29 00:00:00+08:00']);a=p.parse_args();train(a) if a.action=='train' else forward(a)
if __name__=='__main__':main()
