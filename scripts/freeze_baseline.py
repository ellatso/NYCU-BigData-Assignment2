"""Preserve the established baseline and rehearse official-ID format only."""
import argparse,hashlib,json,pickle,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from asg2.core import load_tables,build_events,features,TARGETS
from asg2.run import predict,fingerprint

def main(a):
    baseline=Path(a.baseline);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=fingerprint(a.data)
    if manifest!=json.loads((baseline/'data_manifest.json').read_text()):raise ValueError('Freeze snapshot differs from baseline')
    model_path=baseline/'model_temporal.pkl'
    with model_path.open('rb') as f:model=pickle.load(f)
    ids=pd.read_csv(Path(a.data)/'evaluation/test_input.csv').UserID.to_numpy()
    tables=load_tables(a.data);events=build_events(tables)
    # Explicitly an earlier historical-format rehearsal, never a final Nov12 CSV.
    origin='2026-11-02T00:00:00+08:00'
    X=features(tables,events,origin,user_ids=ids,gap_hours=4)
    p=predict(model['models'],X,model['columns'])>=model['thresholds']
    frame=pd.DataFrame(p.astype(int),columns=TARGETS);frame.insert(0,'UserID',X.UserID)
    if frame.UserID.tolist()!=list(ids):raise AssertionError('User order changed')
    destination=out/'holdout_predictions_format_rehearsal.csv';frame.to_csv(destination,index=False)
    check=subprocess.run([sys.executable,str(Path(a.data)/'evaluation/evaluate.py'),
        '--pred',str(destination),'--input',str(Path(a.data)/'evaluation/test_input.csv')],check=True,capture_output=True,text=True)
    decision=dict(frozen_model='original_temporal_singleweek',source_commit='1d35fcb17bfd9a8e3ab032421d68f5b7abd73136',
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),n_features=len(model['columns']),
        thresholds=model['thresholds'].tolist(),training_source='artifacts_20261102/protocol.json',
        new_recipe_adopted=False,reason='New recipes did not consistently improve inspected development outcomes',
        reserved_origin='2026-10-29T00:00:00+08:00',reserved_outcomes_scored=False,
        rehearsal_origin_taiwan=origin,rehearsal_users=len(frame),format_check=check.stdout.strip(),
        final_submission_generated=False,warning='Rehearsal checks format and ID mapping only. No November12 truth or score is available.')
    challenger=None
    if (out/'singleweek_selection.json').exists():
        selection=json.loads((out/'singleweek_selection.json').read_text())
        name=selection['selected_by_calibration'];config=selection['candidates'][name]
        challenger=out/f"model_{config['recipe']}.pkl"
        decision['experimental_challenger']=dict(name=name,**config,
            model_sha256=hashlib.sha256(challenger.read_bytes()).hexdigest(),
            status='Frozen for comparison only. Post-inspection diagnostic, not adopted.')
    (out/'freeze_decision.json').write_text(json.dumps(decision,indent=2))
    archive=Path(a.archive);archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['model_temporal.pkl','protocol.json','data_manifest.json','thresholds_temporal.json']:
            z.write(baseline/name,'frozen_baseline/'+name)
        z.writestr('frozen_baseline/freeze_decision.json',json.dumps(decision,indent=2))
        if challenger is not None:
            z.write(challenger,'experimental_challenger/model.pkl')
            z.writestr('experimental_challenger/config.json',json.dumps(decision['experimental_challenger'],indent=2))
        z.writestr('README.txt','Frozen original temporal baseline, 2026-11-02.\n'
            'Code: https://github.com/ellatso/NYCU-BigData-Assignment2/tree/1d35fcb17bfd9a8e3ab032421d68f5b7abd73136\n'
            'Install requirements.txt and load frozen_baseline/model_temporal.pkl with pickle.\n'
            'It contains models, columns, thresholds and variant. Build features with asg2.core.features and predict with asg2.run.predict.\n'
            'For later evaluation, preserve these weights and thresholds instead of retraining.\n'
            'Data are not included. Use only the official data. Check coverage before scoring.\n'
            'An experimental challenger, when present, contains separate models and config with frequency_weight and thresholds.\n'
            'For its frequency input use asg2.calibration.frequency12 from the audit-baselines-20261102 branch.\n'
            'Its probability is (1-frequency_weight)*tree_probability + frequency_weight*frequency12.\n'
            'The challenger is not adopted. Keep the original baseline as the reference.\n'
            'This archive contains no final November12 CSV and no official test score.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(decision,indent=2));print('CHECKPOINT',archive,archive.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline',default='artifacts_20261102')
    p.add_argument('--out',default='artifacts_v3_20261102');p.add_argument('--archive',required=True)
    main(p.parse_args())
