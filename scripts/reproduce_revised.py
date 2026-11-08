"""Rebuild the eight tables and refit from scratch; require byte-identical CSV output."""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from asg2.provenance import ROOT,VENDOR,sha256,verify_data
from asg2.revised import experiment,predict_file,save_json

def run(a):
    fixture=getattr(a,'fixture_only',False)
    data=Path(a.data);out=Path(a.out);meta=json.loads((data/'provenance.json').read_text()) if fixture else verify_data(data)
    if fixture and meta.get('dataset_kind')!='synthetic_fixture':raise ValueError('Fixture reproduction accepts only explicitly labeled synthetic source data')
    protocol=json.loads((out/('fixture_prediction_protocol.json' if fixture else 'prediction_protocol.json')).read_text())
    with (out/'model.pkl').open('rb') as f:bundle=pickle.load(f)
    expected=out/('fixture_predictions.csv' if fixture else f'output_{a.student_id}.csv')
    if sha256(expected)!=protocol['output_sha256']:raise ValueError('Expected CSV changed after prediction')
    with tempfile.TemporaryDirectory(prefix='asg2-reproduce-') as td:
        root=Path(td);rebuilt=root/'rebuilt';rerun=root/'rerun'
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_revised.py'),'build','--raw',a.raw,'--out',str(rebuilt)],check=True)
        again=json.loads((rebuilt/'provenance.json').read_text()) if fixture else verify_data(rebuilt)
        if meta['table_sha256']!=again['table_sha256']:raise ValueError('Official tables differ on rebuild')
        config=root/'config.json';save_json(config,bundle['config'])
        experiment(SimpleNamespace(data=str(rebuilt),out=str(rerun),config=str(config),fixture_only=fixture))
        predict_file(SimpleNamespace(data=str(rebuilt),out=str(rerun),input=a.input,
            target_start=protocol['target_start_utc'],cutoff=protocol['feature_cutoff_utc'],student_id=a.student_id,
            refit=protocol['refitted'],fixture_only=fixture))
        actual=rerun/expected.name
        if actual.read_bytes()!=expected.read_bytes():raise ValueError('Predictions differ after independent rebuild/refit')
    save_json(out/('fixture_reproduction_checks.json' if fixture else 'reproduction_checks.json'),dict(passed=True,rebuilt_all_eight_tables=True,independent_model_fit=True,
        identical_csv_bytes=True,output_sha256=sha256(expected),scope='synthetic_fixture_only' if fixture else 'official_gharchive'))
    print('REPRODUCTION PASSED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--data',required=True);p.add_argument('--out',default='artifacts_revised');p.add_argument('--input',default=str(VENDOR/'submission/test_input.csv'));p.add_argument('--student-id',default='113700058');p.add_argument('--fixture-only',action='store_true');run(p.parse_args())
