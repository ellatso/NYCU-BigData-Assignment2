"""Verify the revised official builder and provenance of its newly generated tables."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from .core import TABLES,PK,FKS

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/assignment2'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_vendor(vendor=VENDOR):
    vendor=Path(vendor);pin=json.loads((vendor/'PIN.json').read_text())
    for name,want in pin['files'].items():
        if sha256(vendor/name)!=want:raise ValueError('Pinned official file changed: '+name)
    return pin

def verify_data(data):
    root=Path(data);p=root/'provenance.json'
    if not p.exists():raise ValueError('Revised data provenance missing; run scripts/prepare_revised.py build. Legacy data are not accepted.')
    meta=json.loads(p.read_text());pin=verify_vendor()
    if meta.get('official_commit')!=pin['commit']:raise ValueError('Data were built with another official preprocessing version')
    if meta.get('dataset_kind')!='official_gharchive':raise ValueError('Fixture data cannot produce a submission or real experiment claims')
    for table,want in meta['table_sha256'].items():
        if sha256(root/'db'/f'{table}.parquet')!=want:raise ValueError('Prepared table changed: '+table)
    return meta

def integrity(data):
    frames={n:pd.read_parquet(Path(data)/'db'/f'{n}.parquet') for n in TABLES};checks=[]
    for table,d in frames.items():
        checks.append(dict(table=table,check='unique_primary_key',passed=bool(d[PK[table]].is_unique)))
        for col,parent in FKS[table].items():
            if col in d:
                checks.append(dict(table=table,check=f'{col}->{parent}',passed=bool(d[col].dropna().isin(frames[parent][PK[parent]]).all())))
    u=frames['Users'];checks.append(dict(table='Users',check='anonymous_id_range',passed=bool(u.github_user_id.between(10**14,10**15-1).all())))
    return checks
