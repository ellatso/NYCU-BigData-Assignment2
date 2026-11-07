"""Download GH Archive, optionally retain the exact official cohort filter, then run the unchanged official builder."""
import argparse
import concurrent.futures
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from asg2.core import TABLES
from asg2.provenance import VENDOR,verify_vendor,sha256,integrity

def official_module():
    spec=importlib.util.spec_from_file_location('official_preprocess',VENDOR/'scripts/preprocess.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m._init_worker(m.load_cohort(VENDOR/'scripts/cohort.txt'))
    return m

def hours(start,end):
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:raise ValueError('UTC start/end must include a timezone')
    start=start.tz_convert('UTC');end=end.tz_convert('UTC')
    if start!=start.floor('h') or end!=end.floor('h') or end<=start:raise ValueError('Use increasing whole UTC hours; end is exclusive')
    if end>pd.Timestamp.now(tz='UTC').floor('h'):raise ValueError('Cannot download unfinished future hours')
    return pd.date_range(start,end,freq='h',inclusive='left')

def download(a):
    verify_vendor();raw=Path(a.raw);raw.mkdir(parents=True,exist_ok=True)
    plan=hours(a.start,a.end);old_path=raw/'download_manifest.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else {'files':{}}
    entries={n:r for n,r in old['files'].items() if stamp_hour(r['hour_utc'])>=plan[0] and stamp_hour(r['hour_utc'])<plan[-1]+pd.Timedelta(hours=1)};module=official_module()
    def one(t):
        name=t.strftime('%Y-%m-%d-')+str(t.hour)+'.json.gz';dest=raw/name
        previous=entries.get(name)
        if previous and previous.get('status')=='ok' and previous.get('storage')==a.storage and dest.exists() and sha256(dest)==previous.get('stored_sha256'):
            return name,previous
        record=dict(hour_utc=t.isoformat(),url='https://data.gharchive.org/'+name,status='failed')
        fd,tmp=tempfile.mkstemp(prefix='.download_',dir=raw);os.close(fd)
        try:
            with urllib.request.urlopen(record['url'],timeout=90) as src,open(tmp,'wb') as f:
                while True:
                    block=src.read(1024*1024)
                    if not block:break
                    f.write(block)
            record.update(raw_sha256=sha256(tmp),raw_bytes=Path(tmp).stat().st_size)
            # Validate gzip CRC/end before retaining anything. Missing/truncated
            # files are documented, not silently considered observed inactivity.
            with gzip.open(tmp,'rb') as f:
                for _ in f:pass
            if a.storage=='cohort':
                lines=module.extract_hour(tmp)
                with open(str(tmp)+'.filtered','wb') as f:
                    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:
                        for line in lines:z.write(line+b'\n')
                os.replace(str(tmp)+'.filtered',tmp)
                record['retained_cohort_lines']=len(lines)
            os.replace(tmp,dest)
            record.update(status='ok',storage=a.storage,stored_sha256=sha256(dest),stored_bytes=dest.stat().st_size)
        except urllib.error.HTTPError as e:
            record.update(status='missing' if e.code==404 else 'blocked',http_status=e.code)
        except (OSError,EOFError,ValueError,zlib.error) as e:record['error']=type(e).__name__+': '+str(e)[:180]
        finally:
            for p in [Path(tmp),Path(str(tmp)+'.filtered')]:
                if p.exists():p.unlink()
        return name,record
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        it=iter(plan);running={}
        for _ in range(min(a.workers,len(plan))):
            t=next(it,None)
            if t is not None:running[pool.submit(one,t)]=t
        blocked=False;done=0
        while running:
            completed,_=concurrent.futures.wait(running,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in completed:
                running.pop(f);name,r=f.result();entries[name]=r;done+=1
                manifest=dict(source='https://www.gharchive.org/',start_utc=plan[0].isoformat(),end_utc=(plan[-1]+pd.Timedelta(hours=1)).isoformat(),official_commit=verify_vendor()['commit'],files=entries)
                old_path.write_text(json.dumps(manifest,indent=2))
                print(done,'/',len(plan),name,r['status'],flush=True)
                if r['status']=='blocked':blocked=True
                t=None if blocked else next(it,None)
                if t is not None:running[pool.submit(one,t)]=t
    if blocked:raise SystemExit('Download rejected by the source. Remaining hours were not attempted. Use the supplied notebook in your own authorized runtime; no data or model score is fabricated.')

def build(a):
    pin=verify_vendor();raw=Path(a.raw);mp=raw/'download_manifest.json'
    if not mp.exists():raise SystemExit('Download manifest missing. Run download first to document original source hours.')
    source=json.loads(mp.read_text());files=source['files'];valid={}
    for name,r in files.items():
        if r.get('status')=='ok':
            if not (raw/name).exists() or sha256(raw/name)!=r['stored_sha256']:raise ValueError('Raw input checksum mismatch: '+name)
            valid[name]=r
    if not valid:raise SystemExit('No successfully downloaded source hours. No tables were created.')
    out=Path(a.out)
    if (out/'db').exists():raise ValueError('Use a NEW output directory; existing tables are never mixed with another release')
    out.mkdir(parents=True,exist_ok=True)
    # Select exactly the manifest's successful hours. Cached hours outside this
    # chosen period cannot leak into an earlier reconstructed dataset.
    with tempfile.TemporaryDirectory(prefix='asg2-selected-hours-') as td:
        selected=Path(td)
        for name in valid:(selected/name).symlink_to((raw/name).resolve())
        subprocess.run([sys.executable,str(VENDOR/'scripts/preprocess.py'),'--raw',str(selected),'--out',str(out)],check=True)
    checks=integrity(out)
    if not all(r['passed'] for r in checks):raise ValueError('Official-table integrity check failed')
    kind=source.get('dataset_kind','official_gharchive')
    if kind not in ['official_gharchive','synthetic_fixture']:raise ValueError('Unknown input provenance scope')
    meta=dict(dataset_kind=kind,data_source='https://www.gharchive.org/' if kind=='official_gharchive' else 'synthetic fixture, not observed activity',official_commit=pin['commit'],
        preprocessing_sha256=pin['files']['scripts/preprocess.py'],cohort_sha256=pin['files']['scripts/cohort.txt'],
        raw_manifest=source,table_sha256={n:sha256(out/'db'/f'{n}.parquet') for n in TABLES},integrity_checks=checks,
        warning='File integrity and hour coverage do not establish completeness of each activity in GH Archive.')
    (out/'provenance.json').write_text(json.dumps(meta,indent=2));print(out/'provenance.json')

def stamp_hour(value):return pd.Timestamp(value).tz_convert('UTC')

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    d=sub.add_parser('download');d.add_argument('--raw',required=True);d.add_argument('--start',required=True);d.add_argument('--end',required=True);d.add_argument('--workers',type=int,default=4);d.add_argument('--storage',choices=['cohort','full'],default='cohort')
    b=sub.add_parser('build');b.add_argument('--raw',required=True);b.add_argument('--out',required=True)
    a=p.parse_args();download(a) if a.action=='download' else build(a)
