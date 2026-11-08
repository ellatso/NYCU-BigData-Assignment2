"""Synthetic contract tests, never presented as real model accuracy."""
import gzip
import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from asg2.core import TABLES,load_tables,build_events
from asg2.provenance import VENDOR,verify_vendor,verify_data,sha256,integrity
from asg2.revised import make_frame,fit,probability,columns,hour_coverage,raw_bounds

def official():
    s=importlib.util.spec_from_file_location('official_fixture',VENDOR/'scripts/preprocess.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture_tables(path):
    """Known test actors are generated locally; this is not course source data."""
    m=official();cohort={m.code_int('user',i) for i in range(100,160)}
    builder=m.TableBuilder();serial=1
    for day in pd.date_range('2026-09-01','2026-11-06'):
        for uid in range(100,160):
            if (day.day+uid)%3==0:continue
            stamp=(day+pd.Timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
            actor={'id':uid,'login':f'fixture{uid}'};repo={'id':20+uid%2,'name':'fixture/repo'}
            base=dict(id=str(serial),created_at=stamp,actor=actor,repo=repo)
            k=(uid-100)%6;group=(uid-100)//6;dayno=(day-pd.Timestamp('2026-09-01')).days
            base['repo']=dict(id=20+group%2,name='fixture/repo')
            issueid=500000+dayno*100+group;prid=800000+dayno*100+group;reviewid=900000+dayno*100+group
            issue=dict(id=issueid,number=group+1,user={'id':101+group*6,'login':'fixture_owner'},created_at=stamp,body='fixture',title='fixture')
            pr=dict(id=prid,number=group+1,head=dict(ref='test',sha='a'*40),base=dict(ref='main',sha='b'*40))
            if k==0:
                e=dict(base,type='PushEvent',payload=dict(push_id=serial,ref='refs/heads/test',before='0'*40))
            elif k==1:
                e=dict(base,type='IssuesEvent',payload=dict(action='opened',issue=issue))
            elif k==2:
                e=dict(base,type='IssueCommentEvent',payload=dict(action='created',issue=issue,comment=dict(id=serial,user=actor,created_at=stamp,body='fixture')))
            elif k==3:
                e=dict(base,type='PullRequestEvent',payload=dict(action='opened',pull_request=pr))
            elif k==4:
                e=dict(base,type='PullRequestReviewEvent',payload=dict(action='submitted',pull_request=pr,review=dict(id=reviewid,user=actor,submitted_at=stamp,state='approved',body='fixture')))
            else:
                e=dict(base,type='PullRequestReviewCommentEvent',payload=dict(action='created',pull_request=pr,comment=dict(id=serial,user=actor,created_at=stamp,body='fixture',pull_request_review_id=reviewid,path='fixture.py')))
            builder.add(e);serial+=1
    builder.finish();frames=builder.to_frames(cohort);tables=m.anonymize(frames);path=Path(path);path.mkdir(parents=True,exist_ok=True);m.write_output(path,tables)
    records={}
    for t in pd.date_range('2026-09-01','2026-11-07',freq='h',inclusive='left'):
        records[t.strftime('%Y-%m-%d-')+str(t.hour)+'.json.gz']=dict(status='ok',hour_utc=t.isoformat())
    meta=dict(dataset_kind='synthetic_fixture',raw_manifest=dict(files=records),table_sha256={n:sha256(path/'db'/f'{n}.parquet') for n in TABLES})
    (path/'provenance.json').write_text(json.dumps(meta));return tables,meta

class RevisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name);fixture_tables(cls.path)
        cls.tables=load_tables(cls.path);cls.events=build_events(cls.tables)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_pinned_official_sources_are_unchanged(self):verify_vendor()
    def test_new_db_layout_and_anonymous_ids(self):
        self.assertTrue(self.tables['Users'].github_user_id.between(10**14,10**15-1).all())
        self.assertTrue(all(x['passed'] for x in integrity(self.path)))
    def test_fixture_cannot_be_misused_as_submission_data(self):
        with self.assertRaises(ValueError):verify_data(self.path)
    def test_unmodified_official_cli_preserves_fixture_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw';raw.mkdir();name='2026-09-01-12.json.gz';p=raw/name
            e=dict(id='1',type='PushEvent',actor=dict(id=989,login='fixture'),repo=dict(id=1,name='fixture/repo'),created_at='2026-09-01T12:00:00Z',payload=dict(push_id=1,ref='refs/heads/x'))
            with gzip.open(p,'wb') as f:f.write(json.dumps(e,separators=(',',':')).encode()+b'\n')
            meta=dict(dataset_kind='synthetic_fixture',files={name:dict(status='ok',hour_utc='2026-09-01T12:00:00+00:00',stored_sha256=sha256(p))})
            (raw/'download_manifest.json').write_text(json.dumps(meta))
            subprocess.run([sys.executable,str(VENDOR.parents[1]/'scripts/prepare_revised.py'),'build','--raw',str(raw),'--out',str(root/'data')],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(json.loads((root/'data/provenance.json').read_text())['dataset_kind'],'synthetic_fixture')
            self.assertTrue(all((root/'data/db'/f'{n}.parquet').exists() for n in TABLES))
    def test_data_checksums_detect_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);meta=dict(dataset_kind='official_gharchive',official_commit=verify_vendor()['commit'],table_sha256={'Users':'wrong'})
            (p/'db').mkdir();(p/'db/Users.parquet').write_bytes(b'wrong');(p/'provenance.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):verify_data(p)
    def test_asof_features_ignore_future_rows_and_stable_ids_preserve_order(self):
        ids=self.tables['Users'].github_user_id.iloc[:3].to_numpy()[::-1]
        X=make_frame(self.tables,self.events,'2026-10-01',ids)
        future=self.events.iloc[[0]].copy();future['event_time']=pd.Timestamp('2026-12-01');future['available_at']=pd.Timestamp('2026-12-01')
        Y=make_frame(self.tables,pd.concat([self.events,future],ignore_index=True),'2026-10-01',ids)
        pd.testing.assert_frame_equal(X,Y);self.assertEqual(X.UserID.tolist(),ids.tolist())
    def test_unseen_ids_are_kept_out_of_visible_graph(self):
        ids=np.r_[self.tables['Users'].github_user_id.iloc[0],999999999999999]
        X=make_frame(self.tables,self.events,'2026-10-01',ids)
        self.assertEqual(len(X),1)
    def test_relationship_ablation_only_adds_intended_columns(self):
        X=make_frame(self.tables,self.events,'2026-10-01')
        baseline=set(columns(X,'temporal30'));rel=set(columns(X,'dynamic_relations'))
        self.assertTrue(baseline<rel);self.assertTrue(all(c.startswith('d_') for c in rel-baseline))
        self.assertEqual(len([c for c in columns(X,'no_recency') if c.startswith('t_recency_')]),0)
    def test_two_independent_fits_give_identical_probabilities(self):
        X=make_frame(self.tables,self.events,'2026-10-01');cols=columns(X,'temporal30')
        y=np.tile(np.arange(len(X))[:,None]%2,(1,6))
        a=fit(X,y,cols,trees=3);b=fit(X,y,cols,trees=3)
        np.testing.assert_array_equal(probability(a,X,cols),probability(b,X,cols))
    def test_official_cohort_prefilter_is_semantically_equivalent(self):
        m=official();cohort={m.code_int('user',100)};m._init_worker(cohort)
        def e(uid,num):return dict(id=str(num),type='PushEvent',actor=dict(id=uid,login='fixture'),repo=dict(id=1,name='x/y'),created_at='2026-09-01T12:00:00Z',payload=dict(push_id=num,ref='refs/heads/x'))
        lines=[json.dumps(e(100,1),separators=(',',':')).encode(),json.dumps(e(999,2),separators=(',',':')).encode()]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'2026-09-01-12.json.gz'
            with gzip.open(path,'wb') as f:f.write(b'\n'.join(lines)+b'\n')
            kept=m.extract_hour(path)
            with gzip.open(path,'wb') as f:f.write(b'\n'.join(kept)+b'\n')
            again=m.extract_hour(path)
            self.assertEqual(kept,again);self.assertEqual(len(kept),1)
            full_tables=m.anonymize(m.build_tables([kept],cohort))
            filtered_tables=m.anonymize(m.build_tables([again],cohort))
            for table in TABLES:pd.testing.assert_frame_equal(full_tables[table],filtered_tables[table])

if __name__=='__main__':unittest.main()
