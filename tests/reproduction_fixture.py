"""Artificial hourly events for exercising the full official rebuild command.

The actors are public cohort identifiers only so that the unchanged official
filter can run. Their activity here is GENERATED, not observed GitHub data.
"""
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from asg2.provenance import sha256

def create(root):
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True);records={};serial=1
    actors=[989,1900,2016,2071,2655,6470068]
    for t in pd.date_range('2026-09-01','2026-11-07',freq='h',inclusive='left'):
        name=t.strftime('%Y-%m-%d-')+str(t.hour)+'.json.gz';lines=[]
        if t.hour==12:
            day=(t-pd.Timestamp('2026-09-01')).days
            for k,uid in enumerate(actors):
                when=(t+pd.Timedelta(minutes=k)).strftime('%Y-%m-%dT%H:%M:%SZ')
                actor=dict(id=uid,login='synthetic_fixture')
                e=dict(id=str(serial),actor=actor,repo=dict(id=1234,name='fixture/repo'),created_at=when)
                issue=dict(id=500000+day,number=day+1,user=dict(id=1900,login='fixture_owner'),created_at=when,title='fixture',body='fixture')
                pr=dict(id=800000+day,number=day+1,head=dict(ref='fixture',sha='a'*40),base=dict(ref='main',sha='b'*40))
                if k==0:e.update(type='PushEvent',payload=dict(push_id=serial,ref='refs/heads/fixture'))
                elif k==1:e.update(type='IssuesEvent',payload=dict(action='opened',issue=issue))
                elif k==2:e.update(type='IssueCommentEvent',payload=dict(action='created',issue=issue,comment=dict(id=serial,user=actor,created_at=when,body='fixture')))
                elif k==3:e.update(type='PullRequestEvent',payload=dict(action='opened',pull_request=pr))
                elif k==4:e.update(type='PullRequestReviewEvent',payload=dict(action='submitted',pull_request=pr,review=dict(id=900000+day,user=actor,submitted_at=when,state='approved')))
                else:e.update(type='PullRequestReviewCommentEvent',payload=dict(action='created',pull_request=pr,comment=dict(id=serial,user=actor,created_at=when,pull_request_review_id=900000+day,path='fixture.py')))
                lines.append(json.dumps(e,separators=(',',':')).encode()+b'\n');serial+=1
        p=raw/name
        with p.open('wb') as f:
            with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.writelines(lines)
        records[name]=dict(hour_utc=t.tz_localize('UTC').isoformat(),status='ok',storage='full',stored_sha256=sha256(p),raw_sha256=sha256(p),url='synthetic://'+name)
    (raw/'download_manifest.json').write_text(json.dumps(dict(dataset_kind='synthetic_fixture',files=records),indent=2))
    config=json.loads((Path(__file__).resolve().parents[1]/'config/revised.json').read_text());config['trees']=3
    (root/'config.json').write_text(json.dumps(config))
    print(root)

if __name__=='__main__':create(sys.argv[1])
