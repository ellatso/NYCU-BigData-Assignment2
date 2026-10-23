"""As-of relational features using only the official eight-table release."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

TARGETS=['open_pr','open_issue','issue_comment','pr_review','pr_review_comment','push']
TABLES=['Users','Repositories','PullRequests','Issues','IssueComments','PRReviews','PRReviewComments','Pushes']
TIME={'Users':'first_seen_at','Repositories':'first_seen_at','PullRequests':'first_seen_at','Issues':'first_seen_at','PRReviews':'first_seen_at','IssueComments':'created_at','PRReviewComments':'created_at','Pushes':'created_at'}
# Known-only-later columns are deliberately excluded entirely, rather than guessed.
COLS={'Users':['user_id','github_user_id','is_cohort','first_seen_at'],
'Repositories':['repo_id','first_seen_at'],
'PullRequests':['pr_id','repo_id','author_id','created_at','first_seen_at'],
'Issues':['issue_id','repo_id','author_id','assignee_id','pull_request_id','is_pull_request','created_at','first_seen_at'],
'IssueComments':['comment_id','issue_id','repo_id','author_id','created_at'],
'PRReviews':['review_id','pr_id','repo_id','reviewer_id','submitted_at','first_seen_at'],
'PRReviewComments':['comment_id','pr_id','repo_id','author_id','pull_request_review_id','in_reply_to_id','created_at'],
'Pushes':['push_id','repo_id','user_id','created_at']}
PK={'Users':'user_id','Repositories':'repo_id','PullRequests':'pr_id','Issues':'issue_id','IssueComments':'comment_id','PRReviews':'review_id','PRReviewComments':'comment_id','Pushes':'push_id'}
FKS={'Users':{},'Repositories':{},'PullRequests':{'repo_id':'Repositories','author_id':'Users','merged_by_id':'Users'},'Issues':{'repo_id':'Repositories','author_id':'Users','assignee_id':'Users','pull_request_id':'PullRequests'},'IssueComments':{'issue_id':'Issues','repo_id':'Repositories','author_id':'Users'},'PRReviews':{'pr_id':'PullRequests','repo_id':'Repositories','reviewer_id':'Users'},'PRReviewComments':{'pr_id':'PullRequests','repo_id':'Repositories','author_id':'Users','pull_request_review_id':'PRReviews','in_reply_to_id':'PRReviewComments'},'Pushes':{'user_id':'Users','repo_id':'Repositories'}}

def stamp(x):
    t=pd.Timestamp(x)
    return t.tz_convert('UTC').tz_localize(None) if t.tzinfo is not None else t

def load_tables(data):
    base=Path(data)/'data/db'
    if not base.exists():base=Path(data)
    tables={t:pd.read_parquet(base/f'{t}.parquet',columns=COLS[t]) for t in TABLES}
    for t,d in tables.items():
        for c in d:
            if c.endswith('_at'):d[c]=pd.to_datetime(d[c])
        if d[PK[t]].duplicated().any():raise ValueError(f'Duplicate PK: {t}')
    return tables

def build_events(tables):
    specs=[('PullRequests','author_id','created_at','open_pr'),('Issues','author_id','created_at','open_issue'),('IssueComments','author_id','created_at','issue_comment'),('PRReviews','reviewer_id','submitted_at','pr_review'),('PRReviewComments','author_id','created_at','pr_review_comment'),('Pushes','user_id','created_at','push')]
    frames=[]
    for name,actor,event_time,kind in specs:
        d=tables[name]
        if name=='Issues':d=d[~d.is_pull_request.fillna(False)]
        f=pd.DataFrame({'user_id':d[actor],'repo_id':d.repo_id,'event_time':d[event_time],'available_at':d[TIME[name]],'kind':kind})
        f=f.dropna(subset=['user_id','repo_id','event_time','available_at'])
        f[['user_id','repo_id']]=f[['user_id','repo_id']].astype('int64')
        frames.append(f)
    events=pd.concat(frames,ignore_index=True)
    # Any row with availability later than occurrence must satisfy BOTH cutoffs.
    return events.sort_values('event_time',kind='stable').reset_index(drop=True)

def asof_events(events,t):
    return events[(events.event_time<t)&(events.available_at<t)]

def counts(events,index):
    return events.groupby(['user_id','kind']).size().unstack(fill_value=0).reindex(index=index,columns=TARGETS,fill_value=0).fillna(0)

def features(tables,events,origin,user_ids=None,gap_hours=4):
    origin=stamp(origin);t=origin-pd.Timedelta(hours=gap_hours)
    users=tables['Users'];u=users[(users.first_seen_at<t)&users.is_cohort].set_index('user_id')
    past=asof_events(events,t)
    # Enforce endpoint availability as well as event-row availability.
    valid_u=set(users.loc[users.first_seen_at<t,'user_id'])
    valid_r=set(tables['Repositories'].loc[tables['Repositories'].first_seen_at<t,'repo_id'])
    past=past[past.user_id.isin(valid_u)&past.repo_id.isin(valid_r)]
    recent90=past[past.event_time>=t-pd.Timedelta(days=90)]
    if user_ids is None:
        active=recent90.groupby('user_id').size()
        u=u.loc[u.index.intersection(active[active>=5].index)]
    else:
        ids=pd.Index(np.asarray(user_ids,dtype=np.int64))
        lookup=users.set_index('github_user_id')
        if not ids.is_unique:raise ValueError('Duplicate UserID in input')
        if not ids.isin(lookup.index).all():raise ValueError('Unknown official UserID')
        mapped=lookup.loc[ids]
        if not mapped.is_cohort.all() or not (mapped.first_seen_at<t).all():
            raise ValueError('Input contains non-cohort or not-yet-seen users')
        u=mapped.reset_index().set_index('user_id')
    idx=u.index;X=pd.DataFrame(index=idx)
    X['u_age_days']=(t-u.first_seen_at).dt.total_seconds()/86400
    life=counts(past,idx)
    for kind in TARGETS:X[f'u_lifetime_{kind}']=np.log1p(life[kind])
    windows={}
    for days in [7,14,30,90]:
        e=past[past.event_time>=t-pd.Timedelta(days=days)]
        c=counts(e,idx);windows[days]=c
        for kind in TARGETS:X[f't_count{days}_{kind}']=np.log1p(c[kind])
        X[f't_active_days{days}']=e.assign(day=e.event_time.dt.floor('D')).groupby('user_id').day.nunique().reindex(idx,fill_value=0)
        X[f't_repos{days}']=e.groupby('user_id').repo_id.nunique().reindex(idx,fill_value=0)
    for kind in TARGETS:
        last=past[past.kind==kind].groupby('user_id').event_time.max().reindex(idx)
        X[f't_recency_{kind}']=((t-last).dt.total_seconds()/86400).fillna(400).clip(upper=400)
        X[f't_trend_{kind}']=(windows[7][kind]+1)/(windows[30][kind]/4+1)
    # User -> repo -> users: degree and neighborhood activity, with own counts removed.
    for days,prefix in [(None,'r_static'),(30,'r_temporal')]:
        e=past if days is None else past[past.event_time>=t-pd.Timedelta(days=days)]
        edge=e.groupby(['user_id','repo_id']).size().rename('own').reset_index()
        if edge.empty:continue
        repo=e.groupby('repo_id').agg(total=('user_id','size'),contributors=('user_id','nunique'))
        edge=edge.join(repo,on='repo_id');edge['other']=edge.total-edge.own
        for field in ['other','contributors']:
            z=edge.groupby('user_id')[field].agg(['mean','max']).reindex(idx).fillna(0)
            for stat in ['mean','max']:X[f'{prefix}_{field}_{stat}']=np.log1p(z[stat])
        X[f'{prefix}_degree']=np.log1p(edge.groupby('user_id').repo_id.nunique().reindex(idx,fill_value=0))
        X[f'{prefix}_concentration']=edge.groupby('user_id').own.max().reindex(idx,fill_value=0)/edge.groupby('user_id').own.sum().reindex(idx,fill_value=1)
        for kind in TARGETS:
            totals=e[e.kind==kind].groupby('repo_id').size()
            own=e[e.kind==kind].groupby(['user_id','repo_id']).size()
            z=edge[['user_id','repo_id']].copy()
            z['other_kind']=z.repo_id.map(totals).fillna(0)-pd.MultiIndex.from_frame(z).map(own).fillna(0)
            X[f'{prefix}_neighbor_{kind}']=np.log1p(z.groupby('user_id').other_kind.mean().reindex(idx,fill_value=0))
    # PR/issue author <- discussion/review actor: interaction beyond shared repos.
    for name,foreign,parent,actor in [('IssueComments','issue_id','Issues','author_id'),('PRReviews','pr_id','PullRequests','reviewer_id'),('PRReviewComments','pr_id','PullRequests','author_id')]:
        d=tables[name];parent_rows=tables[parent]
        p=parent_rows[parent_rows[TIME[parent]]<t][[PK[parent],'author_id']].rename(columns={'author_id':'owner_id'})
        d=d[d[TIME[name]]<t].merge(p,left_on=foreign,right_on=PK[parent],how='inner',validate='many_to_one')
        d=d[d[actor]!=d.owner_id]
        for days,prefix in [(None,'r_static'),(30,'r_temporal')]:
            e=d if days is None else d[d[TIME[name]]>=t-pd.Timedelta(days=days)]
            X[f'{prefix}_received_{name}']=np.log1p(e.groupby('owner_id').size().reindex(idx,fill_value=0))
            X[f'{prefix}_partners_{name}']=np.log1p(e.groupby('owner_id')[actor].nunique().reindex(idx,fill_value=0))
    X=X.replace([np.inf,-np.inf],np.nan).fillna(0).astype('float32')
    X.insert(0,'UserID',u.github_user_id.astype('int64'))
    X.insert(1,'origin',origin)
    return X.reset_index(drop=True)

def labels(tables,events,origin,ids,horizon_days=7):
    origin=stamp(origin);end=origin+pd.Timedelta(days=horizon_days)
    # Historical labels must be available by the END of their outcome window.
    e=events[(events.event_time>=origin)&(events.event_time<end)&(events.available_at<end)]
    e=e.merge(tables['Users'][['user_id','github_user_id']],on='user_id',validate='many_to_one')
    c=e.groupby(['github_user_id','kind']).size().unstack(fill_value=0)
    return c.reindex(index=ids,columns=TARGETS,fill_value=0).fillna(0).gt(0).astype('int8').to_numpy()

def score(y,p):
    y=np.asarray(y);p=np.asarray(p)
    tp=((y==1)&(p==1)).sum(0);fp=((y==0)&(p==1)).sum(0);fn=((y==1)&(p==0)).sum(0)
    den=2*tp+fp+fn;per=np.divide(2*tp,den,out=np.ones(6),where=den!=0)
    total=den.sum();micro=float(2*tp.sum()/total) if total else 1.0
    return {'macro_f1':float(per.mean()),'micro_f1':micro,'score':float((per.mean()+micro)/2),**dict(zip(TARGETS,map(float,per)))}

def tune_thresholds(y,prob):
    thresholds=np.full(6,.5)
    grid=np.linspace(.05,.75,29)
    # Optimize the official combined score, rather than six F1 scores separately.
    for _ in range(3):
        for j in range(6):
            candidates=[]
            for v in grid:
                t=thresholds.copy();t[j]=v
                candidates.append(score(y,prob>=t)['score'])
            thresholds[j]=grid[int(np.argmax(candidates))]
    return thresholds

def feature_columns(frame,variant):
    names=[c for c in frame if c not in ['UserID','origin']]
    return [c for c in names if c.startswith('u_') or (variant in ['temporal','full'] and c.startswith('t_')) or (variant in ['relational','full'] and c.startswith('r_static_')) or (variant=='full' and c.startswith('r_temporal_'))]
