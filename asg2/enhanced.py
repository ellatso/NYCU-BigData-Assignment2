"""Second-round features: short-term rhythm and ongoing shared conversations."""
import numpy as np
import pandas as pd
from .core import TARGETS, TIME, PK, stamp, asof_events, counts

def enrich(tables, events, base, origin, gap_hours=4):
    t=stamp(origin)-pd.Timedelta(hours=gap_hours)
    users=tables['Users'];repos=tables['Repositories']
    ids=users.set_index('github_user_id').loc[base.UserID.to_numpy()].user_id.to_numpy()
    idx=pd.Index(ids,name='user_id'); X=base.copy()
    past=asof_events(events,t)
    past=past[past.user_id.isin(users.loc[users.first_seen_at<t,'user_id']) &
              past.repo_id.isin(repos.loc[repos.first_seen_at<t,'repo_id'])]
    e=past[past.event_time>=t-pd.Timedelta(days=56)].copy()
    age=(t-e.event_time).dt.total_seconds()/86400
    week=(age//7).astype(int)
    for k in TARGETS:
        z=e[e.kind==k].copy();a=(t-z.event_time).dt.total_seconds()/86400
        for days in [1,3]:
            c=z[a<days].groupby('user_id').size().reindex(idx,fill_value=0)
            X[f'v2_count{days}_{k}']=np.log1p(c.to_numpy())
        z['week']=(a//7).astype(int)
        w=z.groupby(['user_id','week']).size().unstack(fill_value=0).reindex(index=idx,columns=range(8),fill_value=0).fillna(0)
        active=w.to_numpy()>0
        X[f'v2_active_weeks8_{k}']=active.sum(1)
        X[f'v2_streak_weeks_{k}']=np.cumprod(active,axis=1).sum(1)
        X[f'v2_previous_week_{k}']=np.log1p(w[1].to_numpy())
        for half in [7,28]:
            weighted=z.assign(weight=np.exp2(-a/half)).groupby('user_id').weight.sum().reindex(idx,fill_value=0)
            X[f'v2_decay{half}_{k}']=np.log1p(weighted.to_numpy())
        ordered=z.sort_values('event_time')
        intervals=ordered.groupby('user_id').event_time.diff().dt.total_seconds()/86400
        stats=ordered.assign(interval=intervals).groupby('user_id').interval.agg(['mean','std']).reindex(idx).fillna(56)
        X[f'v2_interval_mean_{k}']=stats['mean'].clip(upper=56).to_numpy()
        X[f'v2_interval_std_{k}']=stats['std'].clip(upper=56).to_numpy()
    # For threads personally touched in the past 14d, count recent OTHER actors.
    # Parent and event availability are both checked; no closed/merged metadata.
    for name,key,actor,parent in [('IssueComments','issue_id','author_id','Issues'),
                                  ('PRReviews','pr_id','reviewer_id','PullRequests'),
                                  ('PRReviewComments','pr_id','author_id','PullRequests')]:
        d=tables[name];parent_rows=tables[parent]
        occurrence='submitted_at' if name=='PRReviews' else 'created_at'
        d=d[(d[TIME[name]]<t)&(d[occurrence]<t)&(d[occurrence]>=t-pd.Timedelta(days=14))]
        d=d[d[key].isin(parent_rows.loc[parent_rows[TIME[parent]]<t,PK[parent]]) &
            d[actor].isin(users.loc[users.first_seen_at<t,'user_id']) &
            d.repo_id.isin(repos.loc[repos.first_seen_at<t,'repo_id'])]
        own=d[[actor,key]].drop_duplicates().rename(columns={actor:'user_id'})
        for days in [3,7]:
            recent=d[d[occurrence]>=t-pd.Timedelta(days=days)]
            joined=own.merge(recent[[actor,key]],on=key)
            joined=joined[joined.user_id!=joined[actor]]
            n=joined.groupby('user_id').size().reindex(idx,fill_value=0)
            threads=joined.groupby('user_id')[key].nunique().reindex(idx,fill_value=0)
            X[f'v2r_thread_other{days}_{name}']=np.log1p(n.to_numpy())
            X[f'v2r_live_threads{days}_{name}']=np.log1p(threads.to_numpy())
    return X.replace([np.inf,-np.inf],0).fillna(0)

def suspect_push_days(events):
    """Heuristic, never ground truth: zero pushes amid other recorded activity."""
    e=events.assign(day=(events.event_time+pd.Timedelta(hours=8)).dt.floor('D'))
    c=e.groupby(['day','kind']).size().unstack(fill_value=0)
    c=c.reindex(pd.date_range(c.index.min(),c.index.max()),fill_value=0).reindex(columns=TARGETS,fill_value=0)
    reference=c.push.shift(1).rolling(28,min_periods=14).median()
    return c.index[(c.push==0)&(c.drop(columns='push').sum(axis=1)>0)&(reference>=100)]

def push_training_mask(origins, suspect_days):
    local=(pd.to_datetime(origins)+pd.Timedelta(hours=8)).dt.normalize()
    bad=np.zeros(len(local),dtype=bool)
    for day in suspect_days:bad|=((local<=day)&(day<local+pd.Timedelta(days=7))).to_numpy()
    return ~bad
