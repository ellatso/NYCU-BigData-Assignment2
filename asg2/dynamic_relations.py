"""Windowed FK aggregation of observed author, repository and collaborator activity."""
import numpy as np
import pandas as pd
from .core import stamp,asof_events,PK,TIME,TARGETS

def dynamic_features(tables,events,base,origin):
    t=stamp(origin)-pd.Timedelta(hours=4);X=base.copy();u=tables['Users'];repos=tables['Repositories']
    idx=pd.Index(u.set_index('github_user_id').loc[base.UserID.to_numpy()].user_id.to_numpy(),name='user_id')
    valid_u=u.loc[u.first_seen_at<t,'user_id'];valid_r=repos.loc[repos.first_seen_at<t,'repo_id']
    e=asof_events(events,t);e=e[e.user_id.isin(valid_u)&e.repo_id.isin(valid_r)]
    e90=e[e.event_time>=t-pd.Timedelta(days=90)];e7=e[e.event_time>=t-pd.Timedelta(days=7)]
    values={};peers=[]
    # Activity received on a user's own PR/issue, with parent visibility checked.
    for name,foreign,parent,actor in [('IssueComments','issue_id','Issues','author_id'),
        ('PRReviews','pr_id','PullRequests','reviewer_id'),('PRReviewComments','pr_id','PullRequests','author_id')]:
        d=tables[name];occ='submitted_at' if name=='PRReviews' else 'created_at'
        d=d[(d[TIME[name]]<t)&(d[occ]<t)&(d[occ]>=t-pd.Timedelta(days=90))]
        p=tables[parent];p=p[(p[TIME[parent]]<t)&p.author_id.isin(valid_u)]
        z=d.merge(p[[PK[parent],'author_id']].rename(columns={'author_id':'owner_id'}),left_on=foreign,right_on=PK[parent],validate='many_to_one')
        z=z[z[actor].isin(valid_u)&z.repo_id.isin(valid_r)&z.owner_id.notna()&(z[actor]!=z.owner_id)]
        peers.append(z[[actor,'owner_id']].rename(columns={actor:'user_id','owner_id':'peer'}))
        peers.append(z[[actor,'owner_id']].rename(columns={actor:'peer','owner_id':'user_id'}))
        for days in [3,7]:
            w=z[z[occ]>=t-pd.Timedelta(days=days)]
            for metric,counts in [('received',w.groupby('owner_id').size()),('partners',w.groupby('owner_id')[actor].nunique())]:
                values[f'd_received_{days}_{name}_{metric}']=np.log1p(counts.reindex(idx,fill_value=0).to_numpy())
    # Repositories reviewed by the user, then newly opened PRs by others in 3 days.
    reviews=tables['PRReviews'];reviews=reviews[(reviews.first_seen_at<t)&(reviews.submitted_at<t)&(reviews.submitted_at>=t-pd.Timedelta(days=90))]
    reviews=reviews[reviews.reviewer_id.isin(valid_u)&reviews.repo_id.isin(valid_r)]
    membership=reviews[['reviewer_id','repo_id']].drop_duplicates().rename(columns={'reviewer_id':'user_id'})
    pr=tables['PullRequests'];pr=pr[(pr.first_seen_at<t)&(pr.created_at<t)&(pr.created_at>=t-pd.Timedelta(days=3))&pr.repo_id.isin(valid_r)]
    z=membership.merge(pr[['repo_id','author_id']],on='repo_id');z=z[z.user_id!=z.author_id]
    values['d_reviewed_repo_new_pr3']=np.log1p(z.groupby('user_id').size().reindex(idx,fill_value=0).to_numpy())
    # Known membership over 90d; neighbors' seven-day events exclude the user.
    membership=e90[['user_id','repo_id']].drop_duplicates()
    for kind in TARGETS:
        w=e7[e7.kind==kind];total=w.groupby('repo_id').size();own=w.groupby(['user_id','repo_id']).size()
        z=membership.copy();z['other']=z.repo_id.map(total).fillna(0)-pd.MultiIndex.from_frame(z).map(own).fillna(0)
        for agg in ['mean','max']:
            counts=z.groupby('user_id').other.agg(agg).reindex(idx,fill_value=0)
            values[f'd_repo7_{kind}_{agg}']=np.log1p(counts.to_numpy())
    # Direct collaborator activity is measured only for fully observed cohort peers.
    peer=pd.concat(peers,ignore_index=True).drop_duplicates()
    cohort=set(u.loc[u.is_cohort&u.first_seen_at.lt(t),'user_id']);peer=peer[peer.peer.isin(cohort)]
    values['d_cohort_collaborators90']=np.log1p(peer.groupby('user_id').peer.nunique().reindex(idx,fill_value=0).to_numpy())
    for kind in TARGETS:
        counts=e7[e7.kind==kind].groupby('user_id').size();z=peer.copy();z['activity']=z.peer.map(counts).fillna(0)
        values[f'd_peer7_{kind}']=np.log1p(z.groupby('user_id').activity.sum().reindex(idx,fill_value=0).to_numpy())
    return pd.concat([X,pd.DataFrame(values,index=X.index).astype('float32')],axis=1)
