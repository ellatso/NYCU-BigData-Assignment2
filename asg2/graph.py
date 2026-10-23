"""Typed temporal user/repo projection from the official eight relational tables."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from .core import TARGETS,TIME,PK,stamp,asof_events,labels
GRAPH_VERSION='temporal-projected-rgnn-v1'

def adjacency(rows,cols,weights,nrow,ncol):
    rows=np.asarray(rows,dtype=np.int64);cols=np.asarray(cols,dtype=np.int64);w=np.asarray(weights,dtype=np.float32)
    denom=np.bincount(rows,weights=w,minlength=nrow)
    v=(w/np.maximum(denom[rows],1e-12)).astype(np.float32)
    return torch.sparse_coo_tensor(torch.as_tensor(np.stack([rows,cols])),torch.as_tensor(v),(nrow,ncol)).coalesce()

def node_features(past,ids,key,t):
    X=pd.DataFrame(index=ids)
    for days in [7,30,90]:
        e=past[past.event_time>=t-pd.Timedelta(days=days)]
        c=e.groupby([key,'kind']).size().unstack(fill_value=0).reindex(index=ids,columns=TARGETS,fill_value=0).fillna(0)
        for k in TARGETS:X[f'count{days}_{k}']=np.log1p(c[k])/6
    for k in TARGETS:
        last=past[past.kind==k].groupby(key).event_time.max().reindex(ids)
        X['recency_'+k]=((t-last).dt.total_seconds()/86400).fillna(90).clip(0,90)/90
    return X.fillna(0).to_numpy(dtype=np.float32)

def graph_snapshot(tables,events,origin,with_labels=True):
    origin=stamp(origin);t=origin-pd.Timedelta(hours=4)
    u=tables['Users'];u=u[u.first_seen_at<t].sort_values('github_user_id').set_index('user_id')
    r=tables['Repositories'];r=r[r.first_seen_at<t].set_index('repo_id')
    past=asof_events(events,t)
    past=past[(past.event_time>=t-pd.Timedelta(days=90))&past.user_id.isin(u.index)&past.repo_id.isin(r.index)]
    r=r.loc[pd.Index(sorted(past.repo_id.unique()))]
    uid=pd.Series(np.arange(len(u)),index=u.index);rid=pd.Series(np.arange(len(r)),index=r.index)
    xu=node_features(past,u.index,'user_id',t);xr=node_features(past,r.index,'repo_id',t)
    xu=np.column_stack([xu,u.is_cohort.astype(float),((t-u.first_seen_at).dt.total_seconds()/86400).clip(0,400)/400]).astype('float32')
    xr=np.column_stack([xr,((t-r.first_seen_at).dt.total_seconds()/86400).clip(0,400)/400]).astype('float32')
    ur=[];ru=[];edge_counts={}
    for kind in TARGETS:
        e=past[past.kind==kind].copy();e['w']=np.exp(-(t-e.event_time).dt.total_seconds()/86400/30)
        g=e.groupby(['user_id','repo_id']).w.sum().reset_index()
        us=g.user_id.map(uid).to_numpy(dtype=np.int64);rs=g.repo_id.map(rid).to_numpy(dtype=np.int64);w=g.w.to_numpy()
        ur.append(adjacency(us,rs,w,len(u),len(r)));ru.append(adjacency(rs,us,w,len(r),len(u)));edge_counts[kind]=len(w)
    peers=[]
    specs=[('IssueComments','issue_id','Issues','author_id','created_at'),('PRReviews','pr_id','PullRequests','reviewer_id','submitted_at'),('PRReviewComments','pr_id','PullRequests','author_id','created_at')]
    for name,fk,parent,actor,when in specs:
        d=tables[name];p=tables[parent]
        d=d[(d[TIME[name]]<t)&(d[when]<t)&(d[when]>=t-pd.Timedelta(days=90))]
        p=p[p[TIME[parent]]<t][[PK[parent],'author_id']].rename(columns={'author_id':'owner'})
        e=d.merge(p,left_on=fk,right_on=PK[parent],how='inner',validate='many_to_one')
        e=e[e[actor].isin(u.index)&e.owner.isin(u.index)&e[actor].ne(e.owner)].copy()
        e['w']=np.exp(-(t-e[when]).dt.total_seconds()/86400/30)
        g=e.groupby([actor,'owner']).w.sum().reset_index()
        sender=g[actor].map(uid).to_numpy(dtype=np.int64);receiver=g.owner.map(uid).to_numpy(dtype=np.int64);w=g.w.to_numpy()
        peers.extend([adjacency(receiver,sender,w,len(u),len(u)),adjacency(sender,receiver,w,len(u),len(u))]);edge_counts[name]=len(w)
    active=past.groupby('user_id').size().reindex(u.index,fill_value=0)
    selected=np.flatnonzero((u.is_cohort&(active>=5)).to_numpy());ids=u.github_user_id.to_numpy(dtype=np.int64)
    y=labels(tables,events,origin,ids[selected]) if with_labels else np.zeros((len(selected),6),np.int8)
    return {'xu':torch.from_numpy(xu),'xr':torch.from_numpy(xr),'ur':ur,'ru':ru,'peers':peers,'selected':torch.as_tensor(selected),'ids':torch.as_tensor(ids),'y':torch.as_tensor(y,dtype=torch.float32),'origin':str(origin),'cutoff':str(t),'edge_counts':edge_counts}

class RelationalNet(nn.Module):
    def __init__(self,user_dim,repo_dim,hidden=32,use_graph=True):
        super().__init__();self.use_graph=use_graph
        self.user_enc=nn.Sequential(nn.Linear(user_dim,hidden),nn.ReLU());self.repo_enc=nn.Sequential(nn.Linear(repo_dim,hidden),nn.ReLU())
        self.repo_self=nn.Linear(hidden,hidden);self.user_self=nn.Linear(hidden,hidden)
        self.to_repo=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in TARGETS]);self.to_user=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in TARGETS])
        self.peer=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(6)])
        self.drop=nn.Dropout(.15);self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(.15),nn.Linear(hidden,6));self.skip=nn.Linear(user_dim,6)
    def forward(self,g):
        u=self.user_enc(g['xu']);h=self.user_self(u)
        if self.use_graph:
            r=self.repo_enc(g['xr']);rh=self.repo_self(r)
            for a,layer in zip(g['ru'],self.to_repo):rh=rh+torch.sparse.mm(a,layer(u))/6
            rh=self.drop(torch.relu(rh))
            for a,layer in zip(g['ur'],self.to_user):h=h+torch.sparse.mm(a,layer(rh))/6
            for a,layer in zip(g['peers'],self.peer):h=h+torch.sparse.mm(a,layer(u))/6
        return self.head(self.drop(torch.relu(h)))+self.skip(g['xu'])

def to_device(g,device):
    return {k:v.to(device) if torch.is_tensor(v) else [x.to(device) for x in v] if k in ['ur','ru','peers'] else v for k,v in g.items()}
