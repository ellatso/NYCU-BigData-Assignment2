"""Coverage-aware diagnostics and push fallback. No unobserved label imputation."""
import numpy as np
import pandas as pd
from .core import TARGETS,stamp,asof_events

def score_subset(y,p):
    y=np.asarray(y);p=np.asarray(p)
    tp=((y==1)&(p==1)).sum(0);fp=((y==0)&(p==1)).sum(0);fn=((y==1)&(p==0)).sum(0)
    den=2*tp+fp+fn;f=np.divide(2*tp,den,out=np.ones(y.shape[1]),where=den!=0)
    micro=2*tp.sum()/den.sum() if den.sum() else 1.
    return dict(macro_f1=float(f.mean()),micro_f1=float(micro),score=float((f.mean()+micro)/2))

def tune_mean_thresholds(y,p,folds):
    t=np.full(6,.5);folds=np.asarray(folds);groups=[folds==f for f in np.unique(folds)]
    grid=np.linspace(.05,.75,29)
    for _ in range(3):
        for j in range(6):
            values=[]
            for v in grid:
                candidate=t.copy();candidate[j]=v
                values.append(np.mean([score_subset(y[m],p[m]>=candidate)['score'] for m in groups]))
            t[j]=grid[int(np.argmax(values))]
    return t

def rank_predictions(prob,rates,ids):
    prob=np.asarray(prob);rates=np.asarray(rates);ids=np.asarray(ids)
    if len(prob)!=len(ids) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid ranking IDs')
    if ((rates<0)|(rates>1)).any():raise ValueError('Invalid prevalence')
    pred=np.zeros(prob.shape,dtype=bool)
    for j,rate in enumerate(rates):
        k=int(np.floor(len(prob)*rate+.5));order=np.lexsort((ids,-prob[:,j]));pred[order[:k],j]=True
    return pred

def push_status(events,cutoff,stale_hours=48):
    cutoff=stamp(cutoff);e=asof_events(events,cutoff)
    latest=e[e.kind=='push'].event_time.max();global_time=e.available_at.max()
    age=float((cutoff-latest).total_seconds()/3600) if pd.notna(latest) else float('inf')
    return dict(last_observed_push_utc=str(latest),push_age_hours=age,
        coverage_warning=bool(age>48),
        fallback_recommended=bool(age>=stale_hours),stale_threshold_hours=stale_hours,
        latest_observed_record_utc=str(global_time),
        warning='Staleness is a coverage warning, not proof of missing records or inactivity.')

def push_frequency(events,tables,cutoff,ids,freeze_clock=False):
    cutoff=stamp(cutoff);e=asof_events(events,cutoff)
    if freeze_clock:
        last=e[e.kind=='push'].event_time.max()
        if pd.isna(last):return np.zeros(len(ids))
        cutoff=min(cutoff,last+pd.Timedelta(nanoseconds=1));e=asof_events(events,cutoff)
    e=e[(e.kind=='push')&(e.event_time>=cutoff-pd.Timedelta(days=84))].copy()
    age=(cutoff-e.event_time).dt.total_seconds().to_numpy()
    e['week']=np.maximum(np.ceil(age/(7*86400)).astype(int)-1,0)
    c=e.drop_duplicates(['user_id','week']).groupby('user_id').size()
    idx=tables['Users'].set_index('github_user_id').loc[np.asarray(ids)].user_id
    return c.reindex(idx,fill_value=0).to_numpy()/12

def complete_training_origins(start,observation_cutoff,include_latest=True):
    """Weekly origins plus the latest complete midnight window, all before cutoff."""
    cutoff=stamp(observation_cutoff)
    latest=(cutoff+pd.Timedelta(hours=8)-pd.Timedelta(days=7)).floor('D')-pd.Timedelta(hours=8)
    dates=list(pd.date_range(stamp(start),latest,freq='7D'))
    if include_latest and latest>=stamp(start) and latest not in dates:dates.append(latest)
    return sorted(t for t in dates if t+pd.Timedelta(days=7)<=cutoff)

def prediction_cutoffs(latest_record,force=False):
    """An early backup ends its feature clock at observed data, never at Nov11."""
    target=stamp('2026-11-12T00:00:00+08:00')
    official_cutoff=target-pd.Timedelta(hours=4)
    latest=stamp(latest_record)
    if pd.isna(latest):raise ValueError('No observed activity records')
    stale=latest<official_cutoff-pd.Timedelta(hours=2)
    if stale and not force:
        raise ValueError('Official final predictions require the final Nov11 release. Use --force for a clearly labeled early backup.')
    feature_cutoff=min(official_cutoff,latest+pd.Timedelta(nanoseconds=1)) if force else official_cutoff
    return dict(target=target,feature_cutoff=feature_cutoff,
        training_cutoff=min(feature_cutoff,latest),stale=bool(stale),backup=bool(force))
