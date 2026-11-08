"""Chronological calibration helpers with explicit sample-age weighting."""
import numpy as np
import pandas as pd
from .core import TARGETS, stamp, asof_events
from .run import Constant, LGBMClassifier

RECIPES = ['full_history', 'recent12', 'weighted6']

def eligible_origins(origins, forecast, recipe):
    if recipe not in RECIPES: raise ValueError(recipe)
    cutoff = stamp(forecast) - pd.Timedelta(hours=4)
    result = sorted(stamp(t) for t in origins if stamp(t)+pd.Timedelta(days=7) <= cutoff)
    if recipe == 'recent12': result = result[-12:]
    if not result: raise ValueError('No fully available training outcomes')
    return result

def recency_weights(origins):
    origin = pd.to_datetime(origins)
    age_weeks = (origin.max()-origin).dt.total_seconds().to_numpy()/(7*86400)
    weights = np.exp2(-age_weeks/6)
    return weights/weights.mean()

def train(X, y, columns, recipe):
    weights = recency_weights(X.origin) if recipe == 'weighted6' else None
    models = []
    for j in range(6):
        yy = y[:,j]
        if len(np.unique(yy)) < 2: model = Constant(float(np.average(yy, weights=weights)))
        else:
            model = LGBMClassifier(n_estimators=160,learning_rate=.045,num_leaves=23,max_depth=-1,
                min_child_samples=70,reg_lambda=4,subsample=.85,subsample_freq=1,
                colsample_bytree=.85,random_state=42,n_jobs=4,verbosity=-1)
            model.fit(X[columns], yy, sample_weight=weights)
        models.append(model)
    return models

def frequency12(tables, events, origin, ids):
    """Fraction of 12 complete trailing observation weeks with each activity."""
    cutoff = stamp(origin)-pd.Timedelta(hours=4)
    u=tables['Users']; r=tables['Repositories']
    e=asof_events(events,cutoff)
    e=e[(e.event_time>=cutoff-pd.Timedelta(days=84)) &
        e.user_id.isin(u.loc[u.first_seen_at<cutoff,'user_id']) &
        e.repo_id.isin(r.loc[r.first_seen_at<cutoff,'repo_id'])].copy()
    # Events exactly on the lower boundary belong to the twelfth, oldest week.
    age=(cutoff-e.event_time).dt.total_seconds().to_numpy()
    e['week']=np.maximum(np.ceil(age/(7*86400)).astype(int)-1,0)
    c=e.drop_duplicates(['user_id','kind','week']).groupby(['user_id','kind']).size().unstack(fill_value=0)
    index=u.set_index('github_user_id').loc[np.asarray(ids)].user_id.to_numpy()
    return c.reindex(index=index,columns=TARGETS,fill_value=0).fillna(0).to_numpy()/12

def blend(tree_probability, frequency, alpha):
    if not 0 <= alpha <= 1: raise ValueError('Blend weight outside [0,1]')
    return (1-alpha)*np.asarray(tree_probability)+alpha*np.asarray(frequency)
