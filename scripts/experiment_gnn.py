"""Temporal heterogeneous GNN on the user-repository graph (extra experiment for Q2/Q3).

Same protocol as `asg2.revised experiment`: identical origins (train 09/10+09/17, calibration
10/01+10/08, development 10/22+10/29, Taiwan), 4-hour observation gap, as-of visibility, labels,
per-activity thresholds tuned on calibration weeks only. The submission is NOT changed.

Graph per origin t (rows visible before t - 4 h):
  nodes  = visible cohort users (with the 43 temporal30 inputs) + visible repositories
           (7/30-day activity per type and contributor counts, own-user events included)
  edges  = user -- repository for every pair with an event in the last 90 days;
           edge features = log1p counts per activity in the last 7/30/90 days and the
           recency of the last interaction (a time encoding of each relation).
Model (`gnn`): user MLP -> repo update = mean over edges of f([h_user, e]) -> user update =
mean over edges of g([h_repo, e]) -> 6 logits from [h_user, h_user']. Two hops: user -> repo -> user.
Ablation (`mlp`): the same user MLP and head without message passing (same inputs, same training).
Each model is an average of 3 seeds; single-thread, deterministic.

Usage:
  python scripts/experiment_gnn.py --data PREPARED --run RUN_DIR --analysis ANALYSIS_DIR --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asg2.core import TARGETS, asof_events, build_events, labels, score, stamp  # noqa: E402
from asg2.coverage import tune_mean_thresholds  # noqa: E402
from asg2.revised import columns, local_dates, make_frame, read_data  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
KINDS = TARGETS
EPOCHS, HID, LR, WD, SEEDS = 150, 64, 3e-3, 1e-4, (42, 43, 44)


def graph(tables, events, origin, frame, gap=4):
    """Bipartite user-repo graph as of origin - gap; target users in frame order."""
    t = stamp(origin) - pd.Timedelta(hours=gap)
    u, r = tables['Users'], tables['Repositories']
    valid_u = set(u.loc[u.first_seen_at < t, 'user_id']); valid_r = r.loc[r.first_seen_at < t, 'repo_id']
    e = asof_events(events, t)
    e = e[e.user_id.isin(valid_u) & e.repo_id.isin(set(valid_r))]
    e90 = e[e.event_time >= t - pd.Timedelta(days=90)]
    uid = u.set_index('github_user_id').loc[frame.UserID.to_numpy()].user_id.to_numpy()
    upos = pd.Series(np.arange(len(uid)), index=uid)
    edges = e90[e90.user_id.isin(upos.index)]
    pairs = edges[['user_id', 'repo_id']].drop_duplicates().reset_index(drop=True)
    repos = np.sort(pairs.repo_id.unique()); rpos = pd.Series(np.arange(len(repos)), index=repos)
    ef = []
    for days in (7, 30, 90):
        w = edges[edges.event_time >= t - pd.Timedelta(days=days)]
        c = w.groupby(['user_id', 'repo_id', 'kind']).size().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
        ef.append(np.log1p(c.reindex(pd.MultiIndex.from_frame(pairs), fill_value=0).to_numpy(float)))
    last = edges.groupby(['user_id', 'repo_id']).event_time.max().reindex(pd.MultiIndex.from_frame(pairs))
    ef.append((((t - last).dt.total_seconds() / 86400 / 90).to_numpy(float)).reshape(-1, 1))
    efeat = np.hstack(ef)
    rf = []
    for days in (7, 30):
        w = e[(e.event_time >= t - pd.Timedelta(days=days)) & e.repo_id.isin(repos)]
        c = w.groupby(['repo_id', 'kind']).size().unstack(fill_value=0).reindex(index=repos, columns=KINDS, fill_value=0)
        rf.append(np.log1p(c.to_numpy(float)))
        rf.append(np.log1p(w.groupby('repo_id').user_id.nunique().reindex(repos, fill_value=0).to_numpy(float)).reshape(-1, 1))
    return dict(ui=torch.tensor(upos.loc[pairs.user_id].to_numpy()), ri=torch.tensor(rpos.loc[pairs.repo_id].to_numpy()),
                ef=torch.tensor(efeat, dtype=torch.float32), rx=torch.tensor(np.hstack(rf), dtype=torch.float32),
                n_users=len(uid), n_repos=len(repos))


def mean_agg(src, index, n):
    out = torch.zeros(n, src.shape[1]).index_add_(0, index, src)
    cnt = torch.zeros(n).index_add_(0, index, torch.ones(len(index))).clamp(min=1)
    return out / cnt[:, None]


class Net(nn.Module):
    def __init__(self, dx, dr, de, graph_on):
        super().__init__()
        self.graph_on = graph_on
        self.user = nn.Sequential(nn.Linear(dx, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())
        if graph_on:
            self.repo = nn.Sequential(nn.Linear(dr, HID), nn.ReLU())
            self.to_repo = nn.Sequential(nn.Linear(HID + de, HID), nn.ReLU())
            self.repo_upd = nn.Linear(2 * HID, HID)
            self.to_user = nn.Sequential(nn.Linear(HID + de, HID), nn.ReLU())
            self.head = nn.Linear(2 * HID, 6)
        else:
            self.head = nn.Linear(HID, 6)

    def forward(self, x, g):
        h = self.user(x)
        if not self.graph_on:
            return self.head(h)
        hr = self.repo(g['rx'])
        m_r = mean_agg(self.to_repo(torch.cat([h[g['ui']], g['ef']], 1)), g['ri'], g['n_repos'])
        hr = torch.relu(self.repo_upd(torch.cat([hr, m_r], 1)))
        m_u = mean_agg(self.to_user(torch.cat([hr[g['ri']], g['ef']], 1)), g['ui'], g['n_users'])
        return self.head(torch.cat([h, m_u], 1))


def fit_predict(train, others, graph_on):
    """train/others: lists of (x tensor, graph, y array|None). Returns probabilities per `others` item."""
    dx, dr, de = train[0][0].shape[1], train[0][1]['rx'].shape[1], train[0][1]['ef'].shape[1]
    out = [np.zeros((o[0].shape[0], 6)) for o in others]
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = Net(dx, dr, de, graph_on)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        loss = nn.BCEWithLogitsLoss()
        for _ in range(EPOCHS):
            opt.zero_grad()
            total = sum(loss(net(x, g), torch.tensor(y, dtype=torch.float32)) for x, g, y in train)
            total.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for k, (x, g, _) in enumerate(others):
                out[k] += torch.sigmoid(net(x, g)).numpy() / len(SEEDS)
    return out


def bootstrap(preds, y, weeks, reps=1000, seed=7):
    rng = np.random.default_rng(seed)
    ix = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    draws = {k: [] for k in preds}
    for _ in range(reps):
        take = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in ix])
        for k in preds:
            draws[k].append(score(y[take], preds[k][take])['score'])
    return {k: np.asarray(v) for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--analysis', required=True, help='analysis dir with bootstrap seed reference (for temporal30 / rule rows)')
    ap.add_argument('--out', required=True); ap.add_argument('--reps', type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    design = json.loads((Path(a.run) / 'design.json').read_text()); cfg = design['config']; gap = cfg['observation_gap_hours']
    tables, _ = read_data(a.data); events = build_events(tables)
    groups = {k: local_dates(cfg[k + '_origins_taiwan']) for k in ['train', 'calibration', 'evaluation']}
    ref = stamp('2026-09-27T00:00:00+08:00')
    data = {}
    for t in sum(groups.values(), []) + [ref]:
        X = make_frame(tables, events, t, gap=gap); y = labels(tables, events, t, X.UserID)
        data[t] = (X, graph(tables, events, t, X, gap), y)
    cols = columns(data[groups['train'][0]][0], 'temporal30')
    # Inputs are clipped to the range seen on training rows, then standardized: unlike trees, a
    # neural network extrapolates linearly, and account age / recency grow past the training range
    # in later weeks (data start 09/01), which otherwise collapses its later-week predictions.
    tr = np.vstack([data[t][0][cols].to_numpy(float) for t in groups['train']])
    lo, hi = tr.min(0), tr.max(0); mu, sd = tr.mean(0), tr.std(0) + 1e-6
    rx_tr = torch.cat([data[t][1]['rx'] for t in groups['train']])
    rlo, rhi, rmu, rsd = rx_tr.min(0).values, rx_tr.max(0).values, rx_tr.mean(0), rx_tr.std(0) + 1e-6
    ef_tr = torch.cat([data[t][1]['ef'] for t in groups['train']])
    elo, ehi, emu, esd = ef_tr.min(0).values, ef_tr.max(0).values, ef_tr.mean(0), ef_tr.std(0) + 1e-6
    items = {}
    for t, (X, g, y) in data.items():
        g = dict(g, rx=(torch.clamp(g['rx'], rlo, rhi) - rmu) / rsd, ef=(torch.clamp(g['ef'], elo, ehi) - emu) / esd)
        x = (np.clip(X[cols].to_numpy(float), lo, hi) - mu) / sd
        items[t] = (torch.tensor(x, dtype=torch.float32), g, y)
    train = [items[t] for t in groups['train']]
    rest = groups['calibration'] + groups['evaluation'] + [ref]
    cy = np.concatenate([items[t][2] for t in groups['calibration']])
    folds = np.concatenate([np.repeat(i, len(items[t][2])) for i, t in enumerate(groups['calibration'])])
    dy = np.concatenate([items[t][2] for t in groups['evaluation']])
    weeks = np.concatenate([np.repeat(i, len(items[t][2])) for i, t in enumerate(groups['evaluation'])])
    rows, preds = [], {}
    for name, graph_on in [('mlp_temporal', False), ('gnn_temporal_graph', True)]:
        probs = dict(zip(rest, fit_predict(train, [items[t] for t in rest], graph_on)))
        cp = np.vstack([probs[t] for t in groups['calibration']])
        thr = tune_mean_thresholds(cy, cp, folds)
        cal = float(np.mean([score(cy[folds == i], cp[folds == i] >= thr)['score'] for i in np.unique(folds)]))
        p = np.vstack([probs[t] for t in groups['evaluation']]) >= thr
        preds[name] = p
        rows.append(dict(model=name, calibration_score=cal, **score(dy, p),
                         reference_week_score=score(items[ref][2], probs[ref] >= thr)['score'],
                         **{f'thr_{k}': v for k, v in zip(TARGETS, thr)}))
    # rule and temporal30 recomputed on the same rows for paired comparisons
    preds['last_week_rule'] = np.vstack([np.column_stack([data[t][0]['t_count7_' + k].to_numpy() > 0 for k in TARGETS])
                                         for t in groups['evaluation']])
    from asg2.revised import fit, probability
    X = pd.concat([data[t][0] for t in groups['train']], ignore_index=True); y = np.concatenate([data[t][2] for t in groups['train']])
    heads = fit(X, y, cols, cfg['trees'], cfg['seed'])
    C = pd.concat([data[t][0] for t in groups['calibration']], ignore_index=True)
    thr = tune_mean_thresholds(cy, probability(heads, C, cols), folds)
    preds['temporal30_lightgbm'] = np.vstack([probability(heads, data[t][0], cols) >= thr for t in groups['evaluation']])
    reported = pd.read_csv(Path(a.run) / 'metrics.csv')
    want = reported[(reported.model == 'temporal30') & (reported.split == 'pooled_observed_development')].score.iloc[0]
    assert abs(score(dy, preds['temporal30_lightgbm'])['score'] - want) < 1e-12
    draws = bootstrap(preds, dy, weeks, a.reps)
    for r in rows:
        for base in ['last_week_rule', 'temporal30_lightgbm', 'mlp_temporal']:
            if base == r['model']:
                continue
            d = draws[r['model']] - draws[base]
            r[f'diff_vs_{base}'] = float(np.mean(d)); r[f'diff_vs_{base}_ci_low'] = float(np.percentile(d, 2.5))
            r[f'diff_vs_{base}_ci_high'] = float(np.percentile(d, 97.5))
        r['ci_low'], r['ci_high'] = (float(np.percentile(draws[r['model']], q)) for q in (2.5, 97.5))
    pd.DataFrame(rows).to_csv(out / 'gnn_metrics.csv', index=False)
    (out / 'gnn_design.json').write_text(json.dumps(dict(
        epochs=EPOCHS, hidden=HID, lr=LR, weight_decay=WD, seeds=SEEDS, torch=torch.__version__,
        graph='user--repository, edges = any event in last 90 days, edge features = log counts per activity in 7/30/90 d + recency',
        user_inputs='temporal30 columns (43); all inputs clipped to the training range, then standardized', protocol=cfg,
        note='Extra experiment; the submitted CSV is unchanged.'), indent=2, default=str))
    print(pd.DataFrame(rows).round(4).T)


if __name__ == '__main__':
    main()
