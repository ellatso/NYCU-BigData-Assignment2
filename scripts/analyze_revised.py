"""Additional analyses for the report (Q1-Q4), computed from the official prepared tables.

Every number used in the slides comes from the files this script writes. It refits the
same variants as `asg2.revised experiment` (same config, seed and single-thread LightGBM)
and first asserts that its pooled development scores equal the experiment's metrics.csv,
so the extra analyses describe exactly the reported models.

Usage:
  python scripts/analyze_revised.py --data PREPARED_DATA --run RUN_DIR --out analysis_revised
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asg2.core import TARGETS, build_events, labels, score, stamp  # noqa: E402
from asg2.coverage import tune_mean_thresholds  # noqa: E402
from asg2.revised import VARIANTS, columns, fit, local_dates, make_frame, probability, read_data  # noqa: E402

KEY_VARIANTS = ['observed_history', 'recent7', 'temporal30', 'static_relations',
                'dynamic_relations', 'availability_clock']
TABLE_OF_KIND = {'open_pr': 'PullRequests', 'open_issue': 'Issues', 'issue_comment': 'IssueComments',
                 'pr_review': 'PRReviews', 'pr_review_comment': 'PRReviewComments', 'push': 'Pushes'}
TW = pd.Timedelta(hours=8)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')


def weekly_volume(events, raw_start, raw_end):
    """Recorded activity rows per Taiwan week (Thursday start, like the forecast origins)."""
    e = events[(events.event_time >= raw_start) & (events.event_time < raw_end)].copy()
    e['tw_day'] = (e.event_time + TW).dt.floor('D')
    daily = e.groupby(['tw_day', 'kind']).size().unstack(fill_value=0).reindex(columns=TARGETS, fill_value=0)
    first = (raw_start + TW).ceil('D')
    while first.dayofweek != 3:  # Thursday
        first += pd.Timedelta(days=1)
    rows = []
    week = first
    while week + pd.Timedelta(days=7) <= raw_end + TW:
        part = daily[(daily.index >= week) & (daily.index < week + pd.Timedelta(days=7))]
        active = e[(e.tw_day >= week) & (e.tw_day < week + pd.Timedelta(days=7))].user_id.nunique()
        rows.append(dict(week_start_tw=week.date(), active_users=int(active), **part.sum().astype(int).to_dict()))
        week += pd.Timedelta(days=7)
    return daily.reset_index(), pd.DataFrame(rows)


def prevalence(tables, events, origins_tw, gap):
    rows = []
    u = tables['Users']
    for o in origins_tw:
        origin = stamp(o)
        ids = u.loc[u.is_cohort & u.first_seen_at.lt(origin - pd.Timedelta(hours=gap)), 'github_user_id'].to_numpy()
        y = labels(tables, events, origin, ids)
        rows.append(dict(origin_tw=o[:10], users=len(ids), **dict(zip(TARGETS, y.mean(0).round(4)))))
    return pd.DataFrame(rows)


def bootstrap(preds, ys, weeks, reps=1000, seed=7):
    """User-level bootstrap, resampling users within each development week (paired across models)."""
    rng = np.random.default_rng(seed)
    idx_by_week = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    names = list(preds)
    draws = {n: [] for n in names}
    for _ in range(reps):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_week])
        for n in names:
            draws[n].append(score(ys[take], preds[n][take])['score'])
    draws = {n: np.asarray(v) for n, v in draws.items()}
    return draws


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def five_target(y, p):
    """Diagnostic without push (push records collapse in the last development week)."""
    keep = [j for j, t in enumerate(TARGETS) if t != 'push']
    y = y[:, keep]; p = p[:, keep]
    tp = ((y == 1) & (p == 1)).sum(0); fp = ((y == 0) & (p == 1)).sum(0); fn = ((y == 1) & (p == 0)).sum(0)
    den = 2 * tp + fp + fn
    per = np.divide(2 * tp, den, out=np.ones(len(keep)), where=den != 0)
    micro = 2 * tp.sum() / den.sum() if den.sum() else 1.0
    return float((per.mean() + micro) / 2)


def group_of(col):
    if col.startswith('u_'):
        return 'user profile'
    if col.startswith('t_recency_'):
        return 'temporal: recency'
    if col.startswith('t_trend_'):
        return 'temporal: trend'
    if col.startswith('t_'):
        return 'temporal: window counts'
    if col.startswith('d_received_'):
        return 'relational: received interactions'
    if col.startswith(('d_peer', 'd_cohort')):
        return 'relational: collaborators'
    if col.startswith('d_repo7_') or col.startswith('d_reviewed_'):
        return 'relational: repository activity'
    if col.startswith('r_static_'):
        return 'relational: static'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--run', required=True, help='output directory of asg2.revised experiment')
    ap.add_argument('--out', default='analysis_revised')
    ap.add_argument('--reps', type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run = Path(a.run)
    design = json.loads((run / 'design.json').read_text()); config = design['config']
    gap = config['observation_gap_hours']
    reported = pd.read_csv(run / 'metrics.csv')
    tables, meta = read_data(a.data)
    events = build_events(tables)
    raw_start, raw_end = [stamp(x) for x in design['source_period_utc']]

    # 1. Data volume over time (Q1 period, Q3 non-stationarity)
    daily, weekly = weekly_volume(events, raw_start, raw_end)
    daily.to_csv(out / 'daily_volume.csv', index=False); weekly.to_csv(out / 'weekly_volume.csv', index=False)

    # 2. Label prevalence for every weekly origin with a complete outcome week
    origins = []
    o = pd.Timestamp('2026-09-10')
    while stamp(o.strftime('%Y-%m-%d') + 'T00:00:00+08:00') + pd.Timedelta(days=7) <= raw_end:
        origins.append(o.strftime('%Y-%m-%d') + 'T00:00:00+08:00'); o += pd.Timedelta(days=7)
    prev = prevalence(tables, events, origins, gap)
    prev.to_csv(out / 'weekly_prevalence.csv', index=False)

    # 3. Refit the reported variants and keep development predictions
    groups = {k: local_dates(config[k + '_origins_taiwan']) for k in ['train', 'calibration', 'evaluation']}
    ref_origin = stamp('2026-09-27T00:00:00+08:00')  # TA reference-score week in README
    cache = {}
    for t in sum(groups.values(), []) + [ref_origin]:
        X = make_frame(tables, events, t, gap=gap)
        cache[t] = (X, labels(tables, events, t, X.UserID))
    X = pd.concat([cache[t][0] for t in groups['train']], ignore_index=True)
    y = np.concatenate([cache[t][1] for t in groups['train']])
    C = pd.concat([cache[t][0] for t in groups['calibration']], ignore_index=True)
    cy = np.concatenate([cache[t][1] for t in groups['calibration']])
    folds = np.concatenate([np.repeat(i, len(cache[t][0])) for i, t in enumerate(groups['calibration'])])
    dev_y = np.concatenate([cache[t][1] for t in groups['evaluation']])
    dev_week = np.concatenate([np.repeat(i, len(cache[t][0])) for i, t in enumerate(groups['evaluation'])])
    preds, models, ref_rows, five_rows, threshold_rows = {}, {}, [], [], []
    for variant in VARIANTS:
        clock = variant == 'availability_clock'
        def frame(t):
            return make_frame(tables, events, t, gap=gap, availability_clock=True) if clock else cache[t][0]
        A = pd.concat([frame(t) for t in groups['train']], ignore_index=True) if clock else X
        B = pd.concat([frame(t) for t in groups['calibration']], ignore_index=True) if clock else C
        cols = columns(A, variant)
        heads = fit(A, y, cols, config['trees'], config['seed'])
        thr = tune_mean_thresholds(cy, probability(heads, B, cols), folds)
        models[variant] = (heads, cols, thr)
        threshold_rows.append(dict(model=variant, **dict(zip(TARGETS, thr))))
        p = np.concatenate([probability(heads, frame(t), cols) >= thr for t in groups['evaluation']])
        preds[variant] = p
        got = score(dev_y, p)['score']
        want = reported[(reported.model == variant) & (reported.split == 'pooled_observed_development')].score.iloc[0]
        assert abs(got - want) < 1e-12, (variant, got, want)
        for i, t in enumerate(groups['evaluation']):
            m = dev_week == i
            five_rows.append(dict(model=variant, week_tw=str((t + TW).date()), five_target_score=five_target(dev_y[m], p[m]),
                                  six_target_score=score(dev_y[m], p[m])['score']))
        Dr, yr = cache[ref_origin]
        ref_rows.append(dict(model=variant, **score(yr, probability(heads, frame(ref_origin) if clock else Dr, cols) >= thr)))
    rule = np.concatenate([np.column_stack([cache[t][0]['t_count7_' + k].to_numpy() > 0 for k in TARGETS])
                           for t in groups['evaluation']])
    preds['last_week_rule'] = rule
    for i, t in enumerate(groups['evaluation']):
        m = dev_week == i
        five_rows.append(dict(model='last_week_rule', week_tw=str((t + TW).date()), five_target_score=five_target(dev_y[m], rule[m]),
                              six_target_score=score(dev_y[m], rule[m])['score']))
    Dr, yr = cache[ref_origin]
    ref_rows.append(dict(model='last_week_rule', **score(yr, np.column_stack([Dr['t_count7_' + k].to_numpy() > 0 for k in TARGETS]))))
    pd.DataFrame(threshold_rows).to_csv(out / 'thresholds.csv', index=False)
    pd.DataFrame(five_rows).to_csv(out / 'five_target_by_week.csv', index=False)
    ref = pd.DataFrame(ref_rows); ref.insert(1, 'week_tw', '2026-09-27..10-04'); ref.insert(2, 'users', len(Dr))
    ref.to_csv(out / 'reference_week.csv', index=False)

    # 4. Paired bootstrap over users: is any difference larger than sampling noise?
    draws = bootstrap({k: preds[k] for k in VARIANTS + ['last_week_rule']}, dev_y, dev_week, a.reps)
    rows = []
    for k, v in draws.items():
        lo, hi = ci(v)
        d_rule = v - draws['last_week_rule']; d_t = v - draws['temporal30']
        rows.append(dict(model=k, score=score(dev_y, preds[k])['score'], ci_low=lo, ci_high=hi,
                         diff_vs_last_week=float(np.mean(d_rule)), diff_vs_last_week_ci_low=ci(d_rule)[0],
                         diff_vs_last_week_ci_high=ci(d_rule)[1], p_better_than_last_week=float(np.mean(d_rule > 0)),
                         diff_vs_temporal30=float(np.mean(d_t)), diff_vs_temporal30_ci_low=ci(d_t)[0],
                         diff_vs_temporal30_ci_high=ci(d_t)[1]))
    pd.DataFrame(rows).to_csv(out / 'bootstrap_development.csv', index=False)

    # 5. Which feature families does the submitted model use? (LightGBM gain)
    heads, cols, _ = models[config['submission_variant']]
    imp = []
    for j, h in enumerate(heads):
        if hasattr(h, 'booster_'):
            gain = h.booster_.feature_importance(importance_type='gain')
            for c, g in zip(cols, gain):
                imp.append(dict(target=TARGETS[j], feature=c, group=group_of(c), gain=float(g)))
    imp = pd.DataFrame(imp)
    imp.to_csv(out / 'feature_importance.csv', index=False)
    share = imp.groupby(['target', 'group']).gain.sum().unstack(fill_value=0)
    share = share.div(share.sum(1), axis=0).reindex(TARGETS)
    share.to_csv(out / 'feature_group_share.csv')

    # 6. How often is each relational feature family non-zero? (explains the received-relations null result)
    cov = []
    for t in groups['evaluation']:
        D = cache[t][0]
        for g in sorted({group_of(c) for c in D.columns if c.startswith(('d_', 'r_static_'))}):
            cs = [c for c in D.columns if group_of(c) == g]
            cov.append(dict(week_tw=str((t + TW).date()), group=g, columns=len(cs), users=len(D),
                            share_users_nonzero=float((D[cs] != 0).any(axis=1).mean())))
    pd.DataFrame(cov).to_csv(out / 'relation_coverage.csv', index=False)
    ic = tables['IssueComments'].merge(tables['Issues'][['issue_id', 'author_id']].rename(columns={'author_id': 'thread_author'}), on='issue_id')
    u = tables['Users'].set_index('user_id').is_cohort
    authored = ic[ic.thread_author.map(u).fillna(False).astype(bool)]
    closure = dict(
        comments_on_cohort_authored_threads=int(len(authored)),
        of_which_by_thread_author=int((authored.author_id == authored.thread_author).sum()),
        of_which_by_other_users=int((authored.author_id != authored.thread_author).sum()),
        note='Only events performed by the 9,000 cohort users are kept, so replies by non-cohort users to a cohort user\'s thread are absent.')
    pr = tables['PullRequests']; rv = tables['PRReviews'].merge(pr[['pr_id', 'author_id']], on='pr_id')
    rv_auth = rv[rv.author_id.map(u).fillna(False).astype(bool)]
    closure.update(reviews_on_cohort_authored_prs=int(len(rv_auth)),
                   of_which_by_other_reviewers=int((rv_auth.reviewer_id != rv_auth.author_id).sum()),
                   users_total=int(len(tables['Users'])), cohort_users=int(tables['Users'].is_cohort.sum()))
    save_json(out / 'cohort_closure.json', closure)

    # 7. Visibility delay: occurrence time vs. the time a row becomes visible (row time)
    delay = []
    for kind in TARGETS:
        e = events[events.kind == kind]
        h = (e.available_at - e.event_time).dt.total_seconds() / 3600
        delay.append(dict(target=kind, table=TABLE_OF_KIND[kind], rows=len(e), share_delayed_over_1h=float((h > 1).mean()),
                          share_delayed_over_24h=float((h > 24).mean()), p99_hours=float(h.quantile(.99))))
    pd.DataFrame(delay).to_csv(out / 'visibility_delay.csv', index=False)

    save_json(out / 'analysis_manifest.json', dict(data_period_utc=design['source_period_utc'], config=config,
              bootstrap_reps=a.reps, reproduced_metrics='pooled development scores of all variants equal metrics.csv',
              reference_week_note='TA README reference week 2026-09-27..10-04 overlaps the calibration window; '
                                  'compare the copy-last-week row with the README, not as an untouched test.'))
    print('wrote', out)


if __name__ == '__main__':
    main()
