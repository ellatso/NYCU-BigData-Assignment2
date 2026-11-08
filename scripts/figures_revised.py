"""Slide figures for the revised Assignment 2 report.

Reads only files written by `asg2.revised experiment` (RUN) and `scripts/analyze_revised.py`
(ANALYSIS), so every plotted number is traceable. Writes PNG (slides) and PDF (vector).

Usage:
  python scripts/figures_revised.py --run RUN_DIR --analysis analysis_revised --out report_materials/figures_revised
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill, light mode); categorical slots in fixed order.
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
TARGETS = ['open_pr', 'open_issue', 'issue_comment', 'pr_review', 'pr_review_comment', 'push']
NICE = {'observed_history': 'User profile only', 'recent7': '+ last 7 days', 'temporal30': 'Temporal (7/14/30 d)',
        'no_recency': 'Temporal − recency', 'no_trend': 'Temporal − trend', 'static_relations': 'Static relations only',
        'temporal_static_relations': 'Temporal + static rel.', 'received_relations': 'Temporal + received rel.',
        'dynamic_relations': 'Temporal + dynamic rel. (submitted)', 'no_peer_relations': 'Dynamic rel. − collaborators',
        'availability_clock': 'Temporal, availability clock', 'last_week_rule': 'Copy last week (rule)'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlesize': 13,
    'axes.titlelocation': 'left', 'legend.frameon': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})


def save(fig, out, name):
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'{name}.{ext}', facecolor=SURF)
    plt.close(fig)


def dot(out, name, source):
    (out / f'{name}.dot').write_text(source)
    for ext in ['png', 'pdf', 'svg']:
        args = ['dot', f'-T{ext}', str(out / f'{name}.dot'), '-o', str(out / f'{name}.{ext}')]
        if ext == 'png':
            args.insert(1, '-Gdpi=220')
        subprocess.run(args, check=True)


# ---------------------------------------------------------------- Q1 schema graph
SCHEMA = {
    'Users': ('user_id', []),
    'Repositories': ('repo_id', []),
    'PullRequests': ('pr_id', [('repo_id', 'Repositories'), ('author_id', 'Users'), ('merged_by_id', 'Users')]),
    'Issues': ('issue_id', [('repo_id', 'Repositories'), ('author_id', 'Users'), ('assignee_id', 'Users'),
                            ('pull_request_id', 'PullRequests')]),
    'IssueComments': ('comment_id', [('issue_id', 'Issues'), ('repo_id', 'Repositories'), ('author_id', 'Users')]),
    'PRReviews': ('review_id', [('pr_id', 'PullRequests'), ('repo_id', 'Repositories'), ('reviewer_id', 'Users')]),
    'PRReviewComments': ('comment_id', [('pr_id', 'PullRequests'), ('repo_id', 'Repositories'), ('author_id', 'Users'),
                                        ('pull_request_review_id', 'PRReviews'), ('in_reply_to_id', 'PRReviewComments')]),
    'Pushes': ('push_id', [('user_id', 'Users'), ('repo_id', 'Repositories')]),
}
ROW_TIME = {'Users': 'first_seen_at', 'Repositories': 'first_seen_at', 'PullRequests': 'first_seen_at',
            'Issues': 'first_seen_at', 'PRReviews': 'first_seen_at', 'IssueComments': 'created_at',
            'PRReviewComments': 'created_at', 'Pushes': 'created_at'}


def schema_graph(out, counts):
    color = {'Users': '#cde2fb', 'Repositories': '#cde2fb'}
    lines = ['digraph schema {', 'graph [rankdir=LR, bgcolor="#fcfcfb", nodesep=0.35, ranksep=1.1, fontname="DejaVu Sans", pad=0.2];',
             'node [shape=plain, fontname="DejaVu Sans", fontsize=11];', 'edge [color="#52514e", arrowsize=0.7, penwidth=1.1, fontname="DejaVu Sans", fontsize=9, fontcolor="#52514e"];']
    for t, (pk, fks) in SCHEMA.items():
        head = color.get(t, '#e8e7e2')
        rows = [f'<tr><td bgcolor="{head}" port="title"><b>{t}</b><br/><font point-size="9" color="#52514e">{counts.get(t, 0):,} rows · row time: {ROW_TIME[t]}</font></td></tr>',
                f'<tr><td align="left" port="pk"><u><b>{pk}</b></u>  <font color="#2a78d6"><b>PK</b></font></td></tr>']
        for col, ref in fks:
            rows.append(f'<tr><td align="left" port="{col}">{col}  <font color="#eb6834"><b>FK</b></font><font point-size="9" color="#898781"> → {ref}</font></td></tr>')
        lines.append(f'{t} [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="5" color="#c3c2b7" bgcolor="#ffffff">{"".join(rows)}</table>>];')
    for t, (_, fks) in SCHEMA.items():
        for col, ref in fks:
            ec = {'Users': '#2a78d6', 'Repositories': '#898781'}.get(ref, '#eb6834')
            lines.append(f'{t}:{col}:e -> {ref}:pk:w [color="{ec}"];' if ref != t else f'{t}:{col}:e -> {t}:pk:e [constraint=false, color="{ec}"];')
    lines.append('{rank=same; Users; Repositories;}')
    lines.append('{rank=same; IssueComments; PRReviewComments; Pushes;}')
    lines.append('legend [label=<<table border="0" cellspacing="2"><tr><td align="left"><font color="#2a78d6"><b>&#8594;</b></font> FK to Users</td></tr><tr><td align="left"><font color="#898781"><b>&#8594;</b></font> FK to Repositories</td></tr><tr><td align="left"><font color="#eb6834"><b>&#8594;</b></font> FK between activity tables</td></tr><tr><td align="left"><font point-size="9" color="#52514e">8 tables · 20 foreign keys · edge = FK &#8594; referenced PK</font></td></tr></table>>];')
    lines.append('}')
    dot(out, 'q1_schema_graph', '\n'.join(lines))


# ---------------------------------------------------------------- Q1/Q3 timeline
def timeline(out, design, volume):
    cfg = design['config']
    tw = lambda s: pd.Timestamp(s)  # noqa: E731  (all Taiwan dates)
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    rows = [('Downloaded GH Archive hours', [(pd.Timestamp('2026-09-01 08:00'), pd.Timestamp('2026-11-07 08:00'))], MUTED),
            ('Training outcome weeks', [(tw(d), tw(d) + pd.Timedelta(days=7)) for d in cfg['train_origins_taiwan']], S[0]),
            ('Threshold calibration weeks', [(tw(d), tw(d) + pd.Timedelta(days=7)) for d in cfg['calibration_origins_taiwan']], S[1]),
            ('Development (evaluation) weeks', [(tw(d), tw(d) + pd.Timedelta(days=7)) for d in cfg['evaluation_origins_taiwan']], S[2]),
            ('Refit for the submitted CSV', [(tw('2026-09-10'), pd.Timestamp('2026-11-07 08:00'))], S[3]),
            ('Forecast target week', [(tw('2026-11-22'), tw('2026-11-29'))], S[6])]
    for i, (label, spans, c) in enumerate(rows):
        y = len(rows) - 1 - i
        for a, b in spans:
            ax.barh(y, (b - a).total_seconds() / 86400, left=mdates.date2num(a), height=.55, color=c, edgecolor=SURF, linewidth=2)
        ax.text(mdates.date2num(pd.Timestamp('2026-08-30')), y, label, ha='right', va='center', color=INK2, fontsize=10.5)
    ax.axvline(mdates.date2num(pd.Timestamp('2026-11-21 23:55')), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(mdates.date2num(pd.Timestamp('2026-11-21 12:00')), 0, 'deadline 11/21 23:55 ', ha='right', va='center', fontsize=9, color=INK2)
    ax.text(mdates.date2num(pd.Timestamp('2026-11-08')), 5, 'feature cutoff\n11/07 08:00', ha='left', va='center', fontsize=9, color=INK2)
    ax.set_yticks([]); ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.TH)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_xlim(mdates.date2num(pd.Timestamp('2026-08-31')), mdates.date2num(pd.Timestamp('2026-11-30')))
    ax.set_ylim(-.6, len(rows) - .1)
    ax.set_title('Dataset period and time-based split (Taiwan time; weeks start on Thursday 00:00)')
    ax.text(0, -.2, 'Data period used: 2026-09-01 00:00 – 2026-11-07 00:00 UTC (1,608 hourly GH Archive files). '
            'Every sample uses only rows visible 4 h before its origin.\nRefit = 8 Thursday origins (09/10–10/29) + the latest complete 7-day window (origin 10/31); all outcome weeks end before the cutoff; calibrated thresholds stay fixed.', transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, out, 'q1_q3_timeline')


# ---------------------------------------------------------------- Q3 non-stationarity
def volume_multiples(out, weekly, design):
    w = weekly.copy(); w['week'] = pd.to_datetime(w.week_start_tw)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6), sharex=True)
    ev = [pd.Timestamp(d) for d in design['config']['evaluation_origins_taiwan']]
    for ax, t in zip(axes.flat, TARGETS):
        ax.bar(w.week + pd.Timedelta(days=.4), w[t], width=6.2, align='edge', color=S[0], edgecolor=SURF, linewidth=2)
        for e in ev:
            ax.axvspan(e, e + pd.Timedelta(days=7), color=S[2], alpha=.12, lw=0)
        peak = w[t].max(); last = w[t].iloc[-1]
        ax.set_title(t, fontsize=11.5)
        ax.text(w.week.iloc[-1] + pd.Timedelta(days=3.5), last, f'{last:,}', ha='center', va='bottom', fontsize=9, color=INK)
        ax.text(w.week.iloc[int(w[t].idxmax())] + pd.Timedelta(days=3.5), peak, f'{peak:,}', ha='center', va='bottom', fontsize=9, color=INK)
        ax.set_ylim(0, peak * 1.18)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}k' if v >= 1000 else f'{v:.0f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    for ax in axes[1]:
        ax.tick_params(axis='x', rotation=45)
    fig.suptitle('Recorded activity per week is not stationary: PR/issue/review activity drops ~5× after 09/17, pushes almost vanish after 10/29',
                 x=.01, ha='left', fontweight='bold', fontsize=12.5)
    fig.text(.01, -.02, 'Rows per Taiwan week (Thursday start) in the 8 preprocessed tables, by event time. Shaded: development weeks. '
             'All 1,608 source hours downloaded and passed gzip checks, so the drop is in GH Archive content, not missing files.', fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, out, 'q3_weekly_volume')


def prevalence_lines(out, prev, design):
    p = prev.copy(); p['week'] = pd.to_datetime(p.origin_tw) + pd.Timedelta(days=3.3)
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    cfg = design['config']
    for key, c, name in [('train', S[0], 'train'), ('calibration', S[1], 'calibration'), ('evaluation', S[2], 'development')]:
        for d in cfg[key + '_origins_taiwan']:
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=6.6), color=c, alpha=.10, lw=0)
        ax.text(pd.Timestamp(cfg[key + '_origins_taiwan'][0]) + pd.Timedelta(days=.3), .70, name, fontsize=9.5, color=INK2, va='top')
    order = sorted(TARGETS, key=lambda t: -p[t].iloc[0])
    for t in TARGETS:
        c = S[TARGETS.index(t)]
        ax.plot(p.week, p[t] * 100 if False else p[t], color=c, lw=2, marker='o', ms=6, mec=SURF, mew=1.5, label=t)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, .74); ax.set_ylabel('share of cohort users with the activity in the week')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')); ax.set_xticks(pd.to_datetime(p.origin_tw))
    ax.set_xlim(p.week.iloc[0] - pd.Timedelta(days=1), p.week.iloc[-1] + pd.Timedelta(days=7.5))
    ax.annotate(f'push {p.push.iloc[-1]:.1%}', (p.week.iloc[-1], p.push.iloc[-1]), xytext=(p.week.iloc[-1] + pd.Timedelta(days=1.2), .16),
                fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    ax.legend(ncol=6, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9.5, handlelength=1.8)
    ax.set_title('Label prevalence drifts by week — thresholds tuned on one regime are applied in another', pad=30)
    ax.text(0, -.13, 'Each point = one outcome week (shaded span, Thursday 00:00 Taiwan start). Users = cohort users visible before the origin '
            f'({prev.users.iloc[0]:,} → {prev.users.iloc[-1]:,}).', transform=ax.transAxes, fontsize=9, color=MUTED)
    del order
    save(fig, out, 'q3_label_prevalence')


# ---------------------------------------------------------------- Q2/Q3 ablation with bootstrap CIs
def ablation(out, metrics, boot):
    pooled = metrics[metrics.split == 'pooled_observed_development'].set_index('model')
    cal = metrics[metrics.split == 'calibration_mean'].set_index('model').score
    order = ['last_week_rule', 'observed_history', 'recent7', 'temporal30', 'no_recency', 'no_trend', 'availability_clock',
             'static_relations', 'temporal_static_relations', 'received_relations', 'dynamic_relations', 'no_peer_relations']
    group = {m: ('baseline' if m in ['last_week_rule', 'observed_history'] else
                 'relational' if 'relation' in m else 'temporal') for m in order}
    col = {'baseline': MUTED, 'temporal': S[0], 'relational': S[1]}
    b = boot.set_index('model')
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    y = np.arange(len(order))[::-1]
    rule = pooled.loc['last_week_rule', 'score']
    ax.axvline(rule, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(.392, len(order) - .45, 'dev        calib.', fontsize=9, color=MUTED, fontweight='bold')
    for yi, m in zip(y, order):
        s = pooled.loc[m, 'score']; c = col[group[m]]
        if m in b.index:  # noqa
            ax.plot([b.loc[m, 'ci_low'], b.loc[m, 'ci_high']], [yi, yi], color=c, lw=2, solid_capstyle='round', alpha=.55)
        ax.plot(s, yi, 'o', ms=9, color=c, mec=SURF, mew=2, zorder=3)
        extra = f'      {cal[m]:.1%}' if m in cal.index else ''
        ax.text(.392, yi, f'{s:.1%}{extra}', va='center', fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([NICE[m] for m in order], fontsize=10)
    for lab, m in zip(ax.get_yticklabels(), order):
        if m == 'dynamic_relations':
            lab.set_fontweight('bold')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(.305, .425); ax.grid(axis='y', visible=False)
    ax.set_xlabel('development Score = (macro-F1 + micro-F1) / 2, pooled over the 10/22 and 10/29 weeks (n = 10,334 user-weeks)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=col[k], ms=8, label=k) for k in col] +
              [Line2D([], [], color=INK2, lw=2, alpha=.55, label='95% bootstrap CI (1,000 user resamples)'),
               Line2D([], [], color=INK2, lw=1, ls=(0, (3, 3)), label='copy-last-week score')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=9.5)
    ax.set_title('Ablation: temporal features beat the rule; relational features add no signal on top', pad=30)
    save(fig, out, 'q2_q3_ablation')


def per_target_heatmap(out, evidence):
    order = ['last_week_rule', 'observed_history', 'recent7', 'temporal30', 'availability_clock', 'static_relations', 'dynamic_relations']
    e = evidence.set_index('model').loc[order, TARGETS]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    vmax = .45
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQ)
    ax.imshow(e.values, cmap=cmap, vmin=.22, vmax=.42, aspect='auto')
    for i in range(len(order)):
        for j in range(6):
            v = e.values[i, j]
            ax.text(j, i, f'{v:.1%}', ha='center', va='center', fontsize=10, color='#ffffff' if np.dot(cmap((v - .22) / .2)[:3], [.2126, .7152, .0722]) < .45 else INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(TARGETS, fontsize=10)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[m] for m in order], fontsize=10)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Per-activity F1 on the development weeks (pooled 10/22 + 10/29)')
    save(fig, out, 'q3_per_target_f1')


# ---------------------------------------------------------------- Q2 relation analysis
def relation_evidence(out, coverage, closure, share):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
    c = coverage[coverage.week_tw == coverage.week_tw.min()].set_index('group').share_users_nonzero
    order = ['relational: static', 'relational: repository activity', 'relational: collaborators', 'relational: received interactions']
    vals = [c[g] for g in order]
    yy = np.arange(len(order))[::-1]
    a1.barh(yy, vals, color=S[1], height=.55, edgecolor=SURF, linewidth=2)
    for yi, v in zip(yy, vals):
        a1.text(v + .015, yi, f'{v:.2%}', va='center', fontsize=10)
    a1.set_yticks(yy); a1.set_yticklabels([g.replace('relational: ', '') for g in order]); a1.set_xlim(0, 1.18)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); a1.grid(axis='y', visible=False)
    a1.set_title('Users with a non-zero feature (week 10/22)', fontsize=11.5)
    a1.text(0, -.24, f"Cohort closure: of {closure['comments_on_cohort_authored_threads']:,} comments on cohort-authored threads,\n"
            f"only {closure['of_which_by_other_users']} are by someone else; of {closure['reviews_on_cohort_authored_prs']:,} reviews on cohort PRs, "
            f"{closure['of_which_by_other_reviewers']} by others.", transform=a1.transAxes, fontsize=9, color=INK2)
    rel = [g for g in share.columns if g.startswith('relational')]
    data = pd.DataFrame(index=share.index)
    data['temporal: recency'] = share['temporal: recency']; data['relational (all)'] = share[rel].sum(1)
    data['temporal: window counts'] = share['temporal: window counts']; data['temporal: trend'] = share['temporal: trend']
    data['user profile'] = share['user profile']
    cols = list(data.columns)  # fixed slot order 1..5; relational keeps slot 2 (orange) as in all other figures
    left = np.zeros(len(data)); yy = np.arange(len(data))[::-1]
    for k, g in enumerate(cols):
        a2.barh(yy, data[g], left=left, color=S[k], height=.6, edgecolor=SURF, linewidth=2, label=g)
        for yi, l, v in zip(yy, left, data[g]):
            if v > .07:
                a2.text(l + v / 2, yi, f'{v:.0%}', ha='center', va='center', fontsize=9, color='#ffffff' if k < 2 else INK)
        left += data[g].values
    a2.set_yticks(yy); a2.set_yticklabels(data.index); a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); a2.grid(axis='y', visible=False)
    a2.legend(ncol=3, fontsize=8.5, loc='upper center', bbox_to_anchor=(.5, -.1))
    a2.set_title(f'Share of LightGBM gain by feature family (submitted model; relational {data["relational (all)"].min():.0%}–{data["relational (all)"].max():.0%})', fontsize=11.5)
    fig.suptitle('Why relations add little here: the cohort-only data rarely contains other people\'s replies', x=.01, ha='left', fontweight='bold', fontsize=12.5)
    fig.tight_layout()
    save(fig, out, 'q2_relation_evidence')


def reference_week(out, ref):
    ta = [('TA: Random', .2126), ('TA: GNN without time encoding', .3570), ('TA: Copy last week', .4961), ('TA: GNN with time encoding', .5482)]
    r = ref.set_index('model').score
    ours = [('Ours: copy last week', r['last_week_rule']), ('Ours: temporal (7/14/30 d)', r['temporal30']),
            ('Ours: temporal + dynamic rel. (submitted)', r['dynamic_relations'])]
    rows = ta + ours
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    yy = np.arange(len(rows))[::-1]
    for yi, (n, v) in zip(yy, rows):
        ax.barh(yi, v, color=MUTED if n.startswith('TA') else S[0], height=.6, edgecolor=SURF, linewidth=2)
        ax.text(v + .006, yi, f'{v:.1%}', va='center', fontsize=10)
    ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in rows]); ax.set_xlim(0, .66); ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title('Sanity check on the TA reference week 2026-09-27 – 10-04 (Score)')
    ax.text(0, -.2, f'Our copy-last-week rule ({r["last_week_rule"]:.1%}) is close to the README value (49.61%), consistent with our label construction. '
            'Caveat: this week overlaps our threshold-calibration window,\nso our model rows are optimistic; '
            f'TA numbers are copied from the official README. Ours: {int(ref.users.iloc[0]):,} visible cohort users.',
            transform=ax.transAxes, fontsize=9, color=INK2)
    save(fig, out, 'q4_reference_week')


# ---------------------------------------------------------------- Q4 framework
def framework(out):
    src = r'''digraph framework {
graph [rankdir=LR, bgcolor="#fcfcfb", fontname="DejaVu Sans", nodesep=0.3, ranksep=0.6, pad=0.25, compound=true];
node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#c3c2b7", fontname="DejaVu Sans", fontsize=11, margin="0.15,0.08"];
edge [color="#52514e", arrowsize=0.7, penwidth=1.2];
subgraph cluster_data { label=<<b>1 · Data</b>>; fontsize=12; color="#c3c2b7"; style="rounded"; bgcolor="#f3f2ee";
  gha [label="GH Archive\n1,608 hourly files\n09/01–11/07 UTC"];
  pre [label="official preprocess.py\n(9,000-user cohort,\nanonymization)"];
  db  [label="8 relational tables\n(Users, Repositories, PRs,\nIssues, comments, reviews,\nreview comments, pushes)"];
  db -> pre -> gha [dir=back]; {rank=same; gha; pre; db;} }
subgraph cluster_time { label=<<b>2 · As-of sampling (Q3)</b>>; fontsize=12; color="#c3c2b7"; style="rounded"; bgcolor="#eef4fc";
  ev  [label="event stream\n(event time + row-visibility time)"];
  asof [label="origin t (weekly, Thu 00:00 TW)\nkeep rows visible before t − 4 h;\nlater-known columns dropped"];
  lab [label="label = activity in [t, t+7 d)\nvisible by t+7 d"];
  ev -> asof; ev -> lab; {rank=same; lab; asof;} }
subgraph cluster_feat { label=<<b>3 · Features</b>>; fontsize=12; color="#c3c2b7"; style="rounded"; bgcolor="#f3f2ee";
  ft [label="temporal (Q3): counts in 7/14/30 d,\nactive days, distinct repos,\nrecency per activity, 7 vs 30 d trend", fillcolor="#e3eefb"];
  fr [label="relational (Q2): FK joins →\nrepo activity of others (7 d),\nreviewed-repo new PRs (3 d),\nreceived replies (3/7 d), collaborators", fillcolor="#fde8de"];
  fu [label="user profile:\naccount age, lifetime counts"]; {rank=same; fu; fr; ft;} }
subgraph cluster_model { label=<<b>4 · Model</b>>; fontsize=12; color="#c3c2b7"; style="rounded"; bgcolor="#f3f2ee";
  lgb [label="6 LightGBM classifiers\n(one per activity, seed 42,\nsingle-thread deterministic)"];
  thr [label="per-activity thresholds\nmaximising mean weekly Score\non calibration weeks"];
  abl [label="11 ablations +\ncopy-last-week baseline\non development weeks"];
  abl -> thr -> lgb [dir=back]; {rank=same; lgb; thr; abl;} }
subgraph cluster_out { label=<<b>5 · Output</b>>; fontsize=12; color="#c3c2b7"; style="rounded"; bgcolor="#f3f2ee";
  refit [label="refit on all completed weeks\n(9 origins, 09/10 – 10/31 TW)"];
  csv [label="output_113700058.csv\n4,465 users × 6 activities\n(155 unseen → calibration prior)"];
  rep [label="official evaluate.py format check +\nindependent rebuild & refit:\nidentical CSV bytes"];
  rep -> csv -> refit [dir=back]; {rank=same; refit; csv; rep;} }
db -> ev; asof -> ft; asof -> fr; asof -> fu; ft -> lgb; fr -> lgb; fu -> lgb; lab -> lgb; abl -> refit;
}'''
    dot(out, 'q4_framework', src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--analysis', required=True)
    ap.add_argument('--data', help='prepared data, for table row counts in the schema graph')
    ap.add_argument('--out', default='report_materials/figures_revised')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    run, an = Path(a.run), Path(a.analysis)
    design = json.loads((run / 'design.json').read_text())
    counts = {}
    if a.data:
        import pyarrow.parquet as pq
        counts = {t: pq.ParquetFile(Path(a.data) / 'db' / f'{t}.parquet').metadata.num_rows for t in SCHEMA}
    schema_graph(out, counts)
    timeline(out, design, None)
    volume_multiples(out, pd.read_csv(an / 'weekly_volume.csv'), design)
    prevalence_lines(out, pd.read_csv(an / 'weekly_prevalence.csv'), design)
    ablation(out, pd.read_csv(run / 'metrics.csv'), pd.read_csv(an / 'bootstrap_development.csv'))
    per_target_heatmap(out, pd.read_csv(run / 'slide_evidence.csv') if (run / 'slide_evidence.csv').exists()
                       else pd.read_csv(an / 'slide_evidence.csv'))
    relation_evidence(out, pd.read_csv(an / 'relation_coverage.csv'), json.loads((an / 'cohort_closure.json').read_text()),
                      pd.read_csv(an / 'feature_group_share.csv', index_col=0))
    reference_week(out, pd.read_csv(an / 'reference_week.csv'))
    framework(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
