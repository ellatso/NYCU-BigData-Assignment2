"""Audit a single official release. Missing events are warnings, not invented labels."""
import argparse, hashlib, importlib.metadata, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pyarrow.parquet as pq
from asg2.core import TABLES, PK, FKS, TIME, load_tables, build_events, TARGETS
from asg2.run import fingerprint

def audit(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data)
    summary, errors = {}, []
    for name in TABLES:
        # Full read is needed to check ALL foreign keys, including excluded model fields.
        d = pd.read_parquet(data/'data/db'/f'{name}.parquet')
        bad_fk = {}
        for col, parent in FKS[name].items():
            v = d[col].dropna()
            n = int((~v.isin(tables[parent][PK[parent]])).sum())
            if n: bad_fk[col] = n
        duplicate_pk = int(d[PK[name]].duplicated().sum())
        null_pk = int(d[PK[name]].isna().sum())
        if bad_fk or duplicate_pk or null_pk: errors.append(name)
        summary[name] = dict(rows=len(d), columns=len(d.columns), duplicate_pk=duplicate_pk,
            null_pk=null_pk, invalid_foreign_keys=bad_fk, row_time=TIME[name],
            row_time_nulls=int(d[TIME[name]].isna().sum()),
            min_time=str(d[TIME[name]].min()), max_time=str(d[TIME[name]].max()))
    events = build_events(tables)
    # Taiwan calendar days; retain partial final day and flag it separately.
    day = (events.event_time + pd.Timedelta(hours=8)).dt.floor('D')
    daily = events.assign(day_taiwan=day).groupby(['day_taiwan','kind']).size().unstack(fill_value=0)
    daily = daily.reindex(pd.date_range('2025-12-01', day.max()), fill_value=0).reindex(columns=TARGETS, fill_value=0)
    daily.index.name = 'day_taiwan'
    daily.to_csv(out/'daily_event_counts.csv')
    last = events.groupby('kind').agg(events=('kind','size'), last_occurrence_utc=('event_time','max'),
        last_availability_utc=('available_at','max'))
    last.to_csv(out/'event_coverage.csv')
    zero_days = {k: [str(x.date()) for x in daily.index[(daily[k] == 0) & (daily.index >= '2026-09-01')]] for k in TARGETS}
    ids = pd.read_csv(data/'evaluation/test_input.csv').UserID
    u = tables['Users']; lookup = u.set_index('github_user_id')
    selection_cutoff = pd.Timestamp('2026-10-28 12:00:00')
    past = events[(events.event_time < selection_cutoff) & (events.available_at < selection_cutoff)
                  & (events.event_time >= selection_cutoff-pd.Timedelta(days=90))]
    counts = past.groupby('user_id').size()
    eligible = u[u.is_cohort & u.user_id.isin(counts[counts >= 5].index)]
    expected = set(eligible.github_user_id); supplied = set(ids)
    test = dict(rows=len(ids), duplicates=int(ids.duplicated().sum()),
        unknown_ids=int((~ids.isin(lookup.index)).sum()),
        all_cohort=bool(lookup.reindex(ids).is_cohort.fillna(False).all()),
        reconstructed_selection_count=len(expected), supplied_not_reconstructed=len(supplied-expected),
        reconstructed_not_supplied=len(expected-supplied),
        note='Selection reconstruction can differ because rows arrive retrospectively.')
    if test['duplicates'] or test['unknown_ids'] or not test['all_cohort']: errors.append('test_users')
    result = dict(tables=summary, structural_errors=errors, test_users=test,
        cohort_users=int(u.is_cohort.sum()), duplicate_github_user_ids=int(u.github_user_id.duplicated().sum()),
        zero_event_days_since_september=zero_days,
        warnings=['Zero recorded events do not prove collection failure or true inactivity.',
                  'Historical truth is reconstructed from this release, not TA private truth.',
                  'Latest per-target timestamps differ; recent six-target scores are provisional.',
                  'Final calendar day is partial.'],
        official_commit=subprocess.check_output(['git','-C',str(data),'rev-parse','HEAD'],text=True).strip(),
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','lightgbm','scikit-learn']})
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    (out/'data_manifest.json').write_text(json.dumps(fingerprint(data),indent=2))
    print(json.dumps(dict(structural_errors=errors,test_users=test,coverage=last.astype(str).to_dict('index')),indent=2))
    if errors: raise ValueError(f'Structural data checks failed: {errors}')

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.data,a.out)
