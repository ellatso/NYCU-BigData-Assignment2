# Data audit and baseline rerun — 2026-11-02

Completed against official release `cfe57edec97a4c4408053a71fd0fdb9d746930dc` (Nov 1). This is a historical experiment, not the November submission score. Source project: `8fa06c6` plus the explicit-ID mapping fix in this branch.

## Data checks

- Eight tables, 2,581,305 total rows, 9,000 cohort users. Primary-key uniqueness/null checks and all declared foreign-key endpoint checks passed.
- Official test roster: 3,960 unique, known cohort users. Reconstructing the instructor's 90-day / at least five records selection at Oct 28 20:00 Taiwan gives exactly the same roster.
- SHA256 and sizes of all eight files: `data_manifest.json`. Table coverage, environment versions, checks and detailed warnings: `audit.json`.
- Significant coverage warning: Pushes ends Oct 26 18:06:14 Taiwan, while other targets extend to Nov 1. Taiwan Oct 27–Nov 1 contain zero recorded pushes; Oct 16–17 and Sep 2–4 also have zero. Oct 23 has only 13 pushes. Zero counts are observed facts, not proof of missing collection. Do not impute positive labels or interpret these periods as verified inactivity.
- See `daily_event_counts.csv` (Taiwan days; final day partial) and `event_coverage.csv` (UTC timestamps). Global max timestamp alone cannot establish complete labels for all six targets.

## Reproduced baseline experiment

Unchanged model settings: six LightGBM classifiers, seed 42, 160 trees. Train origins May 7–Sep 10; calibration Sep 24; previously inspected development diagnostics Oct 8/15. Four-hour observation gap at every origin; seven-day outcome windows. Thresholds and model selection use calibration only.

| Model | Macro-F1 | Micro-F1 | Combined Score |
|---|---:|---:|---:|
| Last-week rule | 40.39% | 52.26% | 46.32% |
| Lifetime user features | 36.84% | 48.61% | 42.73% |
| Temporal user features | 44.55% | 55.79% | 50.17% |
| Static relational features | 37.33% | 49.53% | 43.43% |
| Temporal + relational features | 44.14% | 55.44% | 49.79% |

The rerun reproduces the earlier reported results. These two diagnostic weeks are already inspected and must not be described as fresh test data. All historical labels are reconstructed from the supplied snapshot, including the coverage limitations above.

## Additional chronological evaluation

No parameters or thresholds were adjusted after viewing new-week outcomes. Candidate definitions are in `recheck_design.json`, written before scoring. Models were refit on historical training windows from the latest snapshot; original frozen October model binaries were not available in this checkout. This is therefore a retrospective evaluation of fixed recipes, not a preserved-checkpoint prospective experiment.

The main population is the existing recent-90-day eligibility proxy. A supplementary population selects users **14 days before the forecast origin**, at 20:00 Taiwan, then retains that roster, matching the official roster selection lag. It does not use the actual future official test roster for historical scoring.

Oct 22–29 historical week, combined Score:

| Model | Existing eligibility proxy (4,107 users) | Official-style selection lag (4,441 users) |
|---|---:|---:|
| Last-week rule | 43.21% | 43.10% |
| 12 full historical weeks' frequency >= 0.5 | 44.87% | 44.99% |
| Lifetime features | 37.94% | 36.86% |
| Temporal features | 45.39% | 45.28% |
| Static relational features | 38.49% | 37.47% |
| Temporal + relational | 46.32% | 46.23% |

Exact unrounded values, all-zero/all-one controls, per-target scores, all three scored origins and both populations are in `recheck_metrics.csv`; label rates are in `label_prevalence.csv`. Every historical prediction file passed official format checks and all computed metrics agreed with the official evaluator.

**Coverage caveat:** Oct 22–29 includes the suspicious zero-push period. These are provisional scores against supplied records, not evidence of fully observed future behavior. The full model leads on this week but trails temporal-only in the earlier combined diagnostics. No robust superiority is established; the calibration-selected model remains temporal-only. Do not switch merely because this week favors a different candidate.

Oct 29–Nov 5 is PENDING, because the Nov 1 release does not cover its full outcome horizon. No partial-week score was calculated.

## Correctness fix and executed validation

`features(..., user_ids=...)` could lose the `github_user_id` index name during pandas selection and fail at output-ID creation. Explicitly restoring the index name fixes the official-ID prediction path without changing default training-population features.

- Two focused regression tests passed: explicit list/array/Series IDs preserve the requested stable-ID order and counts; a historical label excludes an event learned after the outcome window.
- Real-data checks passed: injected future event occurrence/availability does not change past features; stable output IDs; exact official scoring including empty classes; nonoverlapping training/calibration/diagnostic windows.
- `py_compile` and `git diff --check` passed. The entire final November refit remains untested until the final release exists.
- The GNN was not retrained or selected in this baseline audit.

## Reproduce

From the repository root, install `requirements.txt`, then clone the official data into a separate sibling directory. Preserve its commit and file hashes; the instructor's repository changes daily.

```bash
python scripts/audit_data.py --data ../official --out artifacts_20261102
python -m asg2.run benchmark --data ../official --out artifacts_20261102
python scripts/checks.py --data ../official --out artifacts_20261102
python -m unittest discover -s tests -p test_baselines.py -v
python scripts/recheck_baselines.py --data ../official --out artifacts_20261102
```

Use a fresh output directory for a different data release or changed feature implementation. Models, cached samples and per-user prediction files remain local; aggregate evidence is committed.

## Next decisions

1. Verify the suspicious push coverage with the course staff or a later official release before interpreting recent six-target scores. Use only official data.
2. Keep candidate definitions fixed for the Oct 29 outcome week. Check target-level coverage before evaluating it after Nov 5.
3. Future feature/threshold improvements need separate development weeks and a later untouched evaluation; Oct 22 has now been inspected.
4. Final submission still requires the Nov 11 release, official-format CSV, English slides and a personally recorded video.
