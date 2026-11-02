# Coverage correction, dynamic relations and push fallback

Update: [the rule-mixture and backup follow-up](../artifacts_hybrid_20261102/REPORT.md) adds new development results and an explicit early-output override. The original measured comparisons below remain unchanged.

This report replaces the earlier primary model comparisons. The old results remain archived as scores against supplied records, but their coverage-affected weeks must not support claims of reliable future behavior.

## Advice adopted and qualified

1. **Coverage is the first priority.** The official main ref was checked again on November 2 and is still `cfe57edec97a4c4408053a71fd0fdb9d746930dc`. Push records end October26 18:06:14 Taiwan while other targets continue into November1. No backfill is observed in this release. Whether zeros represent unobserved actions requires TA confirmation.
2. Removing **only October22 is insufficient**. October15–22 has zero push days on Oct16/17. October8–15 has zero PR line-comment records on Oct9. These weeks are removed from the primary comparison. A zero-day screen is a warning heuristic, never proof of missing collection or confirmed completeness.
3. Thresholds now maximize the **mean per-week official Score** across screened calibration weeks. Fixed top-k positive-rate prediction is tested as a separate method. Neither is assumed superior.
4. The final path refits forward. The original final command already supported forward refitting. It now additionally includes the latest completed Taiwan-midnight seven-day window, excludes flagged push-label training rows, checks release freshness and supports a validated fallback. With the final release, the latest extra training origin can be Nov4 at Taiwan midnight, with labels ending Nov11 midnight. No future label enters training.
5. Dynamic relation features are implemented and tested. The earlier model was not entirely static: it already had 30-day relation aggregates. The new features add shorter interaction windows and more specific roles. They are not forced into the final model when calibration does not select them.
6. The revised PDF has one connected entity graph with every user node appearing once, consistent main evaluation dates, simpler language and no visible internal source-code paths.

## Uniform historical protocol

- Training origins: May7 through Aug6, fourteen weekly origins, 115,448 user-week samples. Training labels finish before the first calibration feature cutoff.
- Calibration origins: Aug20 and Sep10, seven-day outcome windows.
- Evaluation origins: Sep24 and Oct1, covering Sep24–Oct8.
- Each listed calibration/evaluation outcome week has no zero-record day among the six targets in this release. This does not prove complete coverage. Feature lookbacks may still cross historical gaps.
- Every origin simulates a four-hour observation gap. Training and calibration labels finish before the next split's feature cutoff.
- Sep24 was used in earlier experiments, so this is a revised common historical development comparison, not an untouched test.
- Mean weekly Score chooses thresholds and the candidate. The main table pools predictions from the same two evaluation weeks. The metric distinction is explicit.

## Dynamic relation design

The new candidate adds 32 columns to the 51 temporal/user inputs:

- Comments, reviews and line comments received on the user's authored PR/issue in the past 3/7 days, plus distinct interacting partners. Parent, actor, repo and event visibility are checked. Self interactions are excluded.
- New PRs by other authors in the past three days on repositories reviewed by the user in the past 90 days.
- Mean/max seven-day activity in repositories the user touched in the previous 90 days, excluding own event counts, separately for all six targets.
- Seven-day activity of direct actor/author collaborators. Peer activity totals use cohort peers whose activity is collected completely by design. Non-cohort inactivity is not assumed.

The code uses observed interactions and does not infer unresolved PR/issue state from future closed/merged metadata. A recent interaction is a proxy for continuing work, not proof that an object is still open. Feature support is recorded in `dynamic_support.csv`. The most populated new columns are positive for thousands of training rows, unlike the earlier sparse same-thread features.

The retained temporal model already joins the activity tables to users/repositories and includes repository diversity. It has no learned GNN message passing. The additional neighborhood features are an ablation.

## Shared evaluation results

| Method | Macro-F1 | Micro-F1 | Combined Score |
|---|---:|---:|---:|
| temporal_mean_threshold | 46.60% | 54.20% | 50.40% |
| temporal_rank_prevalence | 48.69% | 52.54% | 50.61% |
| previous_relational_mean_threshold | 46.46% | 54.15% | 50.31% |
| previous_relational_rank_prevalence | 48.33% | 52.32% | 50.32% |
| dynamic_relational_mean_threshold | 46.25% | 54.00% | 50.12% |
| dynamic_relational_rank_prevalence | 48.04% | 52.19% | 50.11% |
| last_week_rule | 45.45% | 57.30% | 51.37% |

The temporal threshold model wins mean calibration Score at 45.65%, versus the last-week rule at 43.93%. The last-week rule is strongest on evaluation at 51.37%. Top-k also slightly exceeds the selected learner on evaluation, but its calibration Score is lower. No method is switched merely because its evaluation number is largest. This replaces the earlier blanket statement that temporal learning beats the rule.

Dynamic relations with mean thresholds score 50.12%, compared with 50.40% for temporal inputs. Their calibration Score is also lower. They remain implemented and documented for Q2, but are not retained just to improve the narrative. Two screened evaluation weeks cannot establish universal ineffectiveness of relations or stable superiority of any method.

## Reanalysis of the earlier threshold claim

`old_score_reanalysis.csv` removes Oct22 and separately reports four/five non-push target diagnostics. With Oct22 removed, the recorded five-target non-push diagnostic is 40.85% for the original single-week model and 30.49% for the multiweek-selected weighted/frequency candidate. These are **not six-target official Scores**. Excluding both push and line-comment targets also leaves a decline.

Therefore push contamination cannot explain the entire observed difference. However, the earlier phrase “threshold transfer failed” was too definitive as a general result. On the same recorded examples, changing thresholds changes false positives, but incomplete collection and the historical development design limit conclusions about true future accuracy. The earlier 38.81% and mixed-date tables have been removed from the revised PDF's primary results.

## Conditional push fallback

The fixed fallback uses the fraction of twelve observation weeks with push activity, with its clock ending at the last observed push when recent records stop. Threshold 0.60 is selected from calibration push labels only. The last observed push timestamp is not a verified complete-data boundary.

Simulations remove **only push records from features** while preserving the original outcome labels and evaluation user sets. No true positive is deliberately relabeled zero.

| Simulated missing recent push days | Naive push F1 | Last-observed-clock frequency F1 |
|---|---:|---:|
| 0 | 71.14% | 58.95% |
| 2 | 67.46% | 58.45% |
| 7 | 55.45% | 59.47% |
| 14 | 39.99% | 61.00% |
| 21 | 24.86% | 62.98% |

These are mean **push F1** values across the same two evaluation weeks, not the six-target Score. The frequency method is worse with healthy records and with a two-day dropout. It is therefore not always enabled. The selected engineering policy warns after 48 hours, and activates fallback only after at least seven days without observed pushes. It improved all simulated 7/14/21-day cases. This is stress-test evidence for trailing feature gaps, not validated knowledge of the real gap's cause or a correction of historical internal gaps.

`fallback_config.json` records the threshold, activation delay, tested scope and adoption check. Without a validated policy, a stale-push final prediction fails explicitly instead of silently using a stale recency vector. Other five decisions stay with their classifiers.

## Production changes and checks

- Final refitting uses all completed weekly windows and the latest extra completed Taiwan-midnight window.
- Flagged push outcome weeks are omitted only from the push head. The other heads retain those samples. A regression check confirms target-specific exclusion does not change the other five fitted heads.
- The release must reach within two hours of the final observation cutoff before the final prediction command proceeds. The current November1 snapshot is correctly refused for a November12 final CSV.
- The current 3,960-user rehearsal passes the official evaluator and preserves exact ID order. No official final CSV or future score is generated.
- New and earlier tests cover time filtering, visible FK parents, partner exclusion, rank tie handling, per-target label exclusion, push fallback clock and latest complete origins.
- Real-data as-of checks and official metric equivalence passed. Exact results are in `tests.log`, `checks.json`, `official_equivalence.json` and `production_checks.json`.

## Final command after the final official release

```bash
python -m asg2.run predict --data ../assignment2_data \
  --out artifacts_coverage_20261102 \
  --input ../assignment2_data/evaluation/test_input.csv \
  --push-policy-config artifacts_coverage_20261102/fallback_config.json \
  --student-id 113700058
```

Keep the calibration-selected recipe and thresholds fixed when refitting. Check daily coverage and TA guidance again before final submission. A later backfill can change historical reconstructed labels and should be documented rather than silently mixed into this report.

## TA question

`TA_question_zh.md` contains a ready-to-post question with dates, the official commit and the requested clarification about backfill and test truth. It has not been posted or sent.

## Reproduce this correction

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/experiment_coverage.py --data ../assignment2_data
python scripts/checks.py --data ../assignment2_data --out artifacts_coverage_20261102
python -m unittest discover -s tests -p test_coverage_dynamic.py -v
```

Use a fresh output directory after feature-code changes. The archived model binaries and per-user prediction files stay private. Aggregate results and all model-building code are public.
