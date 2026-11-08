# Rule mixtures and a reliable early-submission path

## What changed

The earlier calibration-selected temporal classifier lost to the last-week rule on Sep24/Oct1, 50.40% versus 51.37%. That is a real limitation of the earlier recipe, not evidence that a learner should always replace a rule. This follow-up tests four small rule mixtures and implements an explicit backup output path. It does not tune to a target evaluation Score or claim an untouched test.

## Same-period results

Every method uses the same 9,597 historical user-week examples and six-target metric. Calibration is Aug20/Sep10; development origins are Sep24/Oct1. The calibration column is mean weekly Score; the development column pools both outcome weeks.

| Method | Calibration Score | Development Score | Development micro-F1 |
|---|---:|---:|---:|
| Last-week rule | 43.93% | 51.37% | 57.30% |
| Original temporal thresholds | 45.65% | 50.40% | 54.20% |
| Dynamic relations, thresholds | 45.53% | 50.12% | 54.00% |
| Temporal, target-wise model/rule selection | 46.62% | 50.35% | 54.89% |
| Relations, target-wise model/rule selection | 46.48% | 50.56% | 55.24% |
| Temporal probability mixture | 47.49% | 52.25% | 56.32% |
| Relational probability mixture | 47.41% | 53.17% | 57.57% |

Both probability mixtures beat the simple rule on these reused development weeks. Target-wise switching alone does not. Temporal mixing wins calibration among the newly defined candidates; the larger development number does not automatically choose the relational mixture.

These recipes were introduced after inspecting the original development results, so the new results are explicitly development diagnostics. They must not be presented as fresh test performance, statistical significance, or an official November score. `selection.json` freezes all four configurations and identifies the calibration-selected temporal mixture. The existing production recipe is preserved unless `--hybrid-config` is explicitly supplied. The final November choice can be reconsidered after a documented forward comparison.

## Calibration and model rationale

The six original `t_count7_*` features already contain the information needed to decide whether the user was active last week. Adding a duplicate binary column is not the first missing piece. The new intervention changes the prediction rule directly:

`mixed_probability = (1 - rule_weight) * tree_probability + rule_weight * last_week_indicator`.

Per-target weights use a fixed grid {0, .25, .5, .75, 1}; thresholds start at the existing calibration thresholds and search .10 to .70 by .05. Two coordinate passes optimize mean weekly combined Score. Target-wise switching uses a binary choice per activity and optimizes the same coupled objective, rather than incorrectly assuming the micro-F1 term is separable. Evaluation labels never enter those selector functions.

The selected temporal mixture assigns rule weight .25 to open_pr, issue_comment, pr_review and pr_review_comment; open_issue and push retain weight zero. Its thresholds are .15, .075, .15, .15, .10 and .425 respectively. Therefore “the improvement must come from lowering push thresholds” is not supported by this experiment.

`target_diagnostics.csv` explains why a general “too conservative” claim is incomplete. Across these weeks, original issue_comment predictions have 1,398 false positives versus 515 for the rule, although they miss fewer positives. The model predicts 1,953 comment positives for 757 recorded true positives. Push has the opposite tradeoff: the classifier predicts 4,087 positives versus 4,770 for the rule, but its push F1 is slightly higher. The problem is target-specific precision/recall balance, not a single threshold direction for every target.

The calibration weeks are strongly heterogeneous. Recorded push positive rates are 20.13% and 29.89%, compared with about 47.7% in both development weeks. Recorded open_pr prevalence changes from 1.81% to 22.16% between calibration weeks. This establishes distribution differences in the supplied reconstructed labels; it does not establish their cause or prove complete collection. Single-week-to-other-week selector checks are also unstable (`metrics.csv`). Calibration improvement on two weeks alone is insufficient to establish reliability.

## Relational modeling for Q2

The final pipeline is tree-based relational aggregation, not GNN/RDL. Foreign-key joins are real cross-table operations; repository diversity alone is weaker than explicit neighborhood modeling. The dynamic candidate contributes 32 time-windowed author/repository/collaborator features, and the relational probability mixture now has a positive measured development result relative to the temporal mixture. It is retained as a serious candidate rather than dismissed on the earlier 0.28 percentage-point comparison.

Two weeks do not demonstrate statistical equivalence or reliable superiority. A preference for the relational model can be a documented design choice, but it should not be justified by calling an unmeasured uncertainty interval a statistical tie or relabeling tree aggregation as a GNN.

## Fixed forward comparison

`evaluate_hybrid_forward.py` compares the frozen four mixtures, original temporal classifier and last-week rule on the Oct29 origin when its full outcome week is available. It checks all six outcome targets for zero-record days before scoring. The current result is `PENDING`: the week has not finished in the available release and push coverage fails. No partial-week score is fabricated. A passed zero-day screen still cannot prove complete collection.

```bash
python scripts/evaluate_hybrid_forward.py --data ../assignment2_data
```

Weights and mixture configurations remain fixed for this comparison; new official data may supply outcomes. Persisted source and checkpoint hashes are in `implementation.json`. The exact outcomes and data release must be documented if this forward comparison later informs the final model choice.

## Early CSV and manual override

`predict --force` creates a labeled backup using the latest observed data timestamp as its feature clock, capped at the official final observation cutoff. This avoids treating the days before the final release as observed inactivity. Normal final prediction still checks release freshness and refits on completed windows. Backup refitting uses only completed labels available in its actual snapshot. `--no-refit` is allowed only with `--force` and reuses saved weights for a faster emergency file.

A validated push policy is discovered automatically from the output directory or supplied explicitly. When the policy is absent and push is stale, ordinary prediction refuses to silently continue; `--force` uses an explicitly unvalidated emergency majority-frequency heuristic and records the override. Overrides do not bypass official format checks, duplicate/unknown user checks or exact user order.

CSV generation writes and validates a temporary file, then atomically moves it to its destination. A failed run cannot truncate an earlier valid output. Each backup has a separate timestamped directory and keeps the required final basename `output_113700058.csv`. Its sidecar records actual cutoff, model, mixture policy, forecast lead, refitting and push warnings. These are early forecasts, not an assertion of final accuracy.

The November2 quick backup uses the calibration-selected temporal mixture and saved historical weights. All 3,960 users pass the official evaluator, with exact input order. It has not been submitted to E3. The available snapshot ends Nov1, so it is a much earlier backup than the planned Nov10 file.

```bash
# Recommended Nov10 backup: refit on labels complete in the then-current snapshot.
python -m asg2.run predict --data ../assignment2_data \
  --out artifacts_coverage_20261102 \
  --input ../assignment2_data/evaluation/test_input.csv \
  --hybrid-config artifacts_hybrid_20261102/selection.json \
  --force --student-id 113700058

# If time is too short for refitting, also pass --no-refit.
# Final Nov11 run: omit --force and --no-refit; choose the documented recipe.
```

Check E3's actual replacement behavior before an early submission, and preserve existing slide/video files. Generating a CSV or scheduling a task is not proof of a successful upload; verify the receipt. Nothing has been submitted to E3 in this run.

## Reproduce and verify

```bash
python scripts/experiment_hybrid.py
python scripts/evaluate_hybrid_forward.py --data ../assignment2_data
python -m unittest discover -s tests -p test_hybrid_production.py -v
```

Twenty non-neural regression tests pass, including earlier as-of tests and new override, mixture and failed-write checks. Real-data backup formatting and six-policy metric equivalence with the official evaluator pass. Per-user prediction CSVs remain private; aggregate metrics, configurations, source and checks are public. The presentation is updated to distinguish reused development evidence, fixed candidates and the actual backup implementation.
