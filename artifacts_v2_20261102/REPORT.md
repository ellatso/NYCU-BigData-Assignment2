> **Coverage correction (Nov2):** These archived scores use weeks with target-specific coverage warnings. Use [the corrected common-period report](../artifacts_coverage_20261102/REPORT.md) for primary comparisons. Do not infer reliable future-performance gains or attribute all declines to thresholds from these tables alone.

# Second-round temporal features and push-coverage sensitivity

Executed 2026-11-02 on the same Nov1 official data snapshot as the baseline audit. All results below are historical development diagnostics against supplied records. No official November prediction or score is claimed.

## Fixed experiment

Six recipes were specified before development scoring: baseline; baseline with push-training exclusion; short-term temporal features, with/without exclusion; temporal plus ongoing-thread features, with/without exclusion. Classifiers, seed, hyperparameters, training users and splits remain those of the baseline. Model/threshold calibration uses Sep24 only. The entire recipe and data SHA256 are in `design.json`; feature implementation and runner hashes are in `implementation.json`.

Training: May7–Sep10 origins, 148,917 user-week rows. Four-hour observation gap, seven-day labels. Calibration: Sep24. Previously inspected development origins: Oct8, Oct15, Oct22 (Taiwan midnight). Oct29–Nov5 outcomes were not scored, and no thresholds were tuned on development labels. Historical feature/label reconstruction still has the data-coverage caveats documented in the first audit.

New temporal features (54 columns): per-target 1/3-day counts, active weeks over eight weeks, consecutive active weeks, prior-week counts, 7/28-day half-life weighted counts and inter-event mean/std. This increases temporal-model inputs from 51 to 105.

New thread features (12 columns): for issue discussions, reviews and line comments, identify threads the user personally touched in the previous 14 days; count other actors' events and distinct live threads over 3/7 days. This does not claim to count every authored PR or every interaction across event types. Both row availability and occurrence, parent availability, user and repo endpoints are filtered before the feature cutoff. All later-known merged/closed fields remain unused.

## Coverage sensitivity

A day is flagged heuristically if it has zero pushes, some other activity, and a previous-28-day median of at least 100 pushes (minimum 14 prior days). Days use Taiwan time. This is not proof that records are missing. The mask excludes only push-model training rows whose seven-day outcome contains such a day; other five classifiers retain all training samples. It does not rewrite zeros as positives or remove suspicious periods from development scoring.

In the current training split, Aug27 and Sep3 outcome weeks are affected. Push training rows fall from 148,917 to 136,681. Calendar features are not added from future coverage flags. See `suspect_push_days.json` and `design.json`.

## Measured combined Score

Each column aggregates its user-week predictions before computing the official `(macro-F1 + micro-F1)/2`; it is not an unweighted average of weekly F1 scores. All percentages are rounded here; exact values and six per-target F1 scores are in `metrics.csv`.

| Recipe | Calibration Sep24 | Original Oct8/15 diagnostics | All three development weeks | Oct22 only |
|---|---:|---:|---:|---:|
| Baseline temporal | 55.81% | 50.17% | 48.75% | 45.39% |
| Baseline + push exclusion | 55.79% | 50.19% | 48.80% | 45.53% |
| Short temporal | 56.01% | 49.74% | 48.71% | 46.30% |
| Short temporal + push exclusion | 56.07% | 49.68% | 48.58% | 45.99% |
| Short temporal + threads | 56.01% | 49.74% | 48.71% | 46.30% |
| Short temporal + threads + push exclusion | 56.07% | 49.68% | 48.58% | 45.99% |

The baseline reproduces the earlier measurements. Calibration alone nominates `short_temporal_pushmask` (ties broken by predetermined recipe order), but it has a worse three-week development Score than the original baseline. Its saved status is experimental only; it has not replaced the official prediction path. The slight improvement from baseline push exclusion is only about 0.06 percentage points in the combined development Score; no confidence or generalization claim is warranted.

Short features improve Oct22 but hurt Oct8/15. These diagnostics are already inspected and cannot support a claim of fresh out-of-sample superiority. Push coverage in Oct22 remains suspicious, and calibration itself is only one week.

## Why the thread ablation is identical

Each thread feature is positive in only 8–36 of the 148,917 training rows. The fitted trees make **zero splits on all 12 thread features**, so this feature addition changes no predictions. This is a measured property, not a claim that relationships never help. Support counts and split counts are saved in `thread_support.json`. There is insufficient support for this particular shared-conversation design at the current tree settings; lowering minimum leaf size to fit a handful of rows would require a separate overfitting-controlled experiment.

## Validation and reproducibility

- Three new tests passed: self/future thread events excluded, week-end-exclusive push masking, and the prior-reference coverage flag.
- Two existing ID-mapping and label-availability regression tests passed.
- Every reported score was checked against the official evaluator, including its empty-class convention.
- Syntax and whitespace checks passed.
- Six saved checkpoints, cached matrices and per-user probabilities stay local and are excluded from git. Aggregate metrics, exact thresholds, recipe, hashes and log are committed.

```bash
python -m unittest discover -s tests -p test_baselines.py -v
python -m unittest discover -s tests -p test_enhanced.py -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/experiment_v2.py \
  --data ../official --baseline artifacts_20261102 --out artifacts_v2_20261102
```

The runner verifies that data hashes match the first audit. Caches are keyed by data and enhanced-feature source hash. Source-code changes require a new output directory to preserve old experiment evidence.

## Decision

Retain the original temporal model as the established baseline; these additions do not demonstrate stable superiority. Preserve the Oct29 outcome week from further development scoring. The feature/coverage experiments provide genuine Q2/Q3 ablations for the report, including negative results. A later iteration should prioritize broader, better-supported relation features and multi-week historical calibration with a chronologically separate evaluation; neither was claimed to have run here.
