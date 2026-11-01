# Multiweek calibration, recent training and frequency blending

Executed November 2, 2026. All scores are historical development diagnostics, not official November test scores. The data snapshot is unchanged from the first audit. Oct29–Nov5 remains unscored.

## Initial fixed protocol

Four calibration origins: Aug13, Aug27, Sep10 and Sep24 at Taiwan midnight. Each origin receives predictions from a separately fitted model using only training weeks whose full seven-day outcomes finish before the origin minus four hours. No random split is used. Eligible training origins come from the fixed May7–Sep10 pool.

Three training recipes use the same original 51 temporal inputs and LightGBM settings:

- `full_history`: every eligible historical origin, 19 origins in the final development model.
- `recent12`: the last 12 eligible origins, 93,339 rows in the final development model.
- `weighted6`: every eligible origin with a six-week half-life by sample age. We normalize weights to mean one.

For each recipe, fixed frequency blend weights are 0%, 25% and 50%. Frequency means the fraction of the previous 12 complete observation weeks with each activity. Tree and frequency probabilities are combined before thresholding. Users, repositories, event occurrence and availability are filtered before the four-hour observation cutoff.

Thresholds maximize the official pooled combined-F1 across the four chronological calibration weeks. This is a threshold-fitting score, not an unbiased validation score. Candidate choice uses this score only, before development scoring. `design.json`, `fold_protocol.json` and `final_training.json` record the exact protocol. Leave-one-calibration-week-out threshold fits in `threshold_stability.csv` are sensitivity diagnostics, not prospective backtests.

The initial calibration choice was `weighted6_freq50`. Its pooled calibration-fit Score was 44.94%. This value cannot be compared directly with the original Sep24-only calibration Score because the scored populations and dates differ.

## Initial development results

| Candidate | Oct8 | Oct15 | Oct22 | Three-week pooled Score |
|---|---:|---:|---:|---:|
| original_singleweek | 53.25% | 46.68% | 45.39% | 48.75% |
| full_history_freq0 | 36.13% | 33.23% | 30.55% | 33.46% |
| recent12_freq0 | 35.20% | 32.73% | 29.50% | 32.63% |
| weighted6_freq0 | 37.28% | 33.97% | 30.65% | 34.14% |
| full_history_freq25 | 37.18% | 34.20% | 31.51% | 34.46% |
| full_history_freq50 | 41.05% | 38.13% | 35.30% | 38.34% |
| recent12_freq25 | 37.16% | 34.77% | 31.26% | 34.56% |
| recent12_freq50 | 40.82% | 37.80% | 34.42% | 37.87% |
| weighted6_freq25 | 38.18% | 35.42% | 32.10% | 35.39% |
| weighted6_freq50 | 41.57% | 38.99% | 35.35% | 38.81% |
| frequency_only_0.5 | 46.97% | 43.11% | 44.87% | 45.12% |

October8, October15 and October22 were already inspected in earlier experiments. They are development data. Recent push-coverage gaps also affect these reconstructed labels. Per-target F1, macro-F1 and micro-F1 are in `metrics.csv`. Pooled Score is computed after concatenating predictions, not by averaging weekly F1.

## What changed when multicalibration failed

The original model and the refitted full-history model produce exactly identical development probabilities: maximum absolute difference is 0.0. Therefore the large full-history decline in this comparison comes from threshold replacement. For example, the issue-comment threshold changes from 0.65 to 0.125. Across the three development weeks, false positives for issue comments rise from 545 to 5,240, while precision falls from 33.3% to 10.1%. PR review false positives rise from 479 to 4,708.

This establishes the measured threshold effect on these development samples. It does not establish a universal cause of poor transfer. Earlier model confidence, label prevalence, changing cohort activity and data coverage may all contribute. `label_prevalence.csv` and `threshold_effect.json` preserve the evidence. Simply pooling more calibration weeks did not improve this setup.

## Explicit follow-up after inspecting initial results

To separate training changes from calibration horizon, a second fixed 3-by-3 comparison reuses the already-trained models but calibrates thresholds on Sep24 only. This diagnostic was added AFTER seeing the initial v3 development results. `diagnostic_amendment.json` records that timing. No new training or tuning on October labels was introduced, and no fresh-test claim is made.

| Candidate | Oct8 | Oct15 | Oct22 | Three-week pooled Score |
|---|---:|---:|---:|---:|
| full_history_single_freq0 | 53.25% | 46.68% | 45.39% | 48.75% |
| full_history_single_freq25 | 53.11% | 47.20% | 47.57% | 49.51% |
| full_history_single_freq50 | 53.10% | 47.08% | 45.87% | 48.92% |
| recent12_single_freq0 | 52.73% | 45.86% | 45.56% | 48.35% |
| recent12_single_freq25 | 52.84% | 46.53% | 46.55% | 48.90% |
| recent12_single_freq50 | 52.61% | 46.54% | 46.18% | 48.69% |
| weighted6_single_freq0 | 52.92% | 46.03% | 45.75% | 48.55% |
| weighted6_single_freq25 | 52.90% | 46.37% | 46.61% | 48.88% |
| weighted6_single_freq50 | 52.13% | 46.68% | 45.44% | 48.33% |

The Sep24-only calibration choice is `recent12_single_freq25`. Its three-week development Score is 48.90%, only 0.16 percentage points above the original 48.75%, and it trails the original on two of the three weeks. It is preserved as an experimental challenger, not adopted.

`full_history_single_freq25` reaches 49.51%, the largest observed development Score in this follow-up. It was not the calibration-selected candidate. Selecting it because it has the largest October Score would use already-inspected development labels for selection. Its performance is recorded without presenting it as a validated winner.

## Freeze decision and format rehearsal

Retain the established original temporal model as the reference. Freeze the calibration-selected recent12/25% single-week challenger for a later comparison with fixed weights and thresholds. The final November prediction route remains unchanged. Stop expanding this search grid on the same October weeks.

The frozen package contains model weights, column order, thresholds, data hashes, protocol and experimental-challenger configuration. It does not contain per-user historical outcomes or the source data. The original model SHA256 and candidate SHA256 are in `freeze_decision.json`.

A real 3,960-user format rehearsal at the separate Nov2 origin passed the supplied evaluator and preserved exact official test-ID order. This checks feature construction, ID mapping and CSV format only. It is not a final Nov12 forecast, and no test score was computed. The rehearsal CSV is kept local and excluded from git.

## Validation

- Five new tests passed: chronological label completion with a four-hour gap, exactly twelve retained origins, sample-weight half-life, frequency deduplication and future-availability filtering, and blend endpoints/bounds.
- Five earlier ID, label, thread and coverage tests passed again.
- Every reported aggregate metric matched the official evaluator.
- Syntax and whitespace checks passed.

## Reproduce

Install the repository requirements and preserve the exact official snapshot from the first audit. Commands run from the repository root:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/experiment_v3.py --data ../official
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/isolate_calibration_v3.py --data ../official
python scripts/freeze_baseline.py --data ../official --archive ../Asg2_Frozen_Baseline_20261102.zip
python -m unittest discover -s tests -p test_calibration.py -v
```

The follow-up reuses earlier feature caches when present and rebuilds missing features from the same official snapshot. Source hashes and data manifests record the measured implementation. Use a new output directory after changing data or code.

## Report material

`report_materials/Q2_Q3_Evidence.md` provides English slide copy and Mandarin narration notes. It distinguishes relational aggregation, the earlier GNN comparison, temporal features, calibration and coverage limits. These are content drafts for the instructor template, not a finished presentation PDF or a recorded video.
