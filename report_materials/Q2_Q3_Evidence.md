# Updated Q2 and Q3 presentation evidence

This replaces the earlier mixed-date content. The main PDF uses the same evaluation origins, Sep24 and Oct1, for all method comparisons. All scores are historical, with collection completeness unconfirmed.

## Q2: recent interactions across linked tables

### English slide copy

The retained model joins activity actors to users and repositories, and includes repository diversity. Additional neighborhood features are evaluated separately.

Cross-table interaction flow: comment/review actor, linked PR or issue, author. Count received events and distinct partners over 3/7 days, with self interactions removed.

Additional signals:

- New PRs by others in the last three days on repos the user reviewed.
- Seven-day observed activity in the user's repos, excluding the user.
- Seven-day activity of direct cohort collaborators.

The 32 additional features produce a 50.12% evaluation Score versus 50.40% for temporal inputs and basic user/repo joins. Calibration also favors the temporal candidate. Keep the extra features as an ablation rather than claiming a gain. The older relation features already included 30-day aggregates, so their earlier failure cannot be attributed solely to being static.

### Mandarin narration

我先用外鍵把不同活動表連到使用者與 repo，基本模型也有 repo 多樣性，所以不能說它完全沒有使用跨表關係。我另外加入 32 個更直接反映近期互動的特徵，例如自己開的 PR 最近收到多少 review，以及曾經 review 的 repo 最近又出現多少新 PR。所有計算都只用當下可見的資料，並扣除自己的事件。這些特徵確實有非零樣本，但在相同評估週沒有改善，所以我保留這組消融，不為了讓報告看起來有改進而換成較差的方法。

## Q3: time and calibration

### English slide copy

- Original temporal family: lifetime counts and age, multi-window counts, active days, repo diversity, per-target recency and trends. Total: 51 inputs.
- Event occurrence and row availability must both precede the feature cutoff.
- Every origin has a four-hour observation gap and a seven-day outcome horizon.
- Train on May7–Aug6 origins. Calibrate on Aug20 and Sep10. Evaluate on Sep24 and Oct1.
- Choose thresholds by mean per-week calibration Score. Compare top-k using mean calibration positive rates.

### Mandarin narration

這一版重新統一了比較範圍。訓練、校準與評估嚴格依時間分開，而且前一階段的完整標籤必須在下一階段的觀測截止時間前結束。校準不再只看一週，而是最大化兩個校準週 Score 的平均。top-k 也有比較，但它雖然在評估資料略高，校準表現較差，所以不能看完評估分數才改選它。

## Main comparison

| Method | Same-period six-target Score |
|---|---:|
| Last-week rule | 51.37% |
| Temporal plus mean-week thresholds | 50.40% |
| Temporal plus top-k positive rate | 50.61% |
| Earlier relation features plus thresholds | 50.31% |
| Dynamic relations plus thresholds | 50.12% |

The last-week rule leads on evaluation, but loses on calibration: 43.93% versus 45.65%. This replaces the old broad claim that the learned temporal model beats the simple rule. The evaluation dates have been used in development and are not an untouched final test.

## Push coverage and fallback

The official Nov1 release still has no push records after Oct26. Oct15 and Oct22 outcome weeks also contain zero push days, and Oct8 contains a zero line-comment day. These weeks are removed from the primary comparison. A zero count does not prove a missing event.

Simulated trailing push-feature loss retains the same outcome labels. Mean push F1 at 7/14/21-day loss changes from 55.45/39.99/24.86% to 59.47/61.00/62.98% with frequency measured at the last observed push time. These are push F1 values, not the six-target Score.

The frequency method is worse with healthy data or a two-day gap, so the automatic fallback waits for at least seven days of staleness. It does not repair historical internal gaps. TA confirmation about backfill and answer coverage is still needed; a draft question is provided in the report directory and has not been posted.

## Final refit

The final command refreshes the official data and refits on completed windows. With the final Nov11 release, an additional origin at Nov4 Taiwan midnight can be included without using future labels. Flagged push outcome weeks are omitted only from push-head training. The program checks data freshness and exact test-ID/binary format. The current snapshot is refused for a final Nov12 CSV.

## Evidence

See `artifacts_coverage_20261102/REPORT.md`, `metrics.csv`, `weekly_zero_days.csv`, `push_dropout_stress.csv`, `production_checks.json` and `TA_question_zh.md`. The revised PDF uses plain-language captions and one connected 30-node entity graph with each user shown once.
