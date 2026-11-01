# Q2 and Q3 presentation content

English slide copy with Mandarin narration notes. These are draft content blocks to place in the instructor's template. They are not the final presentation PDF. Use no more than ten slides overall and personally record the 8–12 minute video.

## Q2 slide: Relational modeling

### English slide copy

**Relational modeling**

- Join activity actors to repositories through the supplied foreign keys.
- Aggregate activity from other participants in the same repositories, excluding the target user's own event counts.
- Join comments and reviews to issue or PR authors to measure received interactions and distinct partners.
- Compare feature aggregation with an explicit user/repository message-passing model.

| Comparison on the original two diagnostic weeks | Combined Score |
|---|---:|
| Temporal tree model | 50.17% |
| Temporal plus relational tree features | 49.79% |
| User MLP, earlier neural experiment | 48.23% |
| Relational GNN, earlier neural experiment | 48.52% |

Relational features did not consistently improve the strongest temporal baseline. The neural pair is a matched message-passing ablation. The tree and neural models have different features.

### Mandarin narration

我用外鍵把活動、使用者、repo，以及 PR 或 issue 的作者串起來。第一種方式把鄰居的活動聚合成特徵，而且扣掉自己的事件，避免只是把自己的活躍度換個欄位名稱。第二種方式用可訓練的訊息傳遞模型，比較加入關係前後的神經網路。神經網路那組有小幅改善，但沒有超過時間特徵的樹模型，因此我沒有宣稱關係建模一定比較好。

### Evidence

- Tree comparison: `artifacts_20261102/metrics.csv`, holdout rows.
- Neural comparison: `artifacts_step2/metrics.csv`, development_diagnostic rows. This is the earlier measured experiment, not a new GNN rerun in November.
- Implementation: `asg2/core.py` and `asg2/graph.py`.

## Q3 slide: Temporal information and availability

### English slide copy

**Temporal information and availability**

- Predict six activities over a seven-day outcome window.
- Filter features by event occurrence and row availability before the cutoff.
- Simulate a four-hour observation gap before each prediction origin.
- Use 7/14/30/90-day activity counts, recency, active days, repository diversity and recent activity trends.
- Exclude fields known only later from the model inputs.
- Keep training outcome windows separate from calibration and evaluation cutoffs.

Original temporal features improve the two-week diagnostic Score from the last-week rule's 46.32% to 50.17%.

### Mandarin narration

時間資訊有兩個層次。第一個是行為規律，例如最近多久活動、不同時間長度的次數，以及近期趨勢。第二個是資訊當時能不能看到。即使某個 issue 很早建立，如果它到後來才出現在資料中，模型在先前時間也不能使用它。我也讓每次回測保留四小時空窗，模擬正式資料更新到晚上八點、預測從午夜開始的條件。

### Evidence

- `artifacts_20261102/protocol.json`, `checks.json` and `metrics.csv`.
- Known-later fields remain excluded by the column allowlist in `asg2/core.py`.

## Q3 experiment slide: Calibration and training history

### English slide copy

**Calibration and training history**

Four chronological calibration origins: Aug13, Aug27, Sep10 and Sep24.

- Full historical training versus the most recent 12 weeks.
- Six-week half-life sample weighting.
- Fixed 0%/25%/50% blends with historical activity frequency.

| Three-week development comparison | Combined Score |
|---|---:|
| Established original model | 48.75% |
| Choice from pooled multiweek calibration | 38.81% |
| Choice from a later single-week diagnostic | 48.90% |

The later single-week comparison was added after inspecting the initial results. Its choice improves the pooled Score by only 0.16 percentage points and trails the original on two of three weeks.

### Mandarin narration

我把訓練歷史長度、近期權重，以及校準方式分開比較。最初用四個歷史週的時間外預測來選門檻，但選出的模型在開發週表現較差。接著我追加最近單週校準的對照，確認問題不完全來自訓練方式。這個追加診斷的時間點需要說清楚，不能把已經看過的十月資料說成全新的測試。追加選出的候選只小幅增加合併分數，而且三週裡有兩週較差，所以我保留原始模型作為基準。

### Evidence

- `artifacts_v3_20261102/design.json`, `fold_protocol.json`, `metrics.csv`.
- `diagnostic_amendment.json`, `singleweek_selection.json`, `singleweek_diagnostic_metrics.csv`.
- Both scores use the same three development origins. The earlier 50.17% uses only Oct8/15, so do not compare 50.17% and 48.75% as if they used the same sample.

## Analysis slide: Coverage limits and model choice

### English slide copy

**Coverage limits and model choice**

- The Nov1 snapshot has no recorded pushes after Oct26, while other targets continue into Nov1.
- Zero recorded events may reflect inactivity or collection coverage. The data alone do not establish the cause.
- Additional conversation features are positive in only 8–36 of 148,917 training rows. The fitted trees use none of these features.
- Preserve the original model and one experimental challenger with fixed weights and thresholds.
- Keep Oct29–Nov5 unscored until the full outcome horizon and target coverage can be checked.

Current results are historical development diagnostics. The final November prediction file and official score are not available.

### Mandarin narration

資料檢查發現不同活動的最後時間不一致，所以我不會把零紀錄直接當成使用者完全沒有活動。我另外做了排除可疑 push 訓練週的敏感度實驗，但改善很小。新增討論串特徵也因為支援樣本太少，沒有被樹模型使用。最後我保存模型權重與門檻，後續用固定設定驗證，避免看完新結果又一直更改模型。

### Evidence

- `artifacts_20261102/audit.json`, `daily_event_counts.csv` and `event_coverage.csv`.
- `artifacts_v2_20261102/thread_support.json` and `metrics.csv`.
- `artifacts_v3_20261102/freeze_decision.json`.

## Consistency checks before recording

- Use the original temporal model as the established reference. Identify the recent12/25% single-week version as experimental.
- Do not label the tree aggregation model as an end-to-end GNN. The explicit neural model is a separate comparison.
- Do not describe a 0.16 percentage-point change as a statistically significant gain.
- Retain the data-coverage caveat near recent results.
- Update all final-model statements together if a later, documented decision changes the selected model.
- Include the public GitHub URL in the final deck and use the teacher's slide template.
