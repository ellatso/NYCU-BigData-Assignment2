# Slide facts — Assignment 2 (revised run of 2026-11-08)

Every number below comes from a file in `artifacts_revised_20261108/` (run outputs) or
`artifacts_revised_20261108/analysis/` (written by `scripts/analyze_revised.py`). Figures are in
`report_materials/figures_revised/` (PNG for slides, PDF/SVG vector). Only claim what is listed here:
the course deducts points for hallucinated results.

Repo: https://github.com/ellatso/NYCU-BigData-Assignment2 (branch `main`)

## Suggested 16-page outline

| # | Slide | Figure | Question |
|---|---|---|---|
| 1 | Title, name, student ID, public GitHub link | — | — |
| 2 | Task, data source, cohort and test users | — | — |
| 3 | Schema graph | `q1_schema_graph.png` | Q1 |
| 4 | Dataset period and time-based split | `q1_q3_timeline.png` | Q1, Q3 |
| 5 | Framework figure | `q4_framework.png` | Q4 |
| 6 | As-of sampling and leakage control | (text / reuse timeline) | Q3 |
| 7 | Temporal features | (table) | Q3 |
| 8 | Data are non-stationary | `q3_weekly_volume.png` | Q3 |
| 9 | Label drift and threshold calibration | `q3_label_prevalence.png` | Q3 |
| 10 | Visibility delay and the availability clock | (table from `visibility_delay.csv`) | Q3 |
| 11 | Relational feature design (FK joins) | (diagram/table) | Q2 |
| 12 | Why relations add little here | `q2_relation_evidence.png` | Q2 |
| 13 | Ablation with bootstrap CIs | `q2_q3_ablation.png` | Q2, Q3 |
| 14 | Per-activity F1 + TA reference-week check | `q3_per_target_f1.png`, `q4_reference_week.png` | Q3, Q4 |
| 15 | Final prediction and reproducibility | (bullets) | Q4 |
| 16 | Conclusions, limitations, next steps | — | all |

## Facts per slide

### 2 · Task and data
- Predict 6 binary activities per user for 2026-11-22 00:00 – 11-29 00:00 Taiwan time: open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push.
- Data source: GH Archive (https://www.gharchive.org/), processed by the unchanged official `preprocess.py` (9,000-user cohort, anonymization).
- Test users: 4,465 (`submission/test_input.csv`); 4,310 have visible history, 155 do not.
- Tables built: Users 10,140 rows (5,255 cohort users seen), Repositories 20,197, PullRequests 41,461, Issues 26,956, IssueComments 15,245, PRReviews 12,524, PRReviewComments 9,999, Pushes 88,781.

### 3 · Q1 Schema graph
- 8 tables, 20 foreign keys; every edge = FK → referenced PK. Users and Repositories are the hubs (8 FKs → Users, 6 → Repositories, 6 between activity tables, including the self-reference `PRReviewComments.in_reply_to_id`).
- Each table has a row time (when the row becomes visible): `first_seen_at` for Users, Repositories, PullRequests, Issues, PRReviews; `created_at` for IssueComments, PRReviewComments, Pushes.

### 4 · Q1 Dataset period
- **Data period used: 2026-09-01 00:00 – 2026-11-07 00:00 UTC** (= 09-01 08:00 – 11-07 08:00 Taiwan), 1,608 hourly files, all downloaded and gzip-verified, 28.7 GB raw.
- Split (outcome weeks start Thursday 00:00 Taiwan): train 09/10, 09/17 · calibration 10/01, 10/08 · development 10/22, 10/29. Each split's labels end before the next split's feature cutoff.
- Final model refit on 9 origins (8 Thursdays 09/10 – 10/29 + latest complete window 10/31); feature cutoff 11/07 08:00 Taiwan, i.e. 352 h before the target week.

### 5 · Q4 Framework (left to right)
Data (GH Archive → official preprocess → 8 tables) → as-of sampling (event stream with occurrence and visibility time; origin t; rows visible before t − 4 h; label in [t, t+7 d) visible by t+7 d) → features (temporal, relational, user profile) → 6 LightGBM classifiers (seed 42, 160 trees, single-thread deterministic) → per-activity thresholds on calibration weeks → 11 ablations + copy-last-week baseline → refit → `output_113700058.csv` → official `evaluate.py` + independent rebuild/refit check.

### 6 · Q3 Leakage control
- Origin t; features use only rows whose row time and occurrence time are before t − 4 h (simulates the gap between data availability and the forecast).
- Later-known columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) are not used at all.
- Historical labels count an activity only if it occurred in [t, t+7 d) **and** was visible by t+7 d.
- Splits by time only, never random.

### 7 · Q3 Temporal features (submitted model: 75 inputs = 7 user + 36 temporal + 32 relational)
- Per activity: counts in last 7/14/30 days, days since last occurrence (recency), 7-day vs 30-day trend.
- Active days and distinct repositories in 7/14/30 days.
- User profile: account age in data, lifetime counts per activity.
- Most important single features (gain): recency of pr_review, account age, recency of pr_review_comment, recency of open_issue, recency of issue_comment.
- Gain share: recency 22–64% depending on activity (62% pr_review, 64% pr_review_comment); window counts dominate push (56%) and open_pr (45%).

### 8 · Q3 Non-stationarity (`weekly_volume.csv`)
- Weekly rows, week of 09/10 → 10/22 → 10/29: open_pr 4,984 → 458 → 692; pr_review 3,986 → 372 → 534; push 6,271 → 5,999 → **66** (peak 19,280 in week of 09/24).
- PR/issue/comment/review rows drop about 5× after the week of 09/17; pushes almost vanish after 10/29.
- All hours were downloaded, so this is in GH Archive content (the course also notes missing information in the raw data). The README says activities missing during the test week are not scored.

### 9 · Q3 Label drift (`weekly_prevalence.csv`)
- Share of cohort users doing the activity, week of 09/10 vs 10/22: open_pr 38.1% → 5.6%; issue_comment 32.0% → 3.9%; push 47.6% → 28.8%, then 1.1% in the week of 10/29.
- Consequence: calibration Score ≈ 52% but development Score ≈ 35%; thresholds tuned on one regime are applied in another.
- Effect of push collapse (`five_target_by_week.csv`): six-activity Score 41–45% in week 10/22 vs 21–24% in week 10/29 for every variant; five-activity Score without push 31–36% in both weeks.
- Submitted thresholds (fixed from calibration): open_pr 0.30, open_issue 0.275, issue_comment 0.375, pr_review 0.175, pr_review_comment 0.225, push 0.425.

### 10 · Q3 Visibility delay (`visibility_delay.csv`)
- Share of rows visible more than 24 h after the event time: Issues 30.4%, PullRequests 2.8%, PRReviews 0.4%; comments, review comments and pushes 0%.
- Availability-clock variant (windows on visibility time instead of occurrence time): development 36.41%, +1.59 pt over copy-last-week, 95% CI of the difference +0.62 to +2.62 pt.

### 11 · Q2 Relational design (32 dynamic + 18 static features)
- Repository activity: other users' 7-day activity (mean/max per activity) in repositories the user touched in 90 days (own events excluded).
- Reviewed-repo new PRs: PRs opened by others in the last 3 days on repositories the user reviewed in 90 days.
- Received interactions: comments, reviews and review comments received on the user's PRs/issues in 3/7 days, and distinct partners (self excluded).
- Collaborators: 7-day activity of direct co-actors; number of cohort collaborators in 90 days.
- Implemented as FK joins + aggregation, followed by trees (not a GNN).

### 12 · Q2 Why relations add little (`relation_coverage.csv`, `cohort_closure.json`, `feature_group_share.csv`)
- Users with any non-zero value (week 10/22): static 99.0%, repository activity 6.2%, collaborators 1.4%, received interactions **0.08%**.
- Cohort closure: of 10,524 comments on cohort-authored threads, only 40 are by someone else; of 2,542 reviews on cohort-authored PRs, 41 are by someone else. The tables keep only cohort users' own events, so replies from non-cohort users are not observed.
- Relational families receive 3–5% of the submitted model's gain.

### 13 · Ablation (`metrics.csv`, `bootstrap_development.csv`; dev = weeks 10/22 + 10/29, 10,334 user-weeks, 1,000 paired user resamples)

| Variant | Dev Score | 95% CI | Difference vs copy-last-week (95% CI) |
|---|---:|---|---|
| Copy last week | 34.82% | 33.29–36.45 | — |
| User profile only | 32.90% | 31.61–34.23 | −1.89 pt (−3.20, −0.58) |
| + last 7 days | 34.22% | 32.80–35.76 | −0.59 pt (−1.65, +0.53) |
| Temporal 7/14/30 d | 35.73% | 34.19–37.37 | +0.91 pt (−0.08, +1.93) |
| Temporal − recency | 36.46% | 34.91–38.10 | +1.64 pt (+0.70, +2.63) |
| Temporal − trend | 35.30% | 33.79–36.91 | +0.48 pt (−0.47, +1.53) |
| Temporal, availability clock | 36.41% | 34.83–38.06 | +1.59 pt (+0.62, +2.62) |
| Static relations only | 33.45% | 32.09–34.73 | −1.37 pt (−2.67, −0.12) |
| Temporal + static relations | 35.02% | 33.50–36.59 | +0.20 pt (−0.70, +1.17) |
| Temporal + received relations | 35.73% | identical to Temporal | all-zero features |
| **Temporal + dynamic relations (submitted)** | 35.12% | 33.63–36.70 | +0.30 pt (−0.65, +1.32) |
| Dynamic − collaborators | 35.30% | 33.84–36.91 | +0.48 pt (−0.47, +1.52) |

- Temporal + dynamic relations vs Temporal alone: −0.60 pt, 95% CI (−1.17, −0.06).
- The submitted variant was fixed in the config before any real-data run, to avoid choosing on development results; calibration-best reference was static relations (52.76%), which is worse on development.

### 14 · Per-activity F1 and reference-week check
- Submitted model per-activity F1 (dev): open_pr 32.6%, open_issue 38.6%, issue_comment 35.5%, pr_review 32.9%, pr_review_comment 24.6%, push 38.2%.
- TA reference week 2026-09-27 – 10-04 (official README): copy last week 49.61%, GNN with time encoding 54.82%. Ours on the same week (4,311 visible cohort users): copy last week 50.39%, temporal 53.79%, submitted 53.95%. Caveat: overlaps our calibration window, so our model rows are optimistic; the rule row shows our labels are consistent with the TA's.

### 15 · Final prediction and reproducibility (`prediction_protocol.json`, `reproduction_checks.json`, `cross_machine_check.json`)
- 4,465 rows in exact input order; passes the official `evaluate.py` format check; 155 unseen users get the calibration prevalence then the fixed threshold.
- Independent check: rebuilt all 8 tables from the raw files, refit all models, CSV byte-identical (SHA-256 f5be6efe…).
- A second execution on a different machine produced a byte-identical model checkpoint and identical metrics.
- Positive-prediction rates in the CSV: open_pr 4.7%, open_issue 3.6%, issue_comment 2.1%, pr_review 9.5%, pr_review_comment 1.4%, push 4.2%.

### 16 · Conclusions and limitations
- Time is the main signal: recent windows, recency and visibility-aware counting.
- Relational aggregation adds little because the cohort-only data rarely contain other users' interactions with a cohort user.
- The data distribution shifts strongly over the period; with only two development weeks, most differences are within ±1.5 pt and CIs overlap.
- Not done: an end-to-end GNN/RDL model; the TA's GNN with time encoding is a natural next step.
