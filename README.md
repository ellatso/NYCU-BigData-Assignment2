# NYCU Big Data Assignment 2 — first measured baseline

Author: Chen Jhen Tso (113700058). Individual assignment, Fall 2026.

[Open the notebook in Colab](https://colab.research.google.com/github/ellatso/NYCU-BigData-Assignment2/blob/main/notebooks/Asg2_Start.ipynb) | [Measured results](artifacts/metrics.csv) | [Schema graph](artifacts/graphs/schema.svg) | [Entity graph](artifacts/graphs/entity_graph.svg)

This is a functioning CPU baseline and experiment scaffold, **not a completed November submission or a claim of full marks**. It uses only the instructor's official relational data. No GitHub user API or external activity data are collected.

## Quick start

The accompanying `Asg2_Start.ipynb` is self-contained: import into Kaggle/Colab, keep Internet on, and Run All. GPU is not needed. It writes this project outside the downloaded instructor-data folder and runs the benchmark, checks and graph extraction. First-run package installation/download and machine speed determine runtime; no specific runtime is guaranteed.

For local execution (from this folder):

```powershell
python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git ../assignment2_data
python -m asg2.run benchmark --data ../assignment2_data --out artifacts
python scripts/checks.py --data ../assignment2_data --out artifacts
python scripts/graphs.py --data ../assignment2_data --out artifacts/graphs
```

Graph extraction always writes DOT. SVG rendering requires Graphviz's `dot` executable (the Python package alone does not supply it). Measured SVG/PNG diagrams are already included under `artifacts/graphs/`. Local measurement used a Graphviz WebAssembly renderer because system `dot` was unavailable; the DOT input is the same.

Per-user historical prediction CSVs are kept locally and are not published here. Public evidence consists of aggregate metrics, split protocols, data hashes, checks and the assignment-required graph visualizations. Running the code generates your own local prediction files.

## What has actually run

The baseline was run on the instructor snapshot whose latest observed row timestamp is **2026-10-22 11:59:23 UTC**. `artifacts/data_manifest.json` records SHA256 for all eight files. Every parquet is read with a column allowlist. The six known-only-later fields are excluded from modeling entirely.

Training origins are Thursdays in Taiwan from May 7 to September 10, 2026 (19 weeks, 148,917 user-week samples). Calibration is September 24 (4,931 samples). Two separate historical holdout origins are October 8 and October 15 (8,731 samples total). All labels are seven-day windows. Calibration thresholds and model selection use ONLY calibration outcomes, never holdout results. A four-hour observation gap is simulated at every origin to match the final release ending Nov11 20:00 Taiwan before the Nov12 00:00 forecast start. Prior outcome windows end before the next split's observation cutoff.

Historical population is a proxy: cohort users with at least five observed target events in the previous 90 days. This does not claim to reconstruct the exact official Oct28 test cohort selection. Final prediction uses the official test IDs directly with no activity-based filtering.

| Model | Macro F1 | Micro F1 | Official combined score |
|---|---:|---:|---:|
| Last-week rule | 40.39% | 52.26% | 46.32% |
| Lifetime user features | 36.84% | 48.61% | 42.73% |
| User temporal features | 44.55% | 55.79% | 50.17% |
| Static relational features | 37.33% | 49.53% | 43.43% |
| Temporal plus relational features | 44.14% | 55.44% | 49.79% |

These are **historical validation results, not the official November score**. Temporal-only was selected by calibration. Current relational features did not improve the held-out combined score over temporal-only; this result is retained rather than hidden. Further experiments need a new untouched forward holdout as more data arrive; do not repeatedly tune to October 8/15.

## Models and features

Six LightGBM binary classifiers, one per target. Fixed seed 42, 160 trees, learning rate .045, 23 leaves, minimum leaf support 70, L2 4. The variants are:

- `static`: user age and six lifetime activity counts.
- `temporal`: adds 7/14/30/90-day counts, active days, distinct repos, last-event recency and recent/longer-term rate ratios.
- `relational`: static user features plus user-repo-user neighborhood aggregation and cross-table PR/issue-author interaction counts.
- `full`: all above plus 30-day relational aggregates.

Relations use FKs to connect activity to users/repositories and join review/discussion events to PR/issue authors. Repository activity excludes each user's own contribution when constructing neighboring activity. Non-cohort activity is incomplete by construction; neighborhood statistics describe the supplied observed graph, not complete GitHub traffic. This is relational feature aggregation followed by tree models, **not a GNN or end-to-end RDL architecture**. Whether a final submission based only on aggregated features meets the instructor's phrase “directly analyze relational databases” should be confirmed; adding an explicit relational graph model is a planned comparison, not yet implemented.

Thresholds are calibrated using the exact `(macro_F1 + micro_F1)/2` objective, including the official convention that an empty target with no false predictions has F1=1. Each threshold search is confined to calibration data. A simple last-week rule is included to guard against mistaking model complexity for improvement.

## Time and identity correctness

- Filter PR/issue/review rows by `first_seen_at`, and event occurrences by their own timestamps, before each prediction cutoff. A historical event learned only after cutoff cannot enter features.
- Users/repositories must also exist before cutoff. Training populations use cohort users only.
- Training labels must have occurrence within the outcome week AND availability before that week ends; retrospective records learned later are not used to train a historical model. Historical labels remain reconstructions from a released snapshot; there is no claim of access to the TAs' private truth.
- Output uses `Users.github_user_id`; table row-index PK/FK values are used only for joins within one snapshot.
- No random train/validation split. Date windows, row counts, data hashes, per-target scores, thresholds and checks are saved.
- Cache directories are keyed by data hashes. After changing feature code, remove `artifacts/cache` before rerunning, or use a new output folder.

`checks.py` has been executed: injecting future-availability records or future-occurrence records did not change past features; output IDs are stable GitHub IDs; scoring matches the official evaluator including empty-class behavior; split windows are correctly separated.

## Q1 diagrams

`artifacts/graphs/schema.svg`: all eight tables, PK/FK ports and 20 FK edges, including the self-reference in PRReviewComments. Org IDs and head-repository GitHub IDs are not row-index FKs according to the instructor schema.

`entity_graph.svg`: seed GitHub IDs 22996444 and 28983449, activity-row times in `[2026-09-15 00:00, 2026-09-17 00:00)` UTC. Blue rows are activity rows in the window; yellow nodes are seed users. Gray nodes are necessary FK endpoints that may predate the window, recursively included as context. No future rows or unrelated repo events are added. Current snapshot gives 30 nodes, 57 edges, 17 in-window activity rows. Internal row IDs may differ in subsequent releases. This is an explicit extraction convention; verify the TA's expected neighborhood definition before final slides.

## Final submission is not generated yet

Official `test_input.csv` is scheduled for Oct28. The final data release is Nov11 at 21:00 Taiwan (through 20:00); deadline is Nov11 23:55. Never submit the historical holdout CSVs as final predictions.

After final model/threshold choice is frozen and the last data release is available:

```powershell
python -m asg2.run predict --data ../assignment2_data --out artifacts --input ../assignment2_data/evaluation/test_input.csv --student-id 113700058
```

This refits the calibration-selected variant using complete historical windows from the final snapshot, keeps calibrated thresholds, writes `artifacts/output_113700058.csv`, and invokes the official evaluator with the exact test-user list. The final refit path is implemented but cannot be executed/validated against future November data today. The program refuses an October snapshot for official final prediction. Final model rankings/thresholds may change after justified development experiments; document any change and evaluate on a fresh forward holdout.

You still need a public GitHub repository, English slides using the teacher's template (<=10 pages, Q1-Q4), and a personally narrated 8-12 minute MP4. No AI-generated presentation video. Model/slide claims must match actual implementation and saved outputs.

## Source attribution

Data, schema and evaluator: https://github.com/nycubigdata-2026fall/BigData2026-Assignment2 . Model and experiment code in this package was prepared with AI assistance. No results are fabricated. Dependencies in `requirements.txt` are direct requirements; actual local versions are saved in `artifacts/environment.json`.
