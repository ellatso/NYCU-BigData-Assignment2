# November 7 revision — implementation and evidence status

Official source pin: af9f335301d9badb482dfafa10c4534eabac845a. Latest included anonymous roster:4,465 users. The target is Nov22–29 Taiwan, deadline Nov21 23:55. F1 has no direct grade weight; GH Archive source attribution is1% and the report/video is14%.

## Completed

- Recovered the earlier presentation sources and preserved earlier development history.
- Acknowledged https://www.gharchive.org/ in the current README and documented the new requirements.
- Included the unchanged official preprocessing, anonymization, cohort, evaluator, schema and roster, with a commit/file-hash pin.
- Implemented resumable hourly downloads, gzip checks, source hashes, explicit missing-hour records and an optional exact official cohort-filter storage mode. The final official eight-table builder remains unchanged.
- Implemented revised `db/` loading, strict new-data provenance, dynamic/temporal ablations, a primary dynamic relational recipe fixed before real execution, calibration-only thresholds and a Nov22 inference path. The old prediction CLI is retired.
- Preserved unknown test users with an explicit cold-start rule; require exact pinned roster identity, correct binary values and input order.
- Added atomic output writing, source/model/data/cutoff metadata, deterministic single-thread fitting and an independent real-data rebuild/refit comparison command.
- Created the11-cell Colab continuation notebook, a16-page English teacher-template working draft and a personally narrated recording-script draft.

## Actually executed checks

Thirty non-neural regression and synthetic contract tests pass (`verification.json`, `tests.log`). They include the actual unchanged official preprocessing CLI, all eight output schemas, FK/anonymous-ID checks, a source/table checksum rejection, visible-time filtering, stable IDs, cold starts, relationship ablations, deterministic fits and all-eight-table equality of the exact cohort filter on a synthetic example.

The entire11-learned-candidate experiment and output path were executed on labeled synthetic fixtures. The pinned official evaluator accepted the4,465-user fixture CSV. Two independent fixture fits produced identical CSV bytes (`fixture_integration.json`). A separate complete synthetic run also passed the actual independent rebuild command: all eight official tables were rebuilt, all model variants were fitted again, and the refitted CSV matched byte for byte (`fixture_full_reproduction.json`). Its source and outputs remain labeled synthetic. All artificial fixture users are outside the actual roster in this integration check, so this also exercises the all-cold-start boundary. Known-user feature/prediction behavior is covered separately. These are software checks, not real performance evidence or a valid assignment submission. Fixture CSVs are not published or delivered as predictions.

A stale checkpoint was also rejected after source changes during development. No old Nov12 CSV was repurposed for the new anonymous list.

## Concrete blocker and incomplete work

The source endpoint `https://data.gharchive.org/2026-10-01-0.json.gz` returned HTTP403, error1010 to the assistant runtime. Requests were stopped; no alternative client/network route was used to bypass that rejection. No real revised-source data were obtained here.

Consequently, **real revised preprocessing, real-data ablations, the actual prediction CSV and the full real-data rebuild/refit verification remain PENDING**. The old November2 scores are not new results. The PDF is explicitly marked WORKING DRAFT, with pending real-data period/results and no fabricated replacement numbers. It must not be submitted as the completed report in this state.

## Continue with the real source

Open `notebooks/Asg2_Revised_113700058.ipynb` in your own authorized Colab runtime. Its default source interval is Sep1–Nov7 UTC, end exclusive. It may download a substantial amount of data and is not promised to finish within a fixed runtime. Early training origins have censored30-day lookbacks; that limitation is reported. A longer source interval can supply full warmup.

After the notebook finishes, retain the private run bundle. It contains the anonymized prepared tables, manifests, model/configuration, measured ablations, checked CSV and independent reproduction check. Use those outputs to fill the actual data period and result pages in the presentation and recording script. Until then, calling the report final or claiming a new real F1/reproducibility pass would be incorrect.

The video must be personally recorded,16–20 minutes, with the same substantive Q1–Q4 evidence in both the English slides and narration. No E3 upload has been performed in this work.
