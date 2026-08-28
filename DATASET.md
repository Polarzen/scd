# Dataset and compact-data contract

## Source identity

The source is MUSIC (Sudden Cardiac Death in Chronic Heart Failure), release
1.0.1, DOI [10.13026/z3m7-rf58](https://doi.org/10.13026/z3m7-rf58). The
repository preserves the source subject metadata, record metadata, supplied
headers, official checksums, license, and provenance. The source version and
DOI are also fixed in `data/integrity/data_contract.yaml`.

This is an analysis-lossless, source-recoverable handoff, not raw compression.
It preserves the rows and fields needed for the approved analyses and the
information needed to identify and recover the source. Original waveform
payloads are intentionally outside the compact Git dataset.

## Committed compact state

| Path | Role |
| --- | --- |
| `data/cohort/subjects.parquet` | One canonical row per source patient, including follow-up and endpoint source fields |
| `data/cohort/records.parquet` | One row per known Holter or high-resolution record |
| `data/cohort/provenance.parquet` | Record-to-source file identity, sizes, and hashes |
| `data/source_exact/subject-info*.csv` | Source subject metadata and code definitions |
| `data/source_exact/RECORDS` | Source record inventory |
| `data/source_exact/headers/` | Source WFDB headers; these are metadata, not waveform payloads |
| `data/features/legacy_120s/` | Legacy 120-second windows, features, and patient aggregation |
| `data/features/full_5min/` | Full 5-minute windows, feature shards, manifest, and patient aggregation |
| `data/integrity/compact_sha256.txt` | SHA256 manifest for the committed compact state |
| `data/integrity/data_contract.yaml` | Dataset identity, endpoint states, missingness, and compression contract |

The official cohort contains 992 subjects, 936 Holter records, and 687
high-resolution records. The compact feature tables use Zstandard-compressed
Parquet. Missing values remain null; they are not changed to zero, `-1`, or a
sentinel string.

The current legacy patient table has 88 eligible patient rows and 2,112
legacy windows. The full feature manifest records the generated 5-minute
window shards and their QC state. Counts and status are verified from the
tables rather than inferred from filenames; see the Phase 3/full-feature
reports for the frozen build details.

## Endpoints and units

Endpoint construction supports 90, 180, 365, and 730 days. The source states
are `POSITIVE`, `NEGATIVE`, `CENSORED`, `COMPETING_EVENT`, and `UNKNOWN`.
Only evaluable positive and negative rows enter binary model validation;
censored, competing-event, and unknown rows remain explicit and are excluded
by the endpoint contract.

The independent analysis unit is the patient. Windows are repeated
measurements within a patient, not independent observations. Window-level
features are aggregated before modelling, and every outer/inner CV split is at
patient level. A patient therefore occurs in only one outer test partition for
each validation run.

## Feature families

The legacy family retains the historical 120-second window convention and its
100 aggregate feature columns. Its comparison with the old random-window
representation is documented as a partial reproduction; the legacy and full
families are not silently treated as identical.

The full family uses the frozen `full_5min_v2` contract: 20 base features,
mean/std/p10/p50/p90 aggregation, and per-feature validity counts. The
`all20` model profile uses all 20 features. The `physiology_only` profile
excludes the `SIGNAL_QUALITY` category while retaining the same patient and
endpoint rules. Endpoint columns are outcomes, never model features.

## Waveform boundary

Raw source is needed only to rerun preprocessing or window extraction, rerun
R-peak detection, or add a new waveform-derived feature. It is not needed for
hash verification, cohort validation, endpoint construction, model
validation, repeated CV, or endpoint sensitivity from the committed compact
tables. The P4-D workflows have no source-data location, acquisition step, or
cohort regeneration step.
