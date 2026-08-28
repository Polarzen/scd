# GitHub handoff

MUSIC-CI is ready for model development from committed compact data. It is an
analysis-lossless derived package with source-recoverable provenance, not a
compressed copy of the raw ECG waveforms. The source is MUSIC v1.0.1, DOI
10.13026/z3m7-rf58.

## Repository source of truth

- `data/cohort/subjects.parquet`: all 992 official subjects and their states.
- `data/cohort/records.parquet` and `provenance.parquet`: record identity and provenance.
- `data/features/full_5min/`: deterministic window manifest, 20 window features, QC, and 100D patient aggregation.
- `data/analysis/`: survival-ready and complete patient analysis-status tables.
- `config/` and `src/`: endpoint, profile, aggregation, model, and validation definitions.
- `data/integrity/compact_sha256.txt`: portable integrity snapshot.

The independent statistical unit is the patient. Windows are repeated
within-patient measurements and are aggregated before modeling. The 365-day
endpoint is 1-year SCD risk stratification, not an hours-before-SCD warning.

## Commands

```bash
python -m src.cli verify
python -m src.cli validate --endpoint 365 --profile all20 --model extratrees --seed 42
python -m src.cli repeated-cv --endpoint 365 --profile all20 --model extratrees --seeds 0:99
python -m src.cli endpoint-sensitivity
```

No command starts an expensive task when invoked without its subcommand.
Formal outputs are written in JSON and CSV/Parquet formats.

## Workflows

- `.github/workflows/ci.yml`: compact integrity, tests, and deterministic smoke validation on push/PR/manual trigger.
- `.github/workflows/model-validation.yml`: manual nested CV for one endpoint/profile/model/seed.
- `.github/workflows/repeated-cv.yml`: manual 10-batch x 10-seed repeated nested CV, max five parallel jobs.
- `.github/workflows/endpoint-sensitivity.yml`: manual 90/180/365/730-day comparison.

These workflows read committed MUSIC-CI data. They do not use
`MUSIC_RAW_DIR`, download MUSIC, or rebuild ECG features.

## Tasks that still require the raw 90GB source

Changing window length, raw filtering, the R-peak detector, signal-level QC,
the 20 base feature extraction, or adding waveform morphology requires a local
full rebuild with `scripts/build_from_music.py`. Model changes, aggregation
experiments, endpoint selection, patient splits, nested CV, repeated CV, and
metric/report generation do not.

## Known limitations

- The compact package does not contain raw waveforms, R-peak series, or RR caches.
- Baseline Holter timing is not the time immediately preceding a recorded SCD.
- The primary binary analysis excludes non-evaluable endpoint states and the predefined sinus-HRV-ineligible group from modeling while preserving every subject in the cohort facts.
