# MUSIC sudden-cardiac-death analysis

This repository contains a reproducible, patient-level analysis of the MUSIC
(Sudden Cardiac Death in Chronic Heart Failure) cohort, version 1.0.1. The
repository handoff is analysis-lossless and source-recoverable: committed
cohort metadata, provenance, hashes, compact windows, and derived feature
tables are sufficient to rerun the analyses, while the original waveform
payload is not a raw-compression archive. The source is recoverable from the
[MUSIC 1.0.1 record](https://doi.org/10.13026/z3m7-rf58) and the committed
SHA256 manifests.

## Quick start

Create an environment with the tested lock and run the integrity checks:

```bash
python -m pip install --requirement requirements-lock.txt
python scripts/verify_all.py
python -m src.cli verify
```

All model commands consume committed compact tables. They do not download
waveforms or rebuild a cohort. A single deterministic validation run is:

```bash
python -m src.cli validate --endpoint 365 --profile all20 --model extratrees --seed 42 --n-jobs 1 --output-dir data/validation/full_model
```

The small CI smoke command is:

```bash
python -m src.cli validate --endpoint 365 --profile all20 --model dummy --seed 42 --smoke --output-dir .tmp/ci-smoke
```

Repeated patient-level validation over ten consecutive seeds can be run with:

```bash
python -m src.cli repeated-cv --endpoint 365 --profile all20 --models extratrees,logistic,dummy --seed-start 1000 --seed-stop 1099 --n-jobs 1 --output-dir data/validation/repeated
```

Endpoint sensitivity uses the four prespecified horizons:

```bash
python -m src.cli endpoint-sensitivity --horizons 90,180,365,730 --profile all20 --model extratrees --seed 42 --n-jobs 1 --output-dir data/validation/endpoint-sensitivity
```

## What is evaluated

The primary endpoint is a binary outcome at 365 days. The supported horizons
are 90, 180, 365, and 730 days. Censored, competing-event, and unknown states
remain explicit in the endpoint table and are not silently converted to zero.
The 365-day model is a risk-stratification analysis over that follow-up
horizon; it is not an hourly warning system and does not estimate the time of
an imminent event.

The independent unit is the patient. Multiple windows from one patient are
repeated measurements and are aggregated to one patient row before modelling.
Outer and inner cross-validation splits are therefore patient-level, so
windows from a patient cannot leak between training and test partitions.

Two frozen feature families are retained for comparison:

* `data/features/legacy_120s/` contains the legacy 120-second fixed-window
  representation and its 100 aggregate columns. Its reproduction status and
  known feature differences are recorded in the Phase 3 reports.
* `data/features/full_5min/` contains the current full 5-minute representation
  with the v2 feature contract. The `all20` profile aggregates the 20 frozen
  base features with mean, standard deviation, p10, p50, and p90 statistics;
  validity counts are retained as quality metadata. `physiology_only` removes
  the `SIGNAL_QUALITY` category without changing endpoint construction.

The model workflow supports ExtraTrees, logistic regression, and a prior-rate
dummy comparator. Thresholds use the fixed target-specificity rule from the
validation configuration. OOF predictions, fold metrics, bootstrap intervals,
population counts, and configuration snapshots are emitted as machine-readable
artifacts by the manual GitHub workflow.

## When the source waveform is needed

Raw source is required only when rebuilding preprocessing or windows, rerunning
R-peak detection, or adding a new waveform-derived feature. It is not needed
for the committed cohort checks, endpoint construction, model validation,
repeated CV, or endpoint sensitivity workflows. The compact data contract
records which files and hashes define the source-recoverable state; see
[`DATASET.md`](DATASET.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## GitHub workflows

`ci.yml` runs on pushes, pull requests, and manual dispatch. It installs the
locked dependencies, verifies compact hashes and the cohort, runs the tests,
and executes the deterministic smoke validation. The three analysis workflows
are manual-only so an expensive analysis cannot start from an ordinary code
push:

* `model-validation.yml` selects one endpoint, profile, model, and seed.
* `repeated-cv.yml` runs ten batches of ten seeds, at most five batches in
  parallel, and aggregates a JSON/CSV distribution plus a plot.
* `endpoint-sensitivity.yml` runs the four endpoint horizons as a bounded
  matrix and uploads one aggregate machine-readable table.

Each workflow reads only committed compact data and invokes `src.cli`; no
workflow step has a source-data location or a raw-data acquisition path.
