# Reproducibility protocol

## Environment

Use a fresh Python environment and the committed lock file. The lock is the
single dependency specification used by CI and the manual analysis workflows:

```bash
python -m pip install --requirement requirements-lock.txt
```

Do not copy an environment between machines. `python scripts/verify_all.py`
checks the compact SHA256 manifest, source headers and metadata, cohort keys,
provenance, endpoint states, and Parquet compression. `python -m src.cli
verify` checks the model feature and subject schemas without fitting a model.

## Fixed analysis commands

The following commands are the local equivalents of the GitHub workflows. All
paths point to committed compact tables; no source waveform location is
needed.

```bash
python scripts/verify_all.py
python -m src.cli verify
python -m src.cli validate --endpoint 365 --profile all20 --model extratrees --seed 42 --n-jobs 1 --output-dir data/validation/full_model
python -m src.cli repeated-cv --endpoint 365 --profile all20 --models extratrees,logistic,dummy --seed-start 1000 --seed-stop 1099 --n-jobs 1 --output-dir data/validation/repeated
python -m src.cli endpoint-sensitivity --horizons 90,180,365,730 --profile all20 --model extratrees --seed 42 --n-jobs 1 --output-dir data/validation/endpoint-sensitivity
```

The deterministic CI smoke check is intentionally small:

```bash
python -m src.cli validate --endpoint 365 --profile all20 --model dummy --seed 42 --smoke --output-dir .tmp/ci-smoke
```

The formal validation defaults are five outer folds, three inner folds,
average-precision scoring, 24 ExtraTrees search iterations, one model worker,
the target-specificity threshold of 0.70, and 2,000 bootstrap resamples with
seed 42. The smoke flag reduces search, folds, and bootstrap resamples only
for the CI oracle.

## Independence and endpoint interpretation

One patient is one independent analysis unit. A patient can contribute many
windows, but those windows are repeated measurements that are aggregated to a
single patient row. Nested CV keeps all windows from a patient on the same
side of each split and emits exactly one OOF prediction per evaluable patient.

The primary 365-day analysis stratifies risk over a 365-day follow-up endpoint.
It is not a warning about an event in the next few hours and must not be
reported as an hourly alarm. Sensitivity analyses at 90, 180, 365, and 730
days make the time horizon explicit. Censored and competing-event states are
retained in the endpoint table and excluded from the binary positive/negative
fit according to the data contract.

## Legacy and full representations

`legacy_120s` is retained to make the historical 120-second representation
auditable. Its 100 aggregate columns and patient-level model are a partial
reproduction of the old random-window result; the comparison report records
where the feature values differ.

`full_5min` is the current frozen representation. It uses the full 5-minute
window contract, 20 base features, five summary statistics, and validity
counts. `all20` and `physiology_only` are explicit profiles; the latter
removes `SIGNAL_QUALITY` features. Results from the two families must be
labelled separately and are not interchangeable.

## Raw-source boundary and provenance

Raw source is required only when preprocessing or window definitions change,
R-peak detection is rerun, or a new waveform feature is introduced. It is not
required for any P4-D validation command above. The source is MUSIC 1.0.1,
DOI [10.13026/z3m7-rf58](https://doi.org/10.13026/z3m7-rf58); source-recovery
files and official hashes are under `data/source_exact/`. The compact handoff
is analysis-lossless and source-recoverable for the approved analyses, not a
raw waveform compression archive.

## Workflow outputs

`model-validation.yml` is a manual single-run workflow with endpoint choices
90/180/365/730, profiles `all20`/`physiology_only`, models
`extratrees`/`logistic`/`dummy`, and an explicit seed. Its artifact contains
OOF predictions, metrics, folds, bootstrap intervals, population inputs, and
configuration snapshots.

`repeated-cv.yml` is manual-only and runs 10 batches × 10 consecutive seeds,
with at most five batches in parallel. Its aggregate artifact contains the
per-seed distribution in CSV/JSON, summary statistics, metadata, and a PNG
distribution plot.

`endpoint-sensitivity.yml` is manual-only and runs the four horizons as a
matrix, then uploads one sorted machine-readable endpoint table and the
configuration snapshot.

The canonical publication-facing command list and artifact expectations are
also recorded in [`reports/GITHUB_HANDOFF.md`](reports/GITHUB_HANDOFF.md).
