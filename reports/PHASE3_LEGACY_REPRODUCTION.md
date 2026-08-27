# Phase 3 legacy reproduction

## Design

- Frozen cohort: 88 SCD patients (37 ≤365 days, 51 >365 days).
- Input: 100 aggregated fixed-window feature columns only.
- Model: ExtraTreesClassifier with the old randomized distributions, 24 AP-scored iterations.
- Validation: one patient-level OOF prediction in each outer 5-fold stratified split; hyperparameters and threshold are selected inside the outer training partition using inner 3-fold CV.
- Threshold: old training-side inner-OOF target-specificity rule at 0.70.

## OOF metrics

- Average precision: `0.4846750782872581`
- ROC AUC: `0.5696873343932167`
- Brier: `0.25783603849431097`
- Sensitivity: `0.459459459459447`
- Specificity: `0.7058823529411626`
- F1: `0.4927536231884058`
- Confusion matrix (TN, FP, FN, TP): `(36, 15, 20, 17)`

## Interpretation boundary

The fixed hourly windows are intentionally not numerically identical to the old random-window CSV. Paper metrics, if any, are reference-only and are not used as acceptance targets.

## Reproduction classification

`PARTIAL_REPRODUCTION`: cohort identity and 100-D construction are reproduced, but 97/100 aggregated features differ from the old random-window CSV.

## Paper reference (reference only)

- ROC-AUC 0.575; AP 0.511; Brier 0.257; Sensitivity 0.459; Specificity 0.784; F1 0.523.

Artifacts: `data/validation/legacy_oof_predictions.parquet`, `data/validation/legacy_model_folds.csv`.
