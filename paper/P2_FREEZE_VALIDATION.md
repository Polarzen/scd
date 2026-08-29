# P2 frozen candidate defensive validation

- Freeze: `P2-FROZEN-V1`
- Source formal seed run: `33189047550`
- Exact seed set: 0-99 (100 seeds)
- OOF integrity: 87,800 rows = 878 unique patients × 100 seeds; one OOF prediction per patient per seed.
- Frozen population: 878 patients, 37 positives; AF 164 patients / 10 positives.

## Unique-patient aggregation across 100 seeds

The 100 OOF probabilities for each patient were averaged first; metrics below therefore use each patient exactly once.

| subgroup | N | positives | AUC | AP | Brier | Brier Skill | calibration slope | calibration intercept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 878 | 37 | 0.6487 | 0.1227 | 0.0395 | 0.0204 | 1.6011 | 1.6962 |
| AF | 164 | 10 | 0.7688 | 0.2680 | 0.0548 | 0.0437 | 2.8657 | 5.3094 |
| sinus | 714 | 27 | 0.6006 | 0.0876 | 0.0360 | 0.0092 | 1.1477 | 0.2886 |

The AF subgroup remains exploratory because it contains only 10 unique positive patients. Patient-bootstrap intervals are stored in `defensive_validation.json`; the 100-seed split distributions are stored separately and must not be called population confidence intervals.

No model, feature, endpoint, threshold strategy, or patient inclusion rule was retuned during this validation.
