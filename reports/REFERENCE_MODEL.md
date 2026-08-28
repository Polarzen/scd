# Full cohort reference model

Reference: 365-day SCD risk stratification, all20, ExtraTrees, seed 42; patient is the independent unit.

| Model | ROC-AUC | AP | Brier | Sensitivity | Specificity | F1 | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| extratrees | 0.584210 | 0.056772 | 0.097353 | 0.555556 | 0.615385 | 0.099338 | 0.05454545454545454 | 0.971963 |
| logistic | 0.585634 | 0.056137 | 0.043542 | 0.555556 | 0.643491 | 0.106007 | 0.05859375 | 0.973154 |
| dummy | 0.477208 | 0.036833 | 0.036938 | 0.000000 | 1.000000 | 0.000000 | None | 0.961593 |

ExtraTrees ROC-AUC 95% patient bootstrap CI: 0.478760–0.690503.
ExtraTrees AP 95% patient bootstrap CI: 0.032542–0.111801.
ExtraTrees Brier 95% patient bootstrap CI: 0.087569–0.107224.

Metrics are descriptive validation outputs, not performance acceptance thresholds.
