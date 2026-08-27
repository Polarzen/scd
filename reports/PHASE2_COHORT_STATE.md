# MUSIC Phase 2 — Complete Cohort State

> Phase 1 passed. This build copied official small files and headers byte-for-byte and did not open `.dat` waveform content. Phase 3 was not started.

## Cohort and records

| Item | Count |
|---|---:|
| Official patients / subjects.parquet rows | 992 / 992 |
| Patient IDs preserved | True |
| Holter / no Holter | 936 / 56 |
| High-resolution / none | 687 / 305 |
| Records total (Holter + high-resolution) | 1623 (936 + 687) |
| Metadata/file inconsistencies | 0 |
| Header anomalies / missing dat / missing hea | 0 / 0 / 0 |
| AF / rhythm unknown | 192 / 2 |
| PVC count available | 932 |
| Follow-up / cause-of-death available | 992 / 992 |

## Endpoint audit

| Horizon (days) | POSITIVE | NEGATIVE | CENSORED | COMPETING_EVENT | UNKNOWN |
|---:|---:|---:|---:|---:|---:|
| 90 | 11 | 978 | 0 | 3 | 0 |
| 180 | 19 | 947 | 7 | 19 | 0 |
| 365 | 38 | 894 | 15 | 45 | 0 |
| 730 | 64 | 816 | 25 | 87 | 0 |

At 365 days, evaluable binary subjects are POSITIVE + NEGATIVE = 932. Competing events and censored observations remain explicit and are not forced to zero.

## Recorded SCD interval

For the official SCD category, the **baseline enrollment/Holter to recorded SCD outcome interval** has n=94 and min/P25/median/P75/max days = {'count': 94, 'min': 33, 'p25': 215, 'median': 479.5, 'p75': 840.25, 'max': 1726}.

Bins: {'le_90d': 11, 'd91_180': 8, 'd181_365': 19, 'd366_730': 26, 'gt_730d': 30}

## Unresolved but preserved

- Official Holter rhythm raw code(s) absent from the supplied code table: ['4']; decoded value remains null/UNKNOWN where selected.
- `pvc_burden` is null for every subject because Phase 2 has no approved denominator.
- Signal eligibility is nullable and marked `PENDING_SIGNAL_QC`; no signal-quality label was invented.

Phase 3 was not started.
