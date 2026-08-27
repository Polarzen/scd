# MUSIC Phase 1 — DATA AUDIT

Generated: `2026-08-27T13:17:40+00:00`

> Scope: official metadata, coding/definition files, manifests, filesystem metadata, and WFDB headers only. No `.dat` waveform content was opened, no features were extracted, and Phase 2 was not started.

## Summary

| Item | Result |
|---|---:|
| MUSIC version | 1.0.1 (confirmed_from_locked_release_directory_name_plus_complete_expected_layout_and_official_manifest) |
| Source DOI | 10.13026/z3m7-rf58 |
| Official patient rows | 992 |
| Unique Patient IDs | 992 |
| Holter records / patients | 936 / 936 |
| High-resolution records / patients | 687 / 687 |
| Metadata/file availability mismatches | 0 |
| Missing records/files | 0 |
| Header anomalies | 0 |
| Small-file/header SHA256 mismatches | 0 |
| SCD total | 94 |
| SCD <=90 / <=180 / <=365 / <=730 days | 11 / 19 / 38 / 64 |
| Permanent AF (unique union) | 192 |
| Estimated legacy 120s windows | 21816 |
| Estimated full 5min windows | 258315 |
| Estimated MUSIC-CI size | 38.6–59.76 MiB |
| Likely <=80 MiB | True |

## Record distributions

- Sampling frequency (all): `{"1000": 687, "200": 936}`
- Channel count (all): `{"2": 38, "3": 1585}`
- Lead combinations (all): `{"X|Y": 38, "X|Y|Z": 1585}`
- Holter duration seconds, min / P25 / median / P75 / max: `29136 / 81330 / 84688.5 / 86389 / 86399`
- High-resolution duration seconds, min / P25 / median / P75 / max: `797.333 / 1196 / 1196 / 1196 / 1196`
- Follow-up days, min / P25 / median / P75 / max: `33 / 1082.25 / 1805 / 1913 / 2065`

## Mapping and integrity

- Metadata Holter yes but header missing: `[]`
- Holter header present but metadata not yes: `[]`
- Metadata high-resolution yes but header missing: `[]`
- High-resolution header present but metadata not yes: `[]`
- Headers with patient absent from metadata: `[]`
- Records without header: `[]`
- Headers absent from RECORDS: `[]`
- Header anomalies: `[]`
- Manifest-declared missing files: `[]`
- Declared signal files missing: `[]`
- `.dat` files shorter than the header requirement: `[]`
- `.dat` files with bytes beyond the header-declared minimum (reported, not treated as corruption): `[{'path': 'Holter_ECG/P0070.dat', 'minimum_bytes_from_header': 65904020, 'actual_bytes': 65933600, 'trailing_bytes': 29580}, {'path': 'Holter_ECG/P0125.dat', 'minimum_bytes_from_header': 49959000, 'actual_bytes': 49960000, 'trailing_bytes': 1000}, {'path': 'Holter_ECG/P0174.dat', 'minimum_bytes_from_header': 69113204, 'actual_bytes': 69113600, 'trailing_bytes': 396}, {'path': 'Holter_ECG/P0247.dat', 'minimum_bytes_from_header': 69111576, 'actual_bytes': 69112000, 'trailing_bytes': 424}, {'path': 'Holter_ECG/P0258.dat', 'minimum_bytes_from_header': 69111384, 'actual_bytes': 69112000, 'trailing_bytes': 616}, {'path': 'Holter_ECG/P0302.dat', 'minimum_bytes_from_header': 69112316, 'actual_bytes': 69112800, 'trailing_bytes': 484}, {'path': 'Holter_ECG/P0305.dat', 'minimum_bytes_from_header': 69111752, 'actual_bytes': 69112000, 'trailing_bytes': 248}, {'path': 'Holter_ECG/P0310.dat', 'minimum_bytes_from_header': 69115384, 'actual_bytes': 69116000, 'trailing_bytes': 616}, {'path': 'Holter_ECG/P0384.dat', 'minimum_bytes_from_header': 67063588, 'actual_bytes': 67064000, 'trailing_bytes': 412}, {'path': 'Holter_ECG/P0421.dat', 'minimum_bytes_from_header': 69112140, 'actual_bytes': 69112800, 'trailing_bytes': 660}, {'path': 'Holter_ECG/P0424.dat', 'minimum_bytes_from_header': 66537420, 'actual_bytes': 66537600, 'trailing_bytes': 180}, {'path': 'Holter_ECG/P0425.dat', 'minimum_bytes_from_header': 56541536, 'actual_bytes': 57684800, 'trailing_bytes': 1143264}, {'path': 'Holter_ECG/P0427.dat', 'minimum_bytes_from_header': 63322020, 'actual_bytes': 63322400, 'trailing_bytes': 380}, {'path': 'Holter_ECG/P0448.dat', 'minimum_bytes_from_header': 69113092, 'actual_bytes': 69113600, 'trailing_bytes': 508}, {'path': 'Holter_ECG/P0460.dat', 'minimum_bytes_from_header': 69111504, 'actual_bytes': 69112000, 'trailing_bytes': 496}, {'path': 'Holter_ECG/P0462.dat', 'minimum_bytes_from_header': 69117508, 'actual_bytes': 69117600, 'trailing_bytes': 92}, {'path': 'Holter_ECG/P0466.dat', 'minimum_bytes_from_header': 66003904, 'actual_bytes': 66004000, 'trailing_bytes': 96}, {'path': 'Holter_ECG/P0489.dat', 'minimum_bytes_from_header': 69117252, 'actual_bytes': 69118400, 'trailing_bytes': 1148}, {'path': 'Holter_ECG/P0490.dat', 'minimum_bytes_from_header': 69110368, 'actual_bytes': 69110400, 'trailing_bytes': 32}, {'path': 'Holter_ECG/P0493.dat', 'minimum_bytes_from_header': 69111568, 'actual_bytes': 69112000, 'trailing_bytes': 432}, {'path': 'Holter_ECG/P0494.dat', 'minimum_bytes_from_header': 69109652, 'actual_bytes': 69111200, 'trailing_bytes': 1548}, {'path': 'Holter_ECG/P0495.dat', 'minimum_bytes_from_header': 69115916, 'actual_bytes': 69116000, 'trailing_bytes': 84}, {'path': 'Holter_ECG/P0588.dat', 'minimum_bytes_from_header': 59570284, 'actual_bytes': 59570400, 'trailing_bytes': 116}, {'path': 'Holter_ECG/P0592.dat', 'minimum_bytes_from_header': 58795500, 'actual_bytes': 58796000, 'trailing_bytes': 500}, {'path': 'Holter_ECG/P0628.dat', 'minimum_bytes_from_header': 69116516, 'actual_bytes': 69118400, 'trailing_bytes': 1884}, {'path': 'Holter_ECG/P0661.dat', 'minimum_bytes_from_header': 69072028, 'actual_bytes': 69092000, 'trailing_bytes': 19972}, {'path': 'Holter_ECG/P0674.dat', 'minimum_bytes_from_header': 69072020, 'actual_bytes': 69116000, 'trailing_bytes': 43980}, {'path': 'Holter_ECG/P0706.dat', 'minimum_bytes_from_header': 59806208, 'actual_bytes': 59806400, 'trailing_bytes': 192}, {'path': 'Holter_ECG/P0738.dat', 'minimum_bytes_from_header': 69111816, 'actual_bytes': 69113600, 'trailing_bytes': 1784}, {'path': 'Holter_ECG/P0792.dat', 'minimum_bytes_from_header': 69112764, 'actual_bytes': 69112800, 'trailing_bytes': 36}, {'path': 'Holter_ECG/P0810.dat', 'minimum_bytes_from_header': 101550000, 'actual_bytes': 102825462, 'trailing_bytes': 1275462}, {'path': 'Holter_ECG/P0813.dat', 'minimum_bytes_from_header': 69109528, 'actual_bytes': 69110400, 'trailing_bytes': 872}, {'path': 'Holter_ECG/P0814.dat', 'minimum_bytes_from_header': 69118916, 'actual_bytes': 69119200, 'trailing_bytes': 284}, {'path': 'Holter_ECG/P0829.dat', 'minimum_bytes_from_header': 69112764, 'actual_bytes': 69112800, 'trailing_bytes': 36}, {'path': 'Holter_ECG/P0831.dat', 'minimum_bytes_from_header': 69112472, 'actual_bytes': 69112800, 'trailing_bytes': 328}, {'path': 'Holter_ECG/P0832.dat', 'minimum_bytes_from_header': 69111180, 'actual_bytes': 69111200, 'trailing_bytes': 20}, {'path': 'Holter_ECG/P0898.dat', 'minimum_bytes_from_header': 69072028, 'actual_bytes': 69080000, 'trailing_bytes': 7972}, {'path': 'Holter_ECG/P0962.dat', 'minimum_bytes_from_header': 65615452, 'actual_bytes': 65616000, 'trailing_bytes': 548}, {'path': 'Holter_ECG/P1073.dat', 'minimum_bytes_from_header': 58811060, 'actual_bytes': 58811200, 'trailing_bytes': 140}, {'path': 'High-resolution_ECG/P0070_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0125_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0247_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0258_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0302_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0305_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0310_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0384_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0421_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0424_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0425_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0427_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0466_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0489_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0588_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0592_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0661_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0674_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0706_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}, {'path': 'High-resolution_ECG/P0898_H.dat', 'minimum_bytes_from_header': 4783998, 'actual_bytes': 7176000, 'trailing_bytes': 2392002}]`
- SHA256 checked for 1628 metadata/header files; mismatches: `[]`
- `.dat` content SHA256: **NOT_CHECKED_IN_PHASE_1_BY_DESIGN** (0 checked). Official hashes remain recorded in `SHA256SUMS.txt`.

## Outcomes, rhythm, and PVC

- Cause-of-death code distribution: `{"0": 726, "1": 61, "3": 94, "6": 100, "7": 11}`
- Endpoint audit definition: Cause of death == 3, per subject-info_codes.csv.
- Permanent AF: high-resolution=183, Holter=171, unique union=192.
- PVC field availability: `{"Number of ventricular premature beats in 24h": {"present": true, "non_missing_count": 932, "missing_count": 60, "summary": {"count": 932, "min": 0, "p25": 38.75, "median": 274, "p75": 1613.75, "max": 44500}}, "Ventricular Extrasystole": {"present": true, "non_missing_count": 927, "missing_count": 65, "summary": {"count": 927, "min": 0, "p25": 1, "median": 2, "p75": 2, "max": 2}}, "Number of ventricular premature contractions per hour": {"present": true, "non_missing_count": 755, "missing_count": 237, "summary": {"count": 755, "min": 0, "p25": 0, "median": 0, "p75": 0, "max": 1025}}}`

## Size estimate

Official metadata compressed independently with deterministic gzip: **255336 bytes**. Combined headers gzip size: **20857 bytes**.

Estimated compact package: **38.6–59.76 MiB**. 20 float64 features per theoretical window; Parquet feature compression factor 0.72-1.05, encoded window metadata factor 0.20-0.75, plus 5 MiB cohort/validation/report allowance; must be replaced by measured Phase 4 size

## Existing code finding

The existing `scd_dataset.csv` contains 88 patients and is not the official full cohort. Existing extraction code can silently skip failed patients/windows; it is therefore not used as audit truth.

## Hard gate

**PASSED**. Reasons: `[]`

Phase 2 was not started and is not authorized by this report.
