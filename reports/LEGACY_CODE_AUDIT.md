# Legacy code audit

Phase 2 base commit: `b6ab167f6eb7873d8a7d024f9ab996aa4a0acf8a`

Evidence classes: **A** paper-explicit; **B** current legacy code; **C** official MUSIC data; **D** not confirmable. No paper/manuscript file is present locally, so no item can currently be classified A.

| Topic | Evidence | Finding |
|---|---|---|
| Patient selection | B+C | `Cause of death == 3`, valid follow-up/cause, and physical Holter header. Phase 3 uses equivalent frozen Phase 2 state `cause_of_death_raw=3`, `event_source_valid`, `has_holter`. |
| SCD definition | B+C | Official cause code `3`. |
| `<=365 d` | B | `followup_days <= 365`, inclusive. |
| `>365 d` | B | SCD patients with `followup_days > 365`. |
| SCD-only design | B | Default legacy cohort contains only patients whose eventual recorded outcome is SCD. This is a legacy case-only timing design, not the complete MUSIC cohort. |
| Holter lookup | B | `<Holter_ECG>/<patient_id>.hea`; old code does not consult the official availability flag. |
| ECG channel | B | Reads all channels, then uses `p_signal[:, 0]`; effectively the first available channel. Phase 3 reads only channel index 0. |
| Filtering | B | Median center, third-order Butterworth bandpass 0.5–40 Hz, `filtfilt`, recenter, population-SD scaling. |
| R-peak detector | B | `find_peaks(filtered_ecg**2)`, minimum distance `0.25*fs`, height at the window's 93rd percentile. |
| RR abnormal values | B | No range or local-median cleaning. RR is `diff(peaks)/fs`. |
| NN definition | D | No NN distinction exists in the old implementation. |
| Interpolation | B | None. `interp1d` is imported but unused. |
| Detrending | B | No ECG/RR preprocessing detrend. DFA internally linearly detrends integrated-RR segments. |
| Welch | B | Not used. `welch` is imported but unused; spectral features use a raw FFT periodogram. |
| LF/HF bands | B+D | Old signal-power bands are 0.5–4, 4–15, and 15–40 Hz. They are not HRV LF/HF bands 0.04–0.15/0.15–0.40 Hz. |
| Nonlinear features | B | RR sample entropy, approximate entropy, and DFA alpha, using the exact old-code implementations. |
| Morphology features | B+D | Signal mean, SD, peak-to-peak, skew and kurtosis; the paper taxonomy is unavailable. |
| Signal quality | D | No legacy signal-quality algorithm or rejection threshold exists. |
| Heart-rate dynamics | B | Beat count, beats/min, mean RR, mean HR and RR variability measures. |
| Window rule | B vs Phase 3 | Old code randomly samples arbitrary start samples without replacement (24 default; fast mode 12). Phase 3 is explicitly fixed at `60+3600*k` seconds, `k=0..23`. This conflict is preserved and is expected to affect feature equality. |
| Patient aggregation | B | Per feature: mean, sample SD (`ddof=1`, one value →0), P10, P50, P90; NaNs omitted per feature. |
| Missing handling | B | Read exceptions silently skipped in old code; patients with fewer than four successful windows dropped; feature NaNs median-imputed in the model pipeline. Phase 3 retains every theoretical window and records failures. |

## Frozen 20-feature implementation

The code produces exactly 20 base values: 7 direct signal/beat measures, 6 time-domain RR measures, 3 nonlinear RR measures and 4 FFT signal-power measures. `config/legacy_features.yaml` is the authoritative Phase 3 mapping. Category assignment is a transparent Phase 3 descriptive taxonomy because the paper taxonomy is unavailable.

## Model audit

- Default model: `ExtraTreesClassifier`, 800 trees, unlimited depth, `min_samples_leaf=2`, `min_samples_split=6`, `max_features=sqrt`, balanced class weights, random state 42.
- Old full tuning: randomized search, 24 iterations, AP scoring, default 4-fold inner CV, random state 42. Phase 3 requirement overrides inner CV to 3 folds.
- Old ordinary CV does not nest hyperparameter search. Phase 3 therefore uses the requested strict 5×3 nested design and records this difference.
- Threshold rule: training-side negative-probability quantile targeting specificity 0.70, using `nextafter` above the selected negative probability.
- Old code fixed `RANDOM_STATE=42`; per-row waveform random-window seeds were `42+i*104729`, but those random windows are not used by the fixed Phase 3 window manifest.

## Unresolved items

- No local paper text exists to independently validate the old code.
- The requested example RR cleaning/interpolation/Welch pipeline is not implemented in the old code and is therefore not introduced in this reproduction.
- Because the mandated fixed hourly windows conflict with the old random fast-mode 12-window dataset, exact numeric feature reproduction is not expected and will not be forced.
