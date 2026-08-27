# Legacy fixed-window feature comparison

This is an audit-only comparison with `scd_dataset.csv`; the old CSV is never used as a replacement feature table.

- New patient rows: 88
- Old CSV patient rows: 88
- Aggregated feature columns compared: 100 / 100
- Exact features: 3
- Near features: 0
- Different features: 97
- Expected differences: Phase 3 uses fixed hourly windows; the old CSV records random legacy windows.

| Feature | Matched | Missing | Mean abs | Median abs | Max abs | Correlation | Class |
|---|---:|---:|---:|---:|---:|---:|---|
| sig_mean_mean | 88 | 0 | 0.0115935 | 0.00811787 | 0.0560408 | 0.9928 | DIFFERENT |
| sig_mean_std | 88 | 0 | 0.00976771 | 0.00636645 | 0.0458226 | 0.858374 | DIFFERENT |
| sig_mean_p10 | 88 | 0 | 0.0217117 | 0.0128506 | 0.259958 | 0.966696 | DIFFERENT |
| sig_mean_p50 | 88 | 0 | 0.014977 | 0.0107783 | 0.166204 | 0.985719 | DIFFERENT |
| sig_mean_p90 | 88 | 0 | 0.0160316 | 0.0101435 | 0.0858177 | 0.984257 | DIFFERENT |
| sig_std_mean | 88 | 0 | 0.00284091 | 0 | 0.0833333 | 0.596042 | DIFFERENT |
| sig_std_std | 88 | 0 | 0.00841031 | 2.72671e-17 | 0.28233 | 0.70019 | DIFFERENT |
| sig_std_p10 | 88 | 0 | 1.26162e-16 | 1.11022e-16 | 2.22045e-16 | 0.517402 | EXACT |
| sig_std_p50 | 88 | 0 | 0 | 0 | 0 | nan | EXACT |
| sig_std_p90 | 88 | 0 | 0 | 0 | 0 | nan | EXACT |
| sig_p2p_mean | 88 | 0 | 0.995514 | 0.701015 | 4.98902 | 0.94385 | DIFFERENT |
| sig_p2p_std | 88 | 0 | 0.875863 | 0.563685 | 3.45082 | 0.81166 | DIFFERENT |
| sig_p2p_p10 | 88 | 0 | 0.739934 | 0.304415 | 6.6545 | 0.887108 | DIFFERENT |
| sig_p2p_p50 | 88 | 0 | 0.93312 | 0.51994 | 5.84109 | 0.941656 | DIFFERENT |
| sig_p2p_p90 | 88 | 0 | 2.04379 | 1.23075 | 9.851 | 0.889112 | DIFFERENT |
| sig_skew_mean | 88 | 0 | 0.226472 | 0.122597 | 1.56581 | 0.986705 | DIFFERENT |
| sig_skew_std | 88 | 0 | 0.310873 | 0.138116 | 2.44728 | 0.761193 | DIFFERENT |
| sig_skew_p10 | 88 | 0 | 0.412565 | 0.281356 | 3.57782 | 0.956693 | DIFFERENT |
| sig_skew_p50 | 88 | 0 | 0.238371 | 0.146616 | 2.13577 | 0.985718 | DIFFERENT |
| sig_skew_p90 | 88 | 0 | 0.272585 | 0.140387 | 2.19172 | 0.974741 | DIFFERENT |
| sig_kurt_mean | 88 | 0 | 2.72963 | 1.13819 | 29.4527 | 0.879664 | DIFFERENT |
| sig_kurt_std | 88 | 0 | 5.82887 | 1.04051 | 83.2856 | 0.599231 | DIFFERENT |
| sig_kurt_p10 | 88 | 0 | 1.18603 | 0.786923 | 15.5031 | 0.92791 | DIFFERENT |
| sig_kurt_p50 | 88 | 0 | 1.4285 | 0.801401 | 11.5695 | 0.9681 | DIFFERENT |
| sig_kurt_p90 | 88 | 0 | 5.88994 | 1.34565 | 95.2416 | 0.739595 | DIFFERENT |
| beats_mean | 88 | 0 | 5.73 | 4.3125 | 19.0455 | 0.969952 | DIFFERENT |
| beats_std | 88 | 0 | 6.54041 | 4.31672 | 37.5451 | 0.676406 | DIFFERENT |
| beats_p10 | 88 | 0 | 7.21591 | 4.9 | 52.9 | 0.934689 | DIFFERENT |
| beats_p50 | 88 | 0 | 6.22159 | 5 | 23.5 | 0.966965 | DIFFERENT |
| beats_p90 | 88 | 0 | 11.3136 | 7.75 | 66.3 | 0.905326 | DIFFERENT |
| beats_per_min_mean | 88 | 0 | 2.865 | 2.15625 | 9.52273 | 0.969952 | DIFFERENT |
| beats_per_min_std | 88 | 0 | 3.2702 | 2.15836 | 18.7725 | 0.676406 | DIFFERENT |
| beats_per_min_p10 | 88 | 0 | 3.60795 | 2.45 | 26.45 | 0.934689 | DIFFERENT |
| beats_per_min_p50 | 88 | 0 | 3.1108 | 2.5 | 11.75 | 0.966965 | DIFFERENT |
| beats_per_min_p90 | 88 | 0 | 5.65682 | 3.875 | 33.15 | 0.905326 | DIFFERENT |
| mean_rr_mean | 88 | 0 | 0.0287457 | 0.019551 | 0.147485 | 0.962909 | DIFFERENT |
| mean_rr_std | 88 | 0 | 0.0416916 | 0.0169862 | 0.379824 | 0.521926 | DIFFERENT |
| mean_rr_p10 | 88 | 0 | 0.0401531 | 0.0273785 | 0.367417 | 0.890013 | DIFFERENT |
| mean_rr_p50 | 88 | 0 | 0.0294256 | 0.0247162 | 0.117707 | 0.968422 | DIFFERENT |
| mean_rr_p90 | 88 | 0 | 0.0464569 | 0.0259828 | 0.693361 | 0.88803 | DIFFERENT |
| sdnn_mean | 88 | 0 | 0.0473147 | 0.0146492 | 0.576063 | 0.785511 | DIFFERENT |
| sdnn_std | 88 | 0 | 0.108839 | 0.011734 | 1.96798 | 0.475904 | DIFFERENT |
| sdnn_p10 | 88 | 0 | 0.019766 | 0.0125753 | 0.117042 | 0.942857 | DIFFERENT |
| sdnn_p50 | 88 | 0 | 0.0232432 | 0.0131206 | 0.190567 | 0.940715 | DIFFERENT |
| sdnn_p90 | 88 | 0 | 0.0898823 | 0.0176405 | 3.17538 | 0.624587 | DIFFERENT |
| rmssd_mean | 88 | 0 | 0.0606447 | 0.0205928 | 0.711274 | 0.831804 | DIFFERENT |
| rmssd_std | 88 | 0 | 0.132398 | 0.0174774 | 2.12736 | 0.542806 | DIFFERENT |
| rmssd_p10 | 88 | 0 | 0.0226625 | 0.0149748 | 0.101123 | 0.971755 | DIFFERENT |
| rmssd_p50 | 88 | 0 | 0.0368417 | 0.0191788 | 0.253026 | 0.937587 | DIFFERENT |
| rmssd_p90 | 88 | 0 | 0.122441 | 0.0314406 | 3.97358 | 0.606811 | DIFFERENT |
| pnn50_mean | 88 | 0 | 4.12049 | 2.44252 | 19.4995 | 0.97897 | DIFFERENT |
| pnn50_std | 88 | 0 | 3.76109 | 2.50093 | 16.2913 | 0.876955 | DIFFERENT |
| pnn50_p10 | 88 | 0 | 5.09099 | 2.92602 | 33.8858 | 0.970777 | DIFFERENT |
| pnn50_p50 | 88 | 0 | 5.77582 | 2.57101 | 55.5708 | 0.948921 | DIFFERENT |
| pnn50_p90 | 88 | 0 | 6.82763 | 3.05396 | 64.626 | 0.898959 | DIFFERENT |
| mean_hr_mean | 88 | 0 | 2.63493 | 1.97173 | 9.48469 | 0.974156 | DIFFERENT |
| mean_hr_std | 88 | 0 | 2.71947 | 2.07053 | 16.1425 | 0.720715 | DIFFERENT |
| mean_hr_p10 | 88 | 0 | 3.48486 | 2.29596 | 21.3436 | 0.94419 | DIFFERENT |
| mean_hr_p50 | 88 | 0 | 3.18651 | 2.71519 | 11.5108 | 0.966257 | DIFFERENT |
| mean_hr_p90 | 88 | 0 | 5.68929 | 3.75457 | 33.4403 | 0.904467 | DIFFERENT |
| rr_cv_mean | 88 | 0 | 0.0371939 | 0.0187708 | 0.337384 | 0.912103 | DIFFERENT |
| rr_cv_std | 88 | 0 | 0.0586875 | 0.0170223 | 0.737693 | 0.528997 | DIFFERENT |
| rr_cv_p10 | 88 | 0 | 0.0246987 | 0.0187851 | 0.125413 | 0.965662 | DIFFERENT |
| rr_cv_p50 | 88 | 0 | 0.0294291 | 0.015005 | 0.205209 | 0.950267 | DIFFERENT |
| rr_cv_p90 | 88 | 0 | 0.0742137 | 0.0225847 | 1.19941 | 0.773719 | DIFFERENT |
| rr_sampen_mean | 88 | 0 | 0.0946462 | 0.0583202 | 0.416898 | 0.974279 | DIFFERENT |
| rr_sampen_std | 88 | 0 | 0.0959225 | 0.0484781 | 0.692172 | 0.609896 | DIFFERENT |
| rr_sampen_p10 | 88 | 0 | 0.0928124 | 0.0547172 | 0.736557 | 0.952407 | DIFFERENT |
| rr_sampen_p50 | 88 | 0 | 0.0952271 | 0.0724365 | 0.640178 | 0.972613 | DIFFERENT |
| rr_sampen_p90 | 88 | 0 | 0.162105 | 0.0962881 | 1.06683 | 0.930135 | DIFFERENT |
| rr_apen_mean | 88 | 0 | 0.038605 | 0.0318586 | 0.135339 | 0.968864 | DIFFERENT |
| rr_apen_std | 88 | 0 | 0.0289076 | 0.0214329 | 0.115722 | 0.775974 | DIFFERENT |
| rr_apen_p10 | 88 | 0 | 0.0520878 | 0.0378266 | 0.25212 | 0.945957 | DIFFERENT |
| rr_apen_p50 | 88 | 0 | 0.0481871 | 0.0387221 | 0.207009 | 0.954562 | DIFFERENT |
| rr_apen_p90 | 88 | 0 | 0.0678055 | 0.0469975 | 0.491766 | 0.850096 | DIFFERENT |
| rr_dfa_alpha_mean | 88 | 0 | 0.0497386 | 0.0318697 | 0.24172 | 0.926391 | DIFFERENT |
| rr_dfa_alpha_std | 88 | 0 | 0.0492229 | 0.0364705 | 0.285411 | 0.674628 | DIFFERENT |
| rr_dfa_alpha_p10 | 88 | 0 | 0.0667658 | 0.0503617 | 0.305399 | 0.853134 | DIFFERENT |
| rr_dfa_alpha_p50 | 88 | 0 | 0.0601739 | 0.0476157 | 0.258474 | 0.901148 | DIFFERENT |
| rr_dfa_alpha_p90 | 88 | 0 | 0.092145 | 0.0704588 | 0.374433 | 0.881861 | DIFFERENT |
| pow_lf_mean | 88 | 0 | 0.0203753 | 0.012662 | 0.108829 | 0.984499 | DIFFERENT |
| pow_lf_std | 88 | 0 | 0.0222641 | 0.0127726 | 0.138672 | 0.789084 | DIFFERENT |
| pow_lf_p10 | 88 | 0 | 0.0300497 | 0.0148098 | 0.318015 | 0.940021 | DIFFERENT |
| pow_lf_p50 | 88 | 0 | 0.0222099 | 0.011831 | 0.195887 | 0.975274 | DIFFERENT |
| pow_lf_p90 | 88 | 0 | 0.0351469 | 0.0211853 | 0.277717 | 0.938898 | DIFFERENT |
| pow_mf_mean | 88 | 0 | 0.017129 | 0.0116563 | 0.0764952 | 0.984311 | DIFFERENT |
| pow_mf_std | 88 | 0 | 0.0175287 | 0.0108236 | 0.0869746 | 0.824429 | DIFFERENT |
| pow_mf_p10 | 88 | 0 | 0.0313546 | 0.0163655 | 0.247754 | 0.932665 | DIFFERENT |
| pow_mf_p50 | 88 | 0 | 0.0167507 | 0.010888 | 0.151167 | 0.98039 | DIFFERENT |
| pow_mf_p90 | 88 | 0 | 0.0252995 | 0.0111438 | 0.360152 | 0.923427 | DIFFERENT |
| pow_hf_mean | 88 | 0 | 0.0070874 | 0.00409679 | 0.0400491 | 0.986188 | DIFFERENT |
| pow_hf_std | 88 | 0 | 0.00698407 | 0.00382083 | 0.0577704 | 0.891592 | DIFFERENT |
| pow_hf_p10 | 88 | 0 | 0.00730333 | 0.00346028 | 0.0494598 | 0.976093 | DIFFERENT |
| pow_hf_p50 | 88 | 0 | 0.00802222 | 0.00370946 | 0.0662246 | 0.976959 | DIFFERENT |
| pow_hf_p90 | 88 | 0 | 0.0129238 | 0.00712172 | 0.117112 | 0.963193 | DIFFERENT |
| pow_hf_ratio_mean | 88 | 0 | 0.0102533 | 0.00528061 | 0.0590952 | 0.984095 | DIFFERENT |
| pow_hf_ratio_std | 88 | 0 | 0.0117424 | 0.00664962 | 0.187608 | 0.811266 | DIFFERENT |
| pow_hf_ratio_p10 | 88 | 0 | 0.00922155 | 0.00378119 | 0.060659 | 0.979478 | DIFFERENT |
| pow_hf_ratio_p50 | 88 | 0 | 0.0108097 | 0.00443608 | 0.0861487 | 0.978148 | DIFFERENT |
| pow_hf_ratio_p90 | 88 | 0 | 0.0183369 | 0.00894197 | 0.15769 | 0.96487 | DIFFERENT |
