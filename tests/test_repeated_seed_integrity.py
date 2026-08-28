import pandas as pd
import pytest

from src.optimization_reporting import IntegrityError, validate_repeated_seed_integrity


def _fixtures(seeds=(0, 1)):
    summaries = []
    oof = []
    for candidate in ("baseline", "candidate"):
        for seed in seeds:
            summaries.append({"candidate": candidate, "profile": "all20", "model": "logistic", "seed": seed, "endpoint_horizon_days": 365})
            for patient, label, probability in (("P0", 0, 0.1), ("P1", 1, 0.8)):
                oof.append(
                    {
                        "candidate": candidate,
                        "profile": "all20",
                        "model": "logistic",
                        "seed": seed,
                        "patient_id": patient,
                        "outer_fold": 1,
                        "y_true": label,
                        "y_prob": probability,
                        "y_pred": int(probability >= 0.5),
                        "threshold": 0.5,
                        "endpoint_horizon_days": 365,
                    }
                )
    return summaries, pd.DataFrame(oof)


def test_nonformal_integrity_accepts_exact_explicit_seed_set():
    summaries, oof = _fixtures()
    result = validate_repeated_seed_integrity(summaries, oof, expected_seeds={0, 1})
    assert result["summary_count"] == 4
    assert result["oof_rows"] == 8


@pytest.mark.parametrize("mutation", ["missing_seed", "duplicate_summary", "duplicate_patient", "mixed_metadata"])
def test_integrity_fails_closed_for_repeated_run_corruption(mutation):
    summaries, oof = _fixtures()
    if mutation == "missing_seed":
        summaries = [row for row in summaries if not (row["candidate"] == "candidate" and row["seed"] == 1)]
    elif mutation == "duplicate_summary":
        summaries.append(dict(summaries[0]))
    elif mutation == "duplicate_patient":
        oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    else:
        summaries[1]["profile"] = "different"
    with pytest.raises(IntegrityError):
        validate_repeated_seed_integrity(summaries, oof, expected_seeds={0, 1})


def test_formal_mode_requires_the_frozen_zero_to_ninety_nine_seed_set():
    summaries, oof = _fixtures()
    with pytest.raises(IntegrityError, match="formal mode"):
        validate_repeated_seed_integrity(summaries, oof, formal=True, expected_seeds={0, 1})
