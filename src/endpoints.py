"""Dynamic MUSIC endpoint construction from preserved official outcome fields."""

from __future__ import annotations

from typing import Any

import pandas as pd


SUPPORTED_HORIZONS = {90, 180, 365, 730}
ALLOWED_STATES = {"POSITIVE", "NEGATIVE", "CENSORED", "COMPETING_EVENT", "UNKNOWN"}
CAUSE_EVENT_TYPES = {
    "0": "NO_SCD_RECORDED",
    "1": "NON_CARDIAC_DEATH",
    "3": "SCD",
    "6": "PUMP_FAILURE_DEATH",
    "7": "PUMP_FAILURE_DEATH",
}


def _clean_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text.replace(",", "."))
    except ValueError:
        return text
    return str(int(parsed)) if parsed.is_integer() else text


def build_endpoint(subjects: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Return one horizon-specific endpoint row per subject.

    Binary labels exist only for POSITIVE/NEGATIVE observations. Censoring,
    competing events, and unknown source states remain explicitly unevaluable.
    The input frame is never mutated.
    """
    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError(f"horizon_days must be one of {sorted(SUPPORTED_HORIZONS)}")
    required = {"patient_id", "followup_days", "cause_of_death_raw", "event_source_valid"}
    missing = sorted(required - set(subjects.columns))
    if missing:
        raise KeyError(f"subjects is missing endpoint source columns: {missing}")
    if subjects["patient_id"].duplicated().any():
        raise ValueError("patient_id must be unique")

    rows: list[dict[str, Any]] = []
    for source in subjects.loc[:, list(subjects.columns)].to_dict("records"):
        patient_id = source["patient_id"]
        followup_value = source["followup_days"]
        followup = None if pd.isna(followup_value) else float(followup_value)
        cause = _clean_code(source["cause_of_death_raw"])
        valid_value = source["event_source_valid"]
        valid = False if pd.isna(valid_value) else bool(valid_value)
        event_type = CAUSE_EVENT_TYPES.get(cause, "UNKNOWN")
        if not valid or followup is None or followup < 0 or cause not in CAUSE_EVENT_TYPES:
            state = "UNKNOWN"
            label = None
        elif cause == "3":
            state = "POSITIVE" if followup <= horizon_days else "NEGATIVE"
            label = 1 if state == "POSITIVE" else 0
        elif cause in {"1", "6", "7"}:
            state = "COMPETING_EVENT" if followup <= horizon_days else "NEGATIVE"
            label = None if state == "COMPETING_EVENT" else 0
        else:
            state = "NEGATIVE" if followup >= horizon_days else "CENSORED"
            label = 0 if state == "NEGATIVE" else None
        rows.append(
            {
                "patient_id": patient_id,
                "endpoint_horizon_days": horizon_days,
                "endpoint_state": state,
                "binary_label_if_evaluable": label,
                "time_to_event": followup,
                "event_type": event_type,
            }
        )
    result = pd.DataFrame(rows)
    result["patient_id"] = result["patient_id"].astype("string")
    result["endpoint_horizon_days"] = result["endpoint_horizon_days"].astype("Int64")
    result["endpoint_state"] = result["endpoint_state"].astype("string")
    result["binary_label_if_evaluable"] = result["binary_label_if_evaluable"].astype("Int64")
    result["time_to_event"] = result["time_to_event"].astype("float64")
    result["event_type"] = result["event_type"].astype("string")
    return result
