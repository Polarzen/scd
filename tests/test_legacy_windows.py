import numpy as np
import pandas as pd
from pathlib import Path

from src import legacy_features as lf


class _Record:
    def __init__(self, signal):
        self.p_signal = signal[:, None]


def test_fixed_schedule_retains_outside_windows_and_bounds_reads(monkeypatch):
    calls = []
    signal = np.zeros(1_800, dtype=float)

    def fake_rdrecord(stem, **kwargs):
        calls.append((stem, kwargs))
        start, stop = kwargs["sampfrom"], kwargs["sampto"]
        return _Record(signal[start:stop])

    monkeypatch.setattr(lf.wfdb, "rdrecord", fake_rdrecord)
    table = lf.extract_fixed_windows(
        "fake/P0001",
        fs=10.0,
        sample_count=len(signal),
        patient_id="P0001",
    )

    assert len(table) == 24
    assert table["window_idx"].tolist() == list(range(24))
    assert table["window_start_sec"].tolist() == list(lf.fixed_window_starts())
    assert table["window_status"].iloc[0] == lf.SUCCESS
    assert (table["window_status"].iloc[1:] == lf.OUTSIDE_RECORD).all()
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["channels"] == [0]
    assert kwargs["sampto"] - kwargs["sampfrom"] == 120 * 10
    assert table["removed_rr_count"].eq(0).all()


def test_short_read_is_not_padded(monkeypatch):
    def fake_rdrecord(stem, **kwargs):
        return _Record(np.zeros(10, dtype=float))

    monkeypatch.setattr(lf.wfdb, "rdrecord", fake_rdrecord)
    table = lf.extract_fixed_windows("fake/P0001", fs=10.0, sample_count=None, max_windows=1)
    assert table.loc[0, "window_status"] == lf.SHORT_READ
    assert table.loc[0, "actual_samples"] == 10
    assert table.loc[0, "sig_mean"] != table.loc[0, "sig_mean"]  # NaN


def test_generated_window_manifest_contract():
    path = Path("data/features/legacy_120s/windows.parquet")
    if not path.is_file():
        return
    table = pd.read_parquet(path)
    assert len(table) == 88 * 24
    assert table.groupby("patient_id").size().eq(24).all()
    assert table["window_id"].between(0, 23).all()
    assert (table["end_sec"] - table["start_sec"]).eq(120).all()
    assert table.sort_values(["patient_id", "window_id"]).groupby("patient_id")["start_sec"].diff().dropna().eq(3600).all()
    records = pd.read_parquet("data/cohort/records.parquet")
    holter = records.loc[records["record_type"].eq("HOLTER"), ["record_id", "sample_count"]]
    checked = table.merge(holter, on="record_id", how="left", validate="many_to_one")
    assert checked["sample_count"].notna().all()
    assert (checked.loc[checked["window_within_record"], "end_sample"] <= checked.loc[checked["window_within_record"], "sample_count"]).all()
    assert checked.loc[~checked["window_within_record"], "waveform_read_success"].eq(False).all()
