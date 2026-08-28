import inspect

import numpy as np

from src import full_features as ff


class _Record:
    def __init__(self, signal):
        self.p_signal = np.asarray(signal, dtype=float)[:, None]


def test_full_schedule_is_complete_nonoverlapping_and_unlimited():
    assert ff.complete_window_starts(359.99) == ()
    assert ff.complete_window_starts(360) == (60,)
    assert ff.complete_window_starts(1_000) == (60, 360, 660)
    assert ff.tail_seconds(1_000) == 40.0
    assert ff.full_window_count(86_399) > 24


def test_full_reader_uses_one_exact_segment_and_no_padding(monkeypatch):
    calls = []

    def fake_rdrecord(stem, **kwargs):
        calls.append((stem, kwargs))
        assert kwargs["channels"] == [0]
        assert kwargs["sampto"] - kwargs["sampfrom"] == 300 * 10
        return _Record(np.zeros(300 * 10))

    monkeypatch.setattr(ff.wfdb, "rdrecord", fake_rdrecord)
    result = ff.extract_full_windows("fake/P0001", fs=10, sample_count=3_600, patient_id="P0001")
    assert len(result) == 1
    assert len(calls) == 1
    assert result.loc[0, "window_start_sec"] == 60
    assert result.loc[0, "window_end_sec"] == 360
    assert result.loc[0, "actual_samples"] == 3_000


def test_extractor_signature_does_not_accept_outcome_or_label():
    names = set(inspect.signature(ff.extract_one_window).parameters)
    assert "label" not in names
    assert "outcome" not in names
    assert "followup_days" not in names
