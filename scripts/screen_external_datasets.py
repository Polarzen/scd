from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import requests
import wfdb
import yaml

from src.full_features import (
    FEATURE_NAMES,
    clean_rr_intervals,
    detect_r_peaks,
    preprocess_ecg,
    window_features,
)


PHYSIONET_FILES = "https://physionet.org/files"
PROBE_WINDOW_SEC = 300


def load_registry(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_header_line(line: str) -> dict[str, Any]:
    parts = line.strip().split()
    if len(parts) < 4:
        return {}
    try:
        nsig = int(parts[1])
    except ValueError:
        nsig = None
    fs_match = re.match(r"([0-9.]+)", parts[2])
    fs = float(fs_match.group(1)) if fs_match else None
    try:
        nsamp = int(parts[3])
    except ValueError:
        nsamp = None
    duration_sec = (nsamp / fs) if (nsamp is not None and fs) else None
    return {
        "header_record": parts[0],
        "nsig": nsig,
        "fs_hz": fs,
        "nsamp": nsamp,
        "duration_sec": duration_sec,
    }


def get_text(url: str, timeout: int = 30) -> tuple[bool, str, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return True, response.text, None
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def run_feature_contract_probe(record: str, pn_dir: str, fs: float, nsamp: int | None) -> dict[str, Any]:
    target_samples = int(round(float(fs) * PROBE_WINDOW_SEC))
    if isinstance(nsamp, int) and nsamp < target_samples:
        return {
            "current_feature_contract_ok": False,
            "current_feature_contract_error": f"record shorter than {PROBE_WINDOW_SEC}s",
            "feature_probe_finite_count": 0,
            "feature_probe_total_count": len(FEATURE_NAMES),
            "feature_probe_valid_rr_count": None,
            "feature_probe_removed_rr_ratio": None,
            "feature_probe_signal_finite_fraction": None,
        }

    try:
        sample = wfdb.rdrecord(
            record,
            pn_dir=pn_dir,
            sampfrom=0,
            sampto=target_samples,
            channels=[0],
            physical=True,
        )
        signal = np.asarray(sample.p_signal, dtype=np.float64)[:, 0]
        finite_fraction = float(np.isfinite(signal).mean()) if signal.size else 0.0
        if finite_fraction < 1.0:
            raise ValueError(f"non-finite waveform samples present; finite_fraction={finite_fraction:.6f}")
        ecg = preprocess_ecg(signal, float(fs))
        values = window_features(ecg, float(fs))
        peaks = detect_r_peaks(ecg, float(fs))
        raw_rr = np.diff(peaks).astype(np.float64) / float(fs) if peaks.size >= 2 else np.array([], dtype=np.float64)
        _, valid_mask = clean_rr_intervals(raw_rr)
        finite_count = sum(np.isfinite(float(values[name])) for name in FEATURE_NAMES)
        removed_ratio = (
            float((raw_rr.size - np.count_nonzero(valid_mask)) / raw_rr.size)
            if raw_rr.size
            else 0.0
        )
        return {
            "current_feature_contract_ok": True,
            "current_feature_contract_error": None,
            "feature_probe_finite_count": int(finite_count),
            "feature_probe_total_count": len(FEATURE_NAMES),
            "feature_probe_valid_rr_count": int(np.count_nonzero(valid_mask)),
            "feature_probe_removed_rr_ratio": removed_ratio,
            "feature_probe_signal_finite_fraction": finite_fraction,
        }
    except Exception as exc:
        return {
            "current_feature_contract_ok": False,
            "current_feature_contract_error": f"{type(exc).__name__}: {exc}",
            "feature_probe_finite_count": 0,
            "feature_probe_total_count": len(FEATURE_NAMES),
            "feature_probe_valid_rr_count": None,
            "feature_probe_removed_rr_ratio": None,
            "feature_probe_signal_finite_fraction": None,
        }


def probe_physionet(entry: dict[str, Any], deep_probe: bool) -> dict[str, Any]:
    pn_dir = entry["physionet_dir"].rstrip("/")
    base = f"{PHYSIONET_FILES}/{pn_dir}"
    ok, records_text, error = get_text(f"{base}/RECORDS")
    result: dict[str, Any] = {
        "remote_probe_ok": ok,
        "remote_probe_error": error,
        "record_count_observed": None,
        "sample_record": None,
        "sample_header_ok": False,
        "sample_waveform_ok": None,
        "sample_waveform_error": None,
        "sample_nsig": None,
        "sample_fs_hz": None,
        "sample_duration_sec": None,
        "current_feature_contract_ok": None,
        "current_feature_contract_error": None,
        "feature_probe_finite_count": None,
        "feature_probe_total_count": None,
        "feature_probe_valid_rr_count": None,
        "feature_probe_removed_rr_ratio": None,
        "feature_probe_signal_finite_fraction": None,
    }
    if not ok:
        return result

    records = [line.strip() for line in records_text.splitlines() if line.strip()]
    result["record_count_observed"] = len(records)
    if not records:
        result["remote_probe_ok"] = False
        result["remote_probe_error"] = "RECORDS is empty"
        return result

    record = records[0]
    result["sample_record"] = record
    ok, header_text, header_error = get_text(f"{base}/{record}.hea")
    if not ok:
        result["sample_waveform_error"] = f"header: {header_error}"
        return result

    result["sample_header_ok"] = True
    first_line = header_text.splitlines()[0] if header_text.splitlines() else ""
    parsed = parse_header_line(first_line)
    result["sample_nsig"] = parsed.get("nsig")
    result["sample_fs_hz"] = parsed.get("fs_hz")
    result["sample_duration_sec"] = parsed.get("duration_sec")

    if not deep_probe or entry.get("probe_mode") != "waveform_sample":
        return result

    try:
        fs = parsed.get("fs_hz") or 250.0
        nsamp = parsed.get("nsamp")
        target = int(float(fs) * PROBE_WINDOW_SEC)
        sampto = min(target, nsamp) if isinstance(nsamp, int) and nsamp > 0 else target
        sample = wfdb.rdrecord(record, pn_dir=pn_dir, sampto=sampto, channels=[0], physical=True)
        shape = list(np.asarray(sample.p_signal).shape)
        result["sample_waveform_ok"] = True
        result["sample_waveform_shape"] = shape
        result.update(run_feature_contract_probe(record, pn_dir, float(fs), nsamp))
    except Exception as exc:
        result["sample_waveform_ok"] = False
        result["sample_waveform_error"] = f"{type(exc).__name__}: {exc}"
        result["current_feature_contract_ok"] = False
        result["current_feature_contract_error"] = result["sample_waveform_error"]
    return result


def screen_entry(entry: dict[str, Any], deep_probe: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": entry["id"],
        "name": entry["name"],
        "source": entry["source"],
        "access": entry["access"],
        "modality": entry["modality"],
        "subjects_declared": entry.get("subjects"),
        "endpoint": entry.get("endpoint"),
        "intended_use": entry.get("intended_use"),
        "verdict": entry.get("verdict"),
        "direct_training_eligible": bool(entry.get("direct_training_eligible")),
        "blocking_reasons": " | ".join(entry.get("blocking_reasons", [])),
        "source_url": entry.get("source_url"),
    }

    if entry.get("probe_mode") == "repository_reference":
        row.update(
            {
                "remote_probe_ok": True,
                "remote_probe_error": None,
                "record_count_observed": None,
                "sample_record": None,
                "sample_header_ok": None,
                "sample_waveform_ok": None,
                "sample_waveform_error": None,
                "sample_nsig": None,
                "sample_fs_hz": None,
                "sample_duration_sec": None,
                "current_feature_contract_ok": True,
                "current_feature_contract_error": None,
                "feature_probe_finite_count": None,
                "feature_probe_total_count": len(FEATURE_NAMES),
                "feature_probe_valid_rr_count": None,
                "feature_probe_removed_rr_ratio": None,
                "feature_probe_signal_finite_fraction": None,
            }
        )
    elif entry.get("probe_mode") == "metadata_only":
        row.update(
            {
                "remote_probe_ok": None,
                "remote_probe_error": "metadata-only candidate; waveform probe intentionally not applicable",
                "record_count_observed": None,
                "sample_record": None,
                "sample_header_ok": None,
                "sample_waveform_ok": None,
                "sample_waveform_error": None,
                "sample_nsig": None,
                "sample_fs_hz": None,
                "sample_duration_sec": None,
                "current_feature_contract_ok": None,
                "current_feature_contract_error": "feature probe not applicable to metadata/RR-only screening",
                "feature_probe_finite_count": None,
                "feature_probe_total_count": len(FEATURE_NAMES),
                "feature_probe_valid_rr_count": None,
                "feature_probe_removed_rr_ratio": None,
                "feature_probe_signal_finite_fraction": None,
            }
        )
    elif entry.get("physionet_dir"):
        row.update(probe_physionet(entry, deep_probe=deep_probe))
    else:
        row.update(
            {
                "remote_probe_ok": None,
                "remote_probe_error": "manual/request-gated source; no automated download attempted",
                "record_count_observed": None,
                "sample_record": None,
                "sample_header_ok": None,
                "sample_waveform_ok": None,
                "sample_waveform_error": None,
                "sample_nsig": None,
                "sample_fs_hz": None,
                "sample_duration_sec": None,
                "current_feature_contract_ok": None,
                "current_feature_contract_error": "requires approved access and source adapter",
                "feature_probe_finite_count": None,
                "feature_probe_total_count": len(FEATURE_NAMES),
                "feature_probe_valid_rr_count": None,
                "feature_probe_removed_rr_ratio": None,
                "feature_probe_signal_finite_fraction": None,
            }
        )
    return row


def build_markdown(rows: list[dict[str, Any]], gates: list[str]) -> str:
    lines = [
        "# External dataset compatibility screening",
        "",
        "Primary task: baseline 24 h ECG/clinical prediction of 365-day SCD in chronic heart failure.",
        "",
        "A remote technical probe only checks that public files can be reached/read. A 5-minute feature-contract probe checks whether the current 20-feature extractor can process one representative waveform segment. Neither technical check makes an endpoint-incompatible dataset suitable for pooled training.",
        "",
        "## Direct-training gates",
        "",
    ]
    lines.extend(f"- {gate}" for gate in gates)
    lines.extend(
        [
            "",
            "## Screening result",
            "",
            "| Dataset | Access | Remote | 20-feature probe | Verdict | Direct pooled training | Intended use |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for row in rows:
        remote = "manual" if row["remote_probe_ok"] is None else ("pass" if row["remote_probe_ok"] else "fail")
        feature = "manual" if row["current_feature_contract_ok"] is None else ("pass" if row["current_feature_contract_ok"] else "fail")
        direct = "yes" if row["direct_training_eligible"] else "no"
        lines.append(
            f"| {row['name']} | {row['access']} | {remote} | {feature} | {row['verdict']} | {direct} | {row['intended_use']} |"
        )

    lines.extend(["", "## Blocking reasons", ""])
    for row in rows:
        reasons = row["blocking_reasons"]
        lines.append(f"### {row['name']}")
        lines.append("")
        if reasons:
            lines.append("- " + reasons.replace(" | ", "\n- "))
        else:
            lines.append("- none")
        if row.get("current_feature_contract_error"):
            lines.append(f"- feature-contract probe: {row['current_feature_contract_error']}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- MUSIC remains the only dataset in this registry approved for direct pooled training of the current 365-day SCD model.",
            "- DEFINITE and SCD-HeFT are high-priority external-validation candidates, conditional on access and endpoint/recording audit.",
            "- SCD Holter, VFDB, CUDB, MVTDB, CHFDB, and CAST-RR are auxiliary datasets for signal, feature, rhythm, or robustness studies; pooling them as ordinary 365-day SCD training rows would change the target or introduce leakage/domain shift.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/external_datasets.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/external-dataset-screening"))
    parser.add_argument("--deep-probe", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.config)
    rows = [screen_entry(entry, deep_probe=args.deep_probe) for entry in registry["datasets"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "external_dataset_screening.csv", index=False)
    (args.output_dir / "external_dataset_screening.json").write_text(
        json.dumps(
            {
                "schema_version": registry["schema_version"],
                "purpose": registry["purpose"],
                "direct_training_gates": registry["direct_training_gates"],
                "datasets": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "external_dataset_screening.md").write_text(
        build_markdown(rows, registry["direct_training_gates"]) + "\n",
        encoding="utf-8",
    )

    public_probe_failures = [
        row
        for row in rows
        if row["access"] == "open"
        and row["id"] != "music"
        and row["remote_probe_ok"] is False
    ]
    if public_probe_failures:
        names = ", ".join(row["id"] for row in public_probe_failures)
        raise SystemExit(f"public dataset technical probes failed: {names}")

    print(
        pd.DataFrame(rows)[
            ["id", "verdict", "direct_training_eligible", "remote_probe_ok", "current_feature_contract_ok"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
