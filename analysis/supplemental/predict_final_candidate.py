"""Predict from already-defined patient-level inputs; never infer units or retrain."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="project checkout used to resolve a relative --model path")
    parser.add_argument("--model", type=Path, required=True,
                        help="explicit serialized candidate model file")
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with exactly the defined patient-level raw predictors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="predictions.csv",
                        help="CSV filename within --output-dir (default: predictions.csv)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path
    model_path = model_path.resolve()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_name = Path(args.output_name)
    if (
        output_name.is_absolute()
        or output_name.drive
        or output_name.name != args.output_name
        or output_name.suffix.lower() != ".csv"
    ):
        raise ValueError("--output-name must be a plain filename with a .csv suffix")
    output_path = (output_dir / output_name).resolve()
    if output_path in {input_path, model_path}:
        raise ValueError("output file must not overwrite the input CSV or model file")
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")

    import joblib
    import pandas as pd

    artifact = joblib.load(model_path)
    cols = artifact["specification"]["raw_predictor_order"]
    data = pd.read_csv(input_path)
    if set(data.columns) != set(cols):
        raise ValueError("CSV must contain exactly the defined raw patient-level predictors; see model specification.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "conditional_SCD_probability": artifact["pipeline"].predict_proba(data[cols])[:, 1]
    }).to_csv(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
