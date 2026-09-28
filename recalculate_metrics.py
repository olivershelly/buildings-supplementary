#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

MODELS = (
    ("SVR", "svr_predicted_cei"),
    ("RF", "rf_predicted_cei"),
    ("XGBoost", "xgboost_predicted_cei"),
    ("BO-SVR", "bo_svr_predicted_cei"),
    ("BO-RF", "bo_rf_predicted_cei"),
    ("BO-XGBoost", "bo_xgboost_predicted_cei"),
)
COLUMNS = ("sample_id", "split", "simulated_cei") + tuple(c for _, c in MODELS)
TABLE_COLUMNS = ("model", "test_r2", "test_rmse", "train_r2", "train_rmse")


def read_samples(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Expected columns: {', '.join(COLUMNS)}")
        records = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line_number}")
            record = {"sample_id": row["sample_id"], "split": row["split"]}
            if record["split"] not in ("train", "test"):
                raise ValueError(f"Invalid split on row {line_number}")
            for column in COLUMNS[2:]:
                try:
                    value = Decimal(row[column])
                except InvalidOperation as exc:
                    raise ValueError(f"Invalid number in {column}, row {line_number}") from exc
                if not value.is_finite():
                    raise ValueError(f"Non-finite number in {column}, row {line_number}")
                record[column] = value
            records.append(record)
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    if set(ids) != {f"D{i:03d}" for i in range(1, 117)}:
        raise ValueError("Expected exactly the 116 IDs D001-D116")
    if Counter(r["split"] for r in records) != {"train": 92, "test": 24}:
        raise ValueError("Expected 92 training and 24 test samples")
    return records


def calculate(records):
    """Calculate R2 and RMSE for each model and split."""
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    metrics = []
    # Use extra precision because XGBoost's training R2 is very close to 1.
    with localcontext() as context:
        context.prec = 60
        for model, prediction_column in MODELS:
            for split in ("train", "test"):
                selected = by_split[split]
                n = Decimal(len(selected))
                mean = sum(r["simulated_cei"] for r in selected) / n
                sst = sum((r["simulated_cei"] - mean) ** 2 for r in selected)
                sse = sum((r[prediction_column] - r["simulated_cei"]) ** 2 for r in selected)
                if sst <= 0:
                    raise ValueError(f"Constant targets in {split}")
                r2 = 1 - sse / sst
                rmse = (sse / n).sqrt()
                variance = sst / n
                # The ratio is undefined for an exactly perfect prediction.
                ratio = None if sse == 0 else rmse ** 2 / (1 - r2)
                if ratio is not None and abs(ratio - variance) > Decimal("1e-40") * max(abs(variance), Decimal(1)):
                    raise ValueError(f"Consistency identity failed for {model} / {split}")
                metrics.append({"model": model, "split": split, "n": int(n), "r2": r2,
                                "rmse": rmse, "sse": sse, "sst": sst,
                                "sst_over_n": variance, "rmse_squared_over_one_minus_r2": ratio})
    return metrics


def make_table(metrics):
    lookup = {(m["model"], m["split"]): m for m in metrics}
    return [dict(zip(TABLE_COLUMNS,
                    [model, format(lookup[model, "test"]["r2"], ".4f"),
                     format(lookup[model, "test"]["rmse"], ".4f"),
                     format(lookup[model, "train"]["r2"], ".4f"),
                     format(lookup[model, "train"]["rmse"], ".4f")]))
            for model, _ in MODELS]


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "sample_level_predictions.csv")
    args = parser.parse_args()
    records = read_samples(args.input)
    metrics = calculate(records)
    table = make_table(metrics)
    variance = {row["split"]: row["sst_over_n"] for row in metrics}
    print("116 samples: 92 training and 24 test; six models share these assignments.")
    print("R-squared is dimensionless; RMSE unit: kgCO2/(m2*a).")
    print("Recomputed Table 9(a), rounded to four decimal places:")
    print("Model       Test R2  Test RMSE  Train R2  Train RMSE")
    for row in table:
        print(f"{row['model']:<11} {row['test_r2']:>7} {row['test_rmse']:>10} {row['train_r2']:>9} {row['train_rmse']:>11}")
    print("\nHigh-precision calculations (60-digit decimal arithmetic):")
    for row in metrics:
        ratio = row["rmse_squared_over_one_minus_r2"]
        ratio_text = "undefined (R2 exactly 1)" if ratio is None else format(ratio, ".25g")
        print(f"{row['model']} / {row['split']}: n={row['n']}, "
              f"R2={row['r2']:.25g}, RMSE={row['rmse']:.25g}, "
              f"RMSE^2/(1-R2)={ratio_text}")
    print(f"\nTraining SST/n: {variance['train']:.25g}")
    print(f"Test SST/n: {variance['test']:.25g}")
    print("Use high-precision values for the consistency ratio, not the rounded table.")


if __name__ == "__main__":
    main()