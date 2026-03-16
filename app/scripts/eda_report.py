from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = BASE_DIR / "archive"
REPORTS_DIR = BASE_DIR / "reports"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _crop_eda() -> dict:
    df = pd.read_csv(ARCHIVE_DIR / "Crop_recommendation.csv")
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "label_distribution": df["label"].value_counts().to_dict(),
        "numeric_summary": df.describe().round(3).to_dict(),
        "correlation": df.drop(columns=["label"]).corr().round(3).to_dict(),
    }
    return summary


def _fertilizer_eda() -> dict:
    df = pd.read_csv(ARCHIVE_DIR / "Fertilizer Prediction.csv")
    df = df.rename(columns=lambda c: c.strip())
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "fertilizer_distribution": df["Fertilizer Name"].value_counts().to_dict(),
        "soil_type_distribution": df["Soil Type"].value_counts().to_dict(),
        "crop_type_distribution": df["Crop Type"].value_counts().to_dict(),
        "numeric_summary": df.select_dtypes(include="number").describe().round(3).to_dict(),
    }
    return summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": _now_iso(),
        "crop": _crop_eda(),
        "fertilizer": _fertilizer_eda(),
    }

    output_path = REPORTS_DIR / "eda_summary.json"
    _save_json(output_path, payload)
    print(f"EDA summary written to {output_path}")


if __name__ == "__main__":
    main()
