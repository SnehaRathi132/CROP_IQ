from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = BASE_DIR / "archive"
APP_DIR = BASE_DIR / "app"
MODELS_DIR = APP_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_comparison(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _save_confusion(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(path)


def train_crop(seed: int, test_size: float) -> dict:
    dataset_path = ARCHIVE_DIR / "Crop_recommendation.csv"
    df = pd.read_csv(dataset_path)
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target = "label"

    X = df[features]
    y = df[target]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    models: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    results: list[dict] = []
    best_name = None
    best_model = None
    best_score = (-1.0, -1.0)  # accuracy, f1
    best_cm = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds, labels=labels)

        results.append(
            {
                "model": name,
                "accuracy": round(float(acc), 6),
                "f1_macro": round(float(f1), 6),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        if (acc, f1) > best_score:
            best_score = (acc, f1)
            best_name = name
            best_model = model
            best_cm = cm

    if best_model is None:
        raise RuntimeError("No crop model trained.")

    model_path = MODELS_DIR / "crop_model.joblib"
    joblib.dump(best_model, model_path)

    comparison_path = REPORTS_DIR / "crop_model_comparison.csv"
    _save_comparison(comparison_path, results)

    cm_path = REPORTS_DIR / "crop_confusion_matrix.csv"
    _save_confusion(cm_path, best_cm, labels)

    metrics = {
        "dataset": str(dataset_path.relative_to(BASE_DIR)),
        "features": features,
        "target": target,
        "label_count": len(labels),
        "labels": labels,
        "comparison_csv": str(comparison_path.relative_to(BASE_DIR)),
        "confusion_matrix_csv": str(cm_path.relative_to(BASE_DIR)),
        "best_model": best_name,
        "best_model_path": str(model_path.relative_to(BASE_DIR)),
        "results": results,
        "trained_at": _now_iso(),
    }

    metrics_path = REPORTS_DIR / "crop_metrics.json"
    _save_json(metrics_path, metrics)

    return metrics


def _clean_fertilizer_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Temparature": "temperature",
        "Humidity ": "humidity",
        "Moisture": "moisture",
        "Soil Type": "soil_type",
        "Crop Type": "crop_type",
        "Nitrogen": "nitrogen",
        "Potassium": "potassium",
        "Phosphorous": "phosphorous",
        "Fertilizer Name": "fertilizer",
    }
    df = df.rename(columns=rename_map)
    df.columns = [c.strip() for c in df.columns]
    return df


def train_fertilizer(seed: int, test_size: float) -> dict:
    dataset_path = ARCHIVE_DIR / "Fertilizer Prediction.csv"
    df = pd.read_csv(dataset_path)
    df = _clean_fertilizer_columns(df)

    features = [
        "temperature",
        "humidity",
        "moisture",
        "soil_type",
        "crop_type",
        "nitrogen",
        "potassium",
        "phosphorous",
    ]
    target = "fertilizer"

    X = df[features]
    y = df[target]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    numeric_features = [
        "temperature",
        "humidity",
        "moisture",
        "nitrogen",
        "potassium",
        "phosphorous",
    ]
    categorical_features = ["soil_type", "crop_type"]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", encoder, categorical_features),
        ]
    )

    models: dict[str, Pipeline] = {
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DecisionTreeClassifier(random_state=seed)),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }

    results: list[dict] = []
    best_name = None
    best_model = None
    best_score = (-1.0, -1.0)
    best_cm = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds, labels=labels)

        results.append(
            {
                "model": name,
                "accuracy": round(float(acc), 6),
                "f1_macro": round(float(f1), 6),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        if (acc, f1) > best_score:
            best_score = (acc, f1)
            best_name = name
            best_model = model
            best_cm = cm

    if best_model is None:
        raise RuntimeError("No fertilizer model trained.")

    model_path = MODELS_DIR / "fertilizer_model.joblib"
    joblib.dump(best_model, model_path)

    comparison_path = REPORTS_DIR / "fertilizer_model_comparison.csv"
    _save_comparison(comparison_path, results)

    cm_path = REPORTS_DIR / "fertilizer_confusion_matrix.csv"
    _save_confusion(cm_path, best_cm, labels)

    metrics = {
        "dataset": str(dataset_path.relative_to(BASE_DIR)),
        "features": features,
        "target": target,
        "label_count": len(labels),
        "labels": labels,
        "comparison_csv": str(comparison_path.relative_to(BASE_DIR)),
        "confusion_matrix_csv": str(cm_path.relative_to(BASE_DIR)),
        "best_model": best_name,
        "best_model_path": str(model_path.relative_to(BASE_DIR)),
        "results": results,
        "trained_at": _now_iso(),
    }

    metrics_path = REPORTS_DIR / "fertilizer_metrics.json"
    _save_json(metrics_path, metrics)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop and fertilizer models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--skip-crop", action="store_true")
    parser.add_argument("--skip-fertilizer", action="store_true")

    args = parser.parse_args()

    _ensure_dirs()

    summary = {}
    if not args.skip_crop:
        summary["crop"] = train_crop(args.seed, args.test_size)
    if not args.skip_fertilizer:
        summary["fertilizer"] = train_fertilizer(args.seed, args.test_size)

    summary_path = REPORTS_DIR / "training_summary.json"
    _save_json(summary_path, summary)
    print(f"Training complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
