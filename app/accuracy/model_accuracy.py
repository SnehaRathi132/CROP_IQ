import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
import warnings

# Suppress version warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, message='.*InconsistentVersionWarning.*')

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'app', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'archive')

def compute_crop_accuracy():
    print("=== Crop Recommendation Model Accuracy ===")

    # Load data
    data_path = os.path.join(DATA_DIR, 'Crop_recommendation.csv')
    df = pd.read_csv(data_path)

    # Features and target
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    X = df[features]
    y = df[target]

    # Split data (80/20, assuming same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    model_path = os.path.join(MODELS_DIR, 'crop_model.joblib')
    model = joblib.load(model_path)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("\nFeature Importances:")
    for feature, importance in zip(features, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    # Confusion matrix (summary)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")

def compute_fertilizer_accuracy():
    print("\n=== Fertilizer Recommendation Model Accuracy ===")

    # Load data
    data_path = os.path.join(DATA_DIR, 'Fertilizer Prediction.csv')
    df = pd.read_csv(data_path)

    # Features and target (matching actual CSV column names)
    features = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    target = 'Fertilizer Name'
    X = df[features].copy()
    y = df[target]

    # Rename columns for consistency (matching the model's expectations)
    X.columns = ['temperature', 'humidity', 'moisture', 'soil_type', 'crop_type', 'nitrogen', 'potassium', 'phosphorous']

    # Split data (BUT keep original categorical columns - the pipeline handles encoding)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    model_path = os.path.join(MODELS_DIR, 'fertilizer_model.joblib')
    model = joblib.load(model_path)

    # Predict on test set - pipeline handles encoding internally
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance from the model's final estimator (if it's a pipeline)
    if hasattr(model, 'named_steps'):
        final_estimator = model.named_steps.get('model') or model.named_steps.get('classifier')
        if final_estimator and hasattr(final_estimator, 'feature_importances_'):
            print("\nFeature Importances (from pipeline's final estimator):")
            # Get feature names from the transformer
            transformer = model.named_steps.get('preprocessor')
            if transformer:
                try:
                    feature_names = transformer.get_feature_names_out()
                    for name, importance in zip(feature_names, final_estimator.feature_importances_):
                        print(f"{name}: {importance:.4f}")
                except:
                    print("Could not extract feature names from transformer.")
    elif hasattr(model, 'feature_importances_'):
        print("\nFeature Importances:")
        for feature, importance in zip(features, model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

def load_existing_metrics():
    print("\n=== Existing Metrics from Reports ===")

    # Crop metrics
    crop_metrics_path = os.path.join(REPORTS_DIR, 'crop_metrics.json')
    if os.path.exists(crop_metrics_path):
        with open(crop_metrics_path, 'r') as f:
            crop_data = json.load(f)
        print("Crop Model:")
        for result in crop_data['results']:
            print(f"  {result['model']}: Accuracy {result['accuracy']:.4f}, F1 {result['f1_macro']:.4f}")

    # Fertilizer metrics
    fert_metrics_path = os.path.join(REPORTS_DIR, 'fertilizer_metrics.json')
    if os.path.exists(fert_metrics_path):
        with open(fert_metrics_path, 'r') as f:
            fert_data = json.load(f)
        print("Fertilizer Model:")
        for result in fert_data['results']:
            print(f"  {result['model']}: Accuracy {result['accuracy']:.4f}, F1 {result['f1_macro']:.4f}")

if __name__ == "__main__":
    compute_crop_accuracy()
    compute_fertilizer_accuracy()
    load_existing_metrics()
    print("\nNote: Disease model accuracy is not computed here as it requires image data and is pre-trained.")