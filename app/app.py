# Importing essential libraries and modules

from __future__ import annotations

import ipaddress
import json
import pickle
from functools import lru_cache
from pathlib import Path
import io
import re
from html import unescape
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')

from flask import Flask, jsonify, redirect, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import joblib

import config
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic

# Optional translation dependency for Hindi output
try:
    from deep_translator import GoogleTranslator

    TRANSLATOR_AVAILABLE = True
except Exception:
    GoogleTranslator = None
    TRANSLATOR_AVAILABLE = False

# Optional torch dependencies for disease model
try:
    import torch
    from torchvision import transforms
    from PIL import Image
    from utils.model import ResNet9

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    transforms = None
    Image = None
    ResNet9 = None
    TORCH_AVAILABLE = False

# ==============================================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "Data"
REPORTS_DIR = BASE_DIR.parent / "reports"
ARCHIVE_DIR = BASE_DIR.parent / "archive"
DATA_PROCESSED_DIR = BASE_DIR.parent / "Data-processed"
DATA_RAW_DIR = BASE_DIR.parent / "Data-raw"

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model (lazy)

disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_disease_model = None


def _load_disease_model():
    if not TORCH_AVAILABLE:
        return None

    global _disease_model
    if _disease_model is not None:
        return _disease_model

    disease_model_path = MODELS_DIR / "plant_disease_model.pth"
    model = ResNet9(3, len(disease_classes))
    model.load_state_dict(torch.load(disease_model_path, map_location=torch.device("cpu")))
    model.eval()
    _disease_model = model
    return _disease_model


# Loading crop recommendation model

CROP_MODEL_PATH = MODELS_DIR / "crop_model.joblib"
LEGACY_CROP_MODEL_PATH = MODELS_DIR / "RandomForest.pkl"


def _load_crop_model():
    if CROP_MODEL_PATH.exists():
        return joblib.load(CROP_MODEL_PATH), CROP_MODEL_PATH.name
    if LEGACY_CROP_MODEL_PATH.exists():
        return pickle.load(open(LEGACY_CROP_MODEL_PATH, "rb")), LEGACY_CROP_MODEL_PATH.name
    return None, None


crop_recommendation_model, crop_model_name = _load_crop_model()

# Loading fertilizer recommendation model

FERTILIZER_MODEL_PATH = MODELS_DIR / "fertilizer_model.joblib"

fertilizer_model = joblib.load(FERTILIZER_MODEL_PATH) if FERTILIZER_MODEL_PATH.exists() else None

# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    if not api_key:
        return None

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"

    try:
        response = requests.get(complete_url, timeout=8)
        x = response.json()
    except Exception:
        return None

    if str(x.get("cod")) == "200":
        y = x.get("main", {})
        temperature = round((y.get("temp", 0.0) - 273.15), 2)
        humidity = y.get("humidity", 0)
        return temperature, humidity

    return None


def weather_fetch_by_coords(lat, lon):
    api_key = config.weather_api_key
    if not api_key:
        return None

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&lat={lat}&lon={lon}"

    try:
        response = requests.get(complete_url, timeout=8)
        x = response.json()
    except Exception:
        return None

    if str(x.get("cod")) == "200":
        y = x.get("main", {})
        temperature = round((y.get("temp", 0.0) - 273.15), 2)
        humidity = y.get("humidity", 0)
        return temperature, humidity

    return None


def reverse_geocode_city(lat, lon):
    base_url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "zoom": 10,
        "addressdetails": 1,
    }
    headers = {"User-Agent": "CropIQ/1.0 (contact: support@cropiq.local)"}

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=8)
        if not response.ok:
            return None
        data = response.json()
    except Exception:
        return None

    address = data.get("address") or {}
    for key in ("city", "town", "village", "hamlet", "municipality", "county", "state_district"):
        value = address.get(key)
        if value:
            return value

    display_name = data.get("display_name")
    if display_name:
        return display_name.split(",")[0].strip()
    return None


def _parse_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _get_lang(payload):
    lang = (payload.get("lang") or "").strip().lower()
    return "hi" if lang == "hi" else "en"


@lru_cache(maxsize=256)
def _translate_cached(text, target_lang):
    if not text:
        return text
    if target_lang != "hi":
        return text
    if not TRANSLATOR_AVAILABLE:
        return "हिंदी अनुवाद उपलब्ध नहीं है।"
    try:
        return GoogleTranslator(source="auto", target="hi").translate(text)
    except Exception:
        return "हिंदी अनुवाद उपलब्ध नहीं है।"


def _translate_text(text, lang):
    if lang != "hi":
        return text
    if not text:
        return text
    if len(text) <= 4500:
        return _translate_cached(text, "hi")

    parts = []
    current = ""
    for line in text.split("\n"):
        if not current:
            current = line
            continue
        if len(current) + len(line) + 1 <= 4500:
            current += "\n" + line
        else:
            parts.append(current)
            current = line
    if current:
        parts.append(current)

    translated_parts = [_translate_cached(part, "hi") for part in parts if part]
    return "\n".join(translated_parts)


def _strip_html(value):
    if value is None:
        return ""
    text = re.sub(r"<br\\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    return unescape(text).strip()


def _translate_html(value, lang):
    if lang != "hi":
        return value
    text = _strip_html(value)
    translated = _translate_text(text, "hi")
    if not translated:
        return translated
    return translated.replace("\n", "<br/>")


NPK_LEVELS = {"low": 0.25, "medium": 0.5, "high": 0.75}
SOIL_TYPE_NPK_LEVELS = {
    "sandy": ("low", "low", "low"),
    "loamy": ("medium", "medium", "medium"),
    "clayey": ("high", "high", "high"),
    "black": ("high", "high", "high"),
    "red": ("low", "medium", "low"),
}
SOIL_TYPE_PH = {
    "sandy": 6.3,
    "loamy": 6.7,
    "clayey": 7.2,
    "black": 7.6,
    "red": 6.2,
}


def _crop_dataset_path():
    candidates = [
        ARCHIVE_DIR / "Crop_recommendation.csv",
        DATA_PROCESSED_DIR / "crop_recommendation.csv",
        DATA_RAW_DIR / "Crop_recommendation.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _npk_quantiles():
    dataset_path = _crop_dataset_path()
    if not dataset_path:
        return None

    df = pd.read_csv(dataset_path)
    quantiles = {}
    for col in ["N", "P", "K"]:
        qs = df[col].quantile([0.25, 0.5, 0.75]).to_dict()
        quantiles[col] = {
            "low": float(qs[0.25]),
            "medium": float(qs[0.5]),
            "high": float(qs[0.75]),
        }
    return quantiles


@lru_cache(maxsize=1)
def _rainfall_quantiles():
    dataset_path = _crop_dataset_path()
    if not dataset_path:
        return None
    df = pd.read_csv(dataset_path)
    qs = df["rainfall"].quantile([0.25, 0.5, 0.75]).to_dict()
    return {
        "low": float(qs[0.25]),
        "medium": float(qs[0.5]),
        "high": float(qs[0.75]),
    }


@lru_cache(maxsize=1)
def _ph_quantiles():
    dataset_path = _crop_dataset_path()
    if not dataset_path:
        return None
    df = pd.read_csv(dataset_path)
    qs = df["ph"].quantile([0.25, 0.5, 0.75]).to_dict()
    return {
        "low": float(qs[0.25]),
        "medium": float(qs[0.5]),
        "high": float(qs[0.75]),
    }


@lru_cache(maxsize=1)
def _dataset_medians():
    dataset_path = _crop_dataset_path()
    if not dataset_path:
        return None
    df = pd.read_csv(dataset_path)
    return {
        "nitrogen": round(float(df["N"].median())),
        "phosphorous": round(float(df["P"].median())),
        "pottasium": round(float(df["K"].median())),
        "ph": round(float(df["ph"].median()), 2),
        "rainfall": round(float(df["rainfall"].median()), 2),
    }


def _normalize_level(value):
    if value is None:
        return None
    key = str(value).strip().lower()
    aliases = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "mid": "medium",
        "madhyam": "medium",
        "kam": "low",
        "zyada": "high",
        "adhik": "high",
        "कम": "low",
        "मध्यम": "medium",
        "ज्यादा": "high",
        "अधिक": "high",
    }
    return aliases.get(key, key if key in NPK_LEVELS else None)


def _normalize_soil_type(value):
    if not value:
        return None
    key = str(value).strip().lower()
    aliases = {
        "sandy": "sandy",
        "sand": "sandy",
        "loamy": "loamy",
        "loam": "loamy",
        "clayey": "clayey",
        "clay": "clayey",
        "black": "black",
        "red": "red",
    }
    return aliases.get(key)


def _guess_npk_levels(payload):
    fertility = _normalize_level(
        payload.get("fertility_level")
        or payload.get("soil_fertility")
        or payload.get("fertility")
    )
    if fertility:
        return fertility, fertility, fertility

    soil_type = _normalize_soil_type(payload.get("soil_type"))
    if soil_type and soil_type in SOIL_TYPE_NPK_LEVELS:
        return SOIL_TYPE_NPK_LEVELS[soil_type]

    return "medium", "medium", "medium"


def _resolve_npk(payload, soil_data=None):
    N = _parse_int(payload.get("nitrogen"))
    P = _parse_int(payload.get("phosphorous"))
    K = _parse_int(payload.get("pottasium"))
    estimated = False

    if None not in (N, P, K):
        return N, P, K, estimated

    if soil_data:
        if N is None and soil_data.get("nitrogen") is not None:
            N = _parse_int(soil_data.get("nitrogen"))
            estimated = True
        if P is None and soil_data.get("phosphorous") is not None:
            P = _parse_int(soil_data.get("phosphorous"))
            estimated = True
        if K is None and soil_data.get("pottasium") is not None:
            K = _parse_int(soil_data.get("pottasium"))
            estimated = True

    if None not in (N, P, K):
        return N, P, K, estimated

    quantiles = _npk_quantiles()
    if not quantiles:
        return N, P, K, estimated

    def _level_value(level, column):
        level_key = _normalize_level(level) or "medium"
        if level_key not in NPK_LEVELS:
            level_key = "medium"
        return round(quantiles[column][level_key])

    n_level_guess, p_level_guess, k_level_guess = _guess_npk_levels(payload)
    n_level = payload.get("nitrogen_level") or n_level_guess
    p_level = payload.get("phosphorous_level") or p_level_guess
    k_level = payload.get("pottasium_level") or k_level_guess

    if N is None:
        N = _level_value(n_level, "N")
        estimated = True
    if P is None:
        P = _level_value(p_level, "P")
        estimated = True
    if K is None:
        K = _level_value(k_level, "K")
        estimated = True

    return N, P, K, estimated


def _resolve_ph(payload, soil_data=None):
    ph = _parse_float(payload.get("ph"))
    if ph is not None:
        return ph, False

    if soil_data:
        soil_ph = soil_data.get("ph") or soil_data.get("phh2o")
        if soil_ph is not None:
            return _parse_float(soil_ph), True

    soil_type = _normalize_soil_type(payload.get("soil_type"))
    if soil_type and soil_type in SOIL_TYPE_PH:
        return SOIL_TYPE_PH[soil_type], True

    level = _normalize_level(payload.get("ph_level"))
    quantiles = _ph_quantiles()
    if level and quantiles:
        return round(quantiles[level], 2), True

    medians = _dataset_medians()
    if medians and medians.get("ph") is not None:
        return float(medians["ph"]), True

    return None, True


def _level_from_season(season):
    if not season:
        return None
    key = str(season).strip().lower()
    mapping = {
        "kharif": "high",
        "rabi": "medium",
        "zaid": "low",
        "kharif season": "high",
        "rabi season": "medium",
        "zaid season": "low",
    }
    return mapping.get(key)


def _resolve_rainfall(payload):
    rainfall = _parse_float(payload.get("rainfall"))
    if rainfall is not None:
        return rainfall, False

    level = _normalize_level(payload.get("rainfall_level"))
    if not level:
        level = _level_from_season(payload.get("season"))

    quantiles = _rainfall_quantiles()
    if level and quantiles:
        return round(quantiles[level], 2), True

    medians = _dataset_medians()
    if medians and medians.get("rainfall") is not None:
        return float(medians["rainfall"]), True

    return None, True


def _resolve_soil_data(payload):
    lat = _parse_float(payload.get("lat"))
    lon = _parse_float(payload.get("lon"))
    if lat is None or lon is None:
        return None
    return soil_fetch_by_coords(lat, lon)


def _request_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict(flat=True)


def _resolve_weather(payload):
    temperature = _parse_float(payload.get("temperature"))
    humidity = _parse_float(payload.get("humidity"))

    if temperature is not None and humidity is not None:
        return temperature, humidity

    lat = _parse_float(payload.get("lat"))
    lon = _parse_float(payload.get("lon"))
    if lat is not None and lon is not None:
        return weather_fetch_by_coords(lat, lon)

    city = payload.get("city")
    if city:
        return weather_fetch(city)

    return None


def predict_image(img, model=None):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    if model is None:
        model = _load_disease_model()

    if model is None or transforms is None or Image is None:
        raise RuntimeError("Disease model dependencies are not available.")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


def _depth_label_from_range(depth_range):
    top = depth_range.get("top_depth")
    bottom = depth_range.get("bottom_depth")
    if top is None or bottom is None:
        return None
    return f"{int(top)}-{int(bottom)}cm"


def _extract_soilgrids_value(layer, depth_label="0-30cm", value_key="mean"):
    depths = layer.get("depths") or []
    if not depths:
        return None

    # try exact label match first
    for depth in depths:
        label = depth.get("label")
        if not label:
            label = _depth_label_from_range(depth.get("range") or {})
        if label == depth_label:
            values = depth.get("values") or depth.get("value") or {}
            if isinstance(values, dict):
                return values.get(value_key)
            return values

    # fallback to first available depth
    depth = depths[0]
    values = depth.get("values") or depth.get("value") or {}
    if isinstance(values, dict):
        return values.get(value_key)
    return values


def _normalize_soilgrids_value(prop, value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if prop == "phh2o" and value > 14:
        return round(value / 10, 2)
    if prop in {"sand", "silt", "clay"}:
        # SoilGrids often returns g/kg; convert to %
        if value > 100:
            return round(value / 10, 2)
        return round(value, 2)
    return round(value, 2)


def _suggest_soil_type(sand_pct, clay_pct, silt_pct):
    if sand_pct is None or clay_pct is None or silt_pct is None:
        return None
    if sand_pct >= 70:
        return "Sandy"
    if clay_pct >= 35:
        return "Clayey"
    return "Loamy"


def _estimate_npk_from_soilgrids(soil):
    quantiles = _npk_quantiles() or {}
    n_q = quantiles.get("N") or {}
    p_q = quantiles.get("P") or {}
    k_q = quantiles.get("K") or {}

    def pick_level(level, qmap, fallback=0):
        if not qmap:
            return fallback
        return round(qmap.get(level, list(qmap.values())[0]))

    sand = soil.get("sand")
    clay = soil.get("clay")
    silt = soil.get("silt")
    nitrogen = soil.get("nitrogen")
    soc = soil.get("soc")

    # Nitrogen level based on SoilGrids nitrogen (g/kg) if available
    if nitrogen is not None:
        if nitrogen < 0.5:
            n_level = "low"
        elif nitrogen < 1.5:
            n_level = "medium"
        else:
            n_level = "high"
    else:
        n_level = "medium"

    # Phosphorous and Potassium levels approximated by texture + organic carbon
    if sand is not None and sand >= 70:
        p_level = "low"
        k_level = "low"
    elif clay is not None and clay >= 35:
        p_level = "high"
        k_level = "high"
    else:
        p_level = "medium"
        k_level = "medium"

    if soc is not None and soc > 20:
        if p_level == "medium":
            p_level = "high"
        if n_level == "medium":
            n_level = "high"

    return {
        "nitrogen": pick_level(n_level, n_q, 50),
        "phosphorous": pick_level(p_level, p_q, 50),
        "pottasium": pick_level(k_level, k_q, 50),
        "soil_type": _suggest_soil_type(sand, clay, silt),
    }


def soil_fetch_by_coords(lat, lon):
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": ["phh2o", "nitrogen", "sand", "silt", "clay", "soc"],
        "depth": "0-30cm",
        "value": "mean",
    }

    try:
        response = requests.get(base_url, params=params, timeout=12)
        data = response.json()
    except Exception:
        return None

    layers = data.get("properties", {}).get("layers") or data.get("layers") or []
    if not layers:
        return None

    values = {}
    for layer in layers:
        name = layer.get("name") or layer.get("property") or layer.get("code")
        if not name:
            continue
        raw = _extract_soilgrids_value(layer, depth_label="0-30cm", value_key="mean")
        values[name] = _normalize_soilgrids_value(name, raw)

    if not values:
        return None

    npk = _estimate_npk_from_soilgrids(values)
    values.update(npk)
    return values


def _top_predictions(model, data, top_n=3):
    if not hasattr(model, "predict_proba"):
        return []

    probs = model.predict_proba(data)[0]
    classes = model.classes_
    ranked = np.argsort(probs)[::-1][:top_n]
    return [
        {"label": classes[idx], "probability": round(float(probs[idx]), 4)}
        for idx in ranked
    ]


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fertilizer_advisory(crop_name, N, P, K):
    if not crop_name:
        return None
    try:
        df = pd.read_csv(DATA_DIR / "fertilizer.csv")
    except Exception:
        return None

    crop_row = df[df["Crop"] == crop_name]
    if crop_row.empty:
        return None

    nr = crop_row["N"].iloc[0]
    pr = crop_row["P"].iloc[0]
    kr = crop_row["K"].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        key = "NHigh" if n < 0 else "Nlow"
    elif max_value == "P":
        key = "PHigh" if p < 0 else "Plow"
    else:
        key = "KHigh" if k < 0 else "Klow"

    return fertilizer_dic.get(key)


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# render home page


@app.route("/")
def home():
    title = "CropIQ - Home"
    return render_template("index.html", title=title)


# render crop recommendation form page


@app.route("/crop-recommend")
def crop_recommend():
    title = "CropIQ - Crop Recommendation"
    return render_template("crop.html", title=title)


# render fertilizer recommendation form page


@app.route("/fertilizer")
def fertilizer_recommendation():
    title = "CropIQ - Fertilizer Suggestion"
    return render_template("fertilizer.html", title=title)


# ===============================================================================================

# API routes


@app.route("/api/weather", methods=["GET"])
def api_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city")

    result = None
    if lat and lon:
        result = weather_fetch_by_coords(lat, lon)
    elif city:
        result = weather_fetch(city)

    if result is None:
        return jsonify({"error": "Unable to fetch weather."}), 404

    temperature, humidity = result
    return jsonify({"temperature": temperature, "humidity": humidity})


@app.route("/api/soil", methods=["GET"])
def api_soil():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    soil = soil_fetch_by_coords(lat, lon)
    if soil is None:
        medians = _dataset_medians()
        if medians is None:
            return jsonify({"error": "Unable to fetch soil data."}), 404
        return jsonify(
            {
                "nitrogen": medians["nitrogen"],
                "phosphorous": medians["phosphorous"],
                "pottasium": medians["pottasium"],
                "ph": medians["ph"],
                "soil_type": None,
                "source": "dataset_defaults",
            }
        )

    return jsonify(
        {
            "nitrogen": soil.get("nitrogen"),
            "phosphorous": soil.get("phosphorous"),
            "pottasium": soil.get("pottasium"),
            "ph": soil.get("phh2o"),
            "soil_type": soil.get("soil_type"),
            "source": "SoilGrids",
        }
    )


@app.route("/api/reverse-geocode", methods=["GET"])
def api_reverse_geocode():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    city = reverse_geocode_city(lat, lon)
    if not city:
        return jsonify({"error": "Unable to fetch location name."}), 404

    return jsonify({"city": city})


@app.route("/api/crop", methods=["POST"])
def api_crop_prediction():
    payload = _request_payload()
    lang = _get_lang(payload)

    if crop_recommendation_model is None:
        error = _translate_text("Crop model not available.", lang)
        return jsonify({"error": error}), 500

    soil_data = None
    if None in (
        _parse_int(payload.get("nitrogen")),
        _parse_int(payload.get("phosphorous")),
        _parse_int(payload.get("pottasium")),
        _parse_float(payload.get("ph")),
    ):
        soil_data = _resolve_soil_data(payload)

    N, P, K, npk_estimated = _resolve_npk(payload, soil_data)
    ph, ph_estimated = _resolve_ph(payload, soil_data)
    rainfall, rainfall_estimated = _resolve_rainfall(payload)

    if None in (N, P, K, ph, rainfall):
        error = _translate_text("Missing or invalid soil inputs.", lang)
        return jsonify({"error": error}), 400

    weather = _resolve_weather(payload)
    if weather is None:
        error = _translate_text("Weather data not available.", lang)
        return jsonify({"error": error}), 404

    temperature, humidity = weather
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = crop_recommendation_model.predict(data)[0]
    top_predictions = _top_predictions(crop_recommendation_model, data)
    if lang == "hi":
        prediction = _translate_text(prediction, lang)
        top_predictions = [
            {**item, "label": _translate_text(item["label"], lang)}
            for item in top_predictions
        ]

    return jsonify(
        {
            "prediction": prediction,
            "top_predictions": top_predictions,
            "temperature": temperature,
            "humidity": humidity,
            "model": crop_model_name,
            "npk_estimated": npk_estimated,
            "ph_estimated": ph_estimated,
            "rainfall_estimated": rainfall_estimated,
        }
    )


@app.route("/api/fertilizer", methods=["POST"])
def api_fertilizer_prediction():
    payload = _request_payload()
    lang = _get_lang(payload)

    if fertilizer_model is None:
        error = _translate_text("Fertilizer model not available.", lang)
        return jsonify({"error": error}), 500

    soil_data = None
    if None in (
        _parse_int(payload.get("nitrogen")),
        _parse_int(payload.get("phosphorous")),
        _parse_int(payload.get("pottasium")),
    ):
        soil_data = _resolve_soil_data(payload)

    N, P, K, npk_estimated = _resolve_npk(payload, soil_data)
    moisture = _parse_float(payload.get("moisture"))
    soil_type = payload.get("soil_type")
    crop_type = payload.get("crop_type")

    if None in (N, P, K, moisture) or not soil_type or not crop_type:
        error = _translate_text("Missing or invalid fertilizer inputs.", lang)
        return jsonify({"error": error}), 400

    weather = _resolve_weather(payload)
    if weather is None:
        error = _translate_text("Weather data not available.", lang)
        return jsonify({"error": error}), 404

    temperature, humidity = weather

    features = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "humidity": humidity,
                "moisture": moisture,
                "soil_type": soil_type,
                "crop_type": crop_type,
                "nitrogen": N,
                "potassium": K,
                "phosphorous": P,
            }
        ]
    )

    prediction = fertilizer_model.predict(features)[0]
    if lang == "hi":
        prediction = _translate_text(prediction, lang)
    confidence = None
    if hasattr(fertilizer_model, "predict_proba"):
        probabilities = fertilizer_model.predict_proba(features)[0]
        confidence = round(float(probabilities.max()), 4)

    advisory_crop = payload.get("cropname")
    advisory = _fertilizer_advisory(advisory_crop, N, P, K)
    if advisory and lang == "hi":
        advisory = _translate_html(advisory, lang)

    return jsonify(
        {
            "prediction": prediction,
            "confidence": confidence,
            "temperature": temperature,
            "humidity": humidity,
            "advisory": advisory,
            "npk_estimated": npk_estimated,
        }
    )


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route("/crop-predict", methods=["POST"])
def crop_prediction():
    title = "CropIQ - Crop Recommendation"

    payload = _request_payload()
    lang = _get_lang(payload)

    if crop_recommendation_model is None:
        message = _translate_text("Crop model not available.", lang)
        return render_template("try_again.html", title=title, message=message)

    soil_data = None
    if None in (
        _parse_int(payload.get("nitrogen")),
        _parse_int(payload.get("phosphorous")),
        _parse_int(payload.get("pottasium")),
        _parse_float(payload.get("ph")),
    ):
        soil_data = _resolve_soil_data(payload)

    N, P, K, npk_estimated = _resolve_npk(payload, soil_data)
    ph, ph_estimated = _resolve_ph(payload, soil_data)
    rainfall, rainfall_estimated = _resolve_rainfall(payload)

    if None in (N, P, K, ph, rainfall):
        message = _translate_text("Invalid soil inputs.", lang)
        return render_template("try_again.html", title=title, message=message)

    weather = _resolve_weather(payload)
    if weather is None:
        message = _translate_text("Weather data not available.", lang)
        return render_template("try_again.html", title=title, message=message)

    temperature, humidity = weather

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]
    top_predictions = _top_predictions(crop_recommendation_model, data)
    if lang == "hi":
        final_prediction = _translate_text(final_prediction, lang)
        top_predictions = [
            {**item, "label": _translate_text(item["label"], lang)}
            for item in top_predictions
        ]

    estimation_note_parts = []
    if npk_estimated:
        estimation_note_parts.append("Soil nutrients (NPK) were estimated")
    if ph_estimated:
        estimation_note_parts.append("soil pH was estimated")
    if rainfall_estimated:
        estimation_note_parts.append("rainfall was estimated")
    estimation_note = None
    if estimation_note_parts:
        estimation_note = ", ".join(estimation_note_parts) + "."
        if lang == "hi":
            estimation_note = _translate_text(estimation_note, lang)

    return render_template(
        "crop-result.html",
        prediction=final_prediction,
        top_predictions=top_predictions,
        temperature=temperature,
        humidity=humidity,
        estimation_note=estimation_note,
        title=title,
    )


# render fertilizer recommendation result page


@app.route("/fertilizer-predict", methods=["POST"])
def fert_recommend():
    title = "CropIQ - Fertilizer Suggestion"

    payload = _request_payload()
    lang = _get_lang(payload)

    if fertilizer_model is None:
        message = _translate_text("Fertilizer model not available.", lang)
        return render_template("try_again.html", title=title, message=message)

    crop_name = payload.get("cropname")
    soil_data = None
    if None in (
        _parse_int(payload.get("nitrogen")),
        _parse_int(payload.get("phosphorous")),
        _parse_int(payload.get("pottasium")),
    ):
        soil_data = _resolve_soil_data(payload)

    N, P, K, npk_estimated = _resolve_npk(payload, soil_data)
    moisture = _parse_float(payload.get("moisture"))
    soil_type = payload.get("soil_type")
    crop_type = payload.get("crop_type")

    if None in (N, P, K, moisture) or not soil_type or not crop_type:
        message = _translate_text("Invalid fertilizer inputs.", lang)
        return render_template("try_again.html", title=title, message=message)

    weather = _resolve_weather(payload)
    if weather is None:
        message = _translate_text("Weather data not available.", lang)
        return render_template("try_again.html", title=title, message=message)

    temperature, humidity = weather

    features = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "humidity": humidity,
                "moisture": moisture,
                "soil_type": soil_type,
                "crop_type": crop_type,
                "nitrogen": N,
                "potassium": K,
                "phosphorous": P,
            }
        ]
    )

    fertilizer_prediction = fertilizer_model.predict(features)[0]
    if lang == "hi":
        fertilizer_prediction = _translate_text(fertilizer_prediction, lang)
    confidence = None
    if hasattr(fertilizer_model, "predict_proba"):
        probabilities = fertilizer_model.predict_proba(features)[0]
        confidence = round(float(probabilities.max()), 4)

    advisory = _fertilizer_advisory(crop_name, N, P, K)
    if advisory and lang == "hi":
        advisory = _translate_html(advisory, lang)

    return render_template(
        "fertilizer-result.html",
        recommendation=Markup(str(advisory)) if advisory else None,
        fertilizer_prediction=fertilizer_prediction,
        confidence=confidence,
        temperature=temperature,
        humidity=humidity,
        title=title,
    )


# render disease prediction result page


@app.route("/disease-predict", methods=["GET", "POST"])
def disease_prediction():
    title = "CropIQ - Disease Detection"

    if request.method == "POST":
        payload = _request_payload()
        lang = _get_lang(payload)
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.get("file")
        if not file:
            return render_template("disease.html", title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            disease_text = disease_dic.get(prediction, "")
            if lang == "hi":
                disease_text = _translate_html(disease_text, lang)
            prediction = Markup(str(disease_text))
            return render_template("disease-result.html", prediction=prediction, title=title)
        except Exception:
            return render_template(
                "disease.html",
                title=title,
                error=_translate_text(
                    "Disease model is unavailable. Install PyTorch + torchvision to enable this feature.",
                    lang,
                ),
            )
    return render_template("disease.html", title=title)


# ===============================================================================================
if __name__ == "__main__":
    import ssl
    # Create a self-signed certificate for local HTTPS development
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Harvestify"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Write certificate and key to temporary files
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem') as cert_file:
            cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
            cert_path = cert_file.name
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.key') as key_file:
            key_file.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            key_path = key_file.name
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_path, key_path)
        
        print("Starting HTTPS server on https://localhost:5000")
        print("Note: You may see a security warning in your browser - click 'Advanced' and 'Proceed to localhost'")
        app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context, debug=False)
        
        # Clean up temporary files
        os.unlink(cert_path)
        os.unlink(key_path)
        
    except ImportError:
        print("HTTP server running on http://localhost:5000")       
        print("Accessible from other devices at: http://<your-computer-ip>:5000")        
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Failed to create SSL certificate: {e}")
        print("Falling back to HTTP on http://localhost:5000")
