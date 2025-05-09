import joblib
import pandas as pd
from parse2database import StegoTraceAnalyzer
from stegofeatureextractor import StegoFeatureExtractor

MODEL_PATH = "./training/models/rf_clean_vs_stego.pkl"
SINGLE_MODEL_PATH = "./training/models/rf_single_clean_vs_stego.pkl"
DB_PATH = "./training/database/steganalysis.db"

def predict_from_images(original_path: str, stego_path: str):
    """
    Predict the steganography tool used by comparing a stego image to its original.

    Args:
        original_path (str): Path to the original image.
        stego_path (str): Path to the stego image.

    Returns:
        dict: Contains prediction, confidence, and class probabilities.
    """
    trace = StegoTraceAnalyzer(original_path, stego_path, DB_PATH)
    features = trace.run_all()

    clf = joblib.load(MODEL_PATH)
    df = pd.DataFrame([features]).reindex(columns=clf.feature_names_in_, fill_value=0.0)

    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]
    confidence = float(max(proba))

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba))
    }

def predict_clean_or_stego(image_path: str):
    """
    Predict whether an image is clean or stego by comparing it to itself (used with relative features).

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Contains prediction, confidence, and class probabilities.
    """
    print(f"[INFO] Analyzing image: {image_path}")
    trace = StegoTraceAnalyzer(image_path, image_path, DB_PATH)
    features = trace.run_all()

    clf = joblib.load(MODEL_PATH)
    df = pd.DataFrame([features]).reindex(columns=clf.feature_names_in_, fill_value=0.0)

    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]
    confidence = float(max(proba))

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba))
    }

def predict_single_clean_or_stego(image_path: str):
    """
    Predict whether a single image is clean or stego using absolute feature analysis.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Contains prediction, confidence, and class probabilities.
    """
    print(f"[INFO] Analyzing single image: {image_path}")
    extractor = StegoFeatureExtractor(image_path)
    features = extractor.extract_features()

    clf = joblib.load(SINGLE_MODEL_PATH)
    df = pd.DataFrame([features]).reindex(columns=clf.feature_names_in_, fill_value=0.0)

    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]
    confidence = float(max(proba))

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba))
    }

def debug_trace_features(image_path: str):
    """
    Run trace analysis on a single image and print each feature with debug markers.

    Args:
        image_path (str): Path to the image to be analyzed.

    Returns:
        dict: Extracted feature values.
    """
    print(f"[DEBUG] Running trace analysis for: {image_path}")
    trace = StegoTraceAnalyzer(image_path, image_path, DB_PATH)
    features = trace.run_all()

    print("\n[FEATURES]")
    for key, value in features.items():
        if isinstance(value, float):
            if abs(value) < 1e-6:
                print(f"  {key}: {value:.2e} ⚠️ near zero")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return features
