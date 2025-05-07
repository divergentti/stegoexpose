import joblib
import pandas as pd
from parse2database import StegoTraceAnalyzer

MODEL_PATH = "./training/models/rf_clean_vs_stego.pkl"
DB_PATH = "./training/database/steganalysis.db"



def predict_from_images(original_path: str, stego_path: str):
    trace = StegoTraceAnalyzer(original_path, stego_path, DB_PATH)
    features = trace.run_all()

    # Lataa malli
    clf = joblib.load(MODEL_PATH)

    # Muunna piirteet DataFrameksi ja varmista oikea järjestys
    df = pd.DataFrame([features])
    df = df.reindex(columns=clf.feature_names_in_, fill_value=0.0)

    # Ennustus
    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]
    confidence = float(max(proba))

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba))
    }



def predict_clean_or_stego(image_path: str):
    print(f"[INFO] Analyzing image: {image_path}")

    # Käytetään samaa kuvaa sekä orig että "stego", koska ei ole vertailukuvaa
    trace = StegoTraceAnalyzer(image_path, image_path, DB_PATH)
    features = trace.run_all()

    # Lataa malli
    clf = joblib.load(MODEL_PATH)

    # Valmistellaan ominaisuudet oikeaan järjestykseen
    df = pd.DataFrame([features])
    df = df.reindex(columns=clf.feature_names_in_, fill_value=0.0)

    # Ennustus
    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]
    confidence = float(max(proba))

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": dict(zip(clf.classes_, proba))
    }


def debug_trace_features(image_path: str):
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

    missing = set(trace.run_all().__annotations__.keys()) - set(features.keys())
    if missing:
        print(f"\n[WARNING] Missing features: {missing}")

    return features

debug_trace_features("./training/testimages/openstegografana.png")



"""

print("Predicting used tool, new images out of dataset compared to original image...\n")

print("Predicting Outguess encrypted file...")
result = predict_from_images(
    "./training/testimages/original2.jpg",
    "./training/testimages/outguess.jpg"
)
print("Tool predicted:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Full probabilities:", result["probabilities"])

print("Predicting Openstego encrypted file...")
result = predict_from_images(
    "./training/testimages/original.png",
    "./training/testimages/openstego.png"
)
print("Tool predicted:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Full probabilities:", result["probabilities"])

print("Predicting Steghide encrypted file...")

result = predict_from_images(
    "./training/testimages/original2.jpg",
    "./training/testimages/steghide.jpg"
)
print("Tool predicted:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Full probabilities:", result["probabilities"])


print("Predicting new images without originals ...\n")
print("Predict is clean or stego ... this should be clean")
result = predict_clean_or_stego("./training/testimages/grafana4.png")
print("Prediction:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])

print("Predict is clean or stego ... this should be OpenStego")
result = predict_clean_or_stego("./training/testimages/openstegografana.png")
print("Prediction:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])

print("Predict is clean or stego ... this should be OutGuess")
result = predict_clean_or_stego("./training/testimages/jopejaipeoutguess.jpg")
print("Prediction:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])


print("Predict is clean or stego ... this should be Steghide")
result = predict_clean_or_stego("./training/testimages/JopeKalassa.JPG")
print("Prediction:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])




print("Predict is clean or stego ... this is encrypted witj Jari's tool")
result = predict_clean_or_stego("./training/testimages/encrypted_kimalainen.jpg")
print("Prediction:", result["prediction"])
print("Confidence:", f"{result['confidence']:.2%}")
print("Probabilities:", result["probabilities"])

"""