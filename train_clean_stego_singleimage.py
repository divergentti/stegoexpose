import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from stegofeatureextractor import StegoFeatureExtractor

DB_PATH = "./training/database/steganalysis.db"
MODEL_PATH = "./training/models/rf_single_clean_vs_stego.pkl"

def load_image_list(db_path: str):
    """
    Load image metadata from the SQLite database and assign stego/clean labels.

    Args:
        db_path (str): Path to the database.

    Returns:
        pd.DataFrame: DataFrame with image paths and stego_class labels.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, filename, tool
        FROM stego_images
        WHERE filename IS NOT NULL
    """, conn)
    conn.close()

    df['stego_class'] = df['tool'].apply(lambda x: 'clean' if x is None else 'stego')
    return df

def extract_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for a batch of images using StegoFeatureExtractor.

    Args:
        df (pd.DataFrame): Input DataFrame containing image paths and labels.

    Returns:
        pd.DataFrame: Feature vectors with stego class and filename.
    """
    feature_rows = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            extractor = StegoFeatureExtractor(row['filename'])
            features = extractor.extract_features()
            features['stego_class'] = row['stego_class']
            features['filename'] = row['filename']
            feature_rows.append(features)
        except Exception as e:
            print(f"[WARN] Skipping {row['filename']}: {e}")
            skipped += 1

    print(f"[INFO] Extracted features for {len(feature_rows)} images, skipped {skipped}.")
    return pd.DataFrame(feature_rows)

def main():
    """
    Main function to train a clean vs stego classifier using image features.
    Saves a trained RandomForest model for use in single-image predictions.
    """
    print("[INFO] Loading image list...")
    df_files = load_image_list(DB_PATH)

    print("[INFO] Extracting features...")
    df = extract_features_batch(df_files)

    drop_cols = ['filename', 'stego_class']
    X = df.drop(columns=drop_cols)
    y = df['stego_class']

    print(f"[INFO] Class balance:\n{y.value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("[RESULT] Classification report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=['clean', 'stego'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['clean', 'stego'], yticklabels=['clean', 'stego'])
    plt.title("Confusion Matrix (Clean vs Stego)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("[INFO] Top Features:\n", importances.head(10))
    importances.head(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
