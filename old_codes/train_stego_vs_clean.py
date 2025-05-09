import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DB_PATH = "./training/database/steganalysis.db"
MODEL_PATH = "./training/models/rf_clean_vs_stego.pkl"

def load_trace_dataframe(db_path: str) -> pd.DataFrame:
    """
    Load trace analysis results from the database, joining stego image metadata
    with extracted feature traces.

    Args:
        db_path (str): Path to the SQLite database.

    Returns:
        pd.DataFrame: Feature data joined with stego class labels.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            s.id AS stego_id,
            s.tool,
            s.filetype,
            s.shape_mismatch,
            t.*
        FROM stego_images s
        JOIN trace_results t ON s.id = t.stego_id
        WHERE t.kl_r IS NOT NULL
    """, conn)
    conn.close()

    df['shape_mismatch'] = df['shape_mismatch'].astype(int)
    df['stego_class'] = df['tool'].apply(lambda x: 'clean' if x is None else 'stego')

    return df

def main():
    """
    Main function for training a classifier to distinguish stego vs clean images
    based on statistical trace analysis features.
    """
    print("[INFO] Loading data...")
    df = load_trace_dataframe(DB_PATH)
    print(f"[INFO] Loaded {len(df)} records")

    drop_cols = ['stego_id', 'tool', 'filetype', 'stego_class']
    X = df.drop(columns=drop_cols)
    y = df['stego_class']

    print(f"[INFO] Classes: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("[RESULT] Classification report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=['clean', 'stego'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['clean', 'stego'], yticklabels=['clean', 'stego'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("[INFO] Top 10 features:\n", importances.head(10))
    importances.head(10).plot(kind='barh')
    plt.title("Top Feature Importances (clean vs stego)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    joblib.dump(clf, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
