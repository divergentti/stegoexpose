



import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DB_PATH = "./training/database/steganalysis.db"

def load_trace_dataframe(db_path: str) -> pd.DataFrame:
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
    return df

def main():
    print("[INFO] Loading data...")
    df = load_trace_dataframe(DB_PATH)

    # Poistetaan tunnisteet ja ei-numeeriset sarakkeet pois piirteistä
    drop_cols = ['stego_id', 'tool', 'filetype']
    X = df.drop(columns=drop_cols)
    y = df['tool']  # Voit vaihtaa tähän 'filetype' jos haluat

    print(f"[INFO] Data shape: {X.shape}, Labels: {y.nunique()} classes")

    # Jaa opetus- ja testijoukkoihin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Malli
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Ennustukset ja tulokset
    y_pred = clf.predict(X_test)
    print("[RESULT] Classification report:\n", classification_report(y_test, y_pred))
    print("[RESULT] Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Ominaisuuksien tärkeys
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("[INFO] Top 10 features:\n", importances.head(10))
    importances.head(10).plot(kind='barh')
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    joblib.dump(clf, "./training/models/rf_stegomodel.pkl")
    print("[INFO] Model saved to ./training/models/rf_stegomodel.pkl")

if __name__ == "__main__":
    main()
