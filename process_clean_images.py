import os
import sqlite3
from parse2database import StegoTraceAnalyzer

DB_PATH = "./training/database/steganalysis.db"
CLEAN_FOLDER = "./training/clean/"

def insert_clean_originals(db_path: str, folder: str):
    """
    Insert clean images from a folder into the 'originals' table of the database.
    Avoids duplicates by checking for existing filenames.

    Args:
        db_path (str): Path to the SQLite database.
        folder (str): Path to the folder containing clean images.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    added = 0
    for fname in files:
        full_path = os.path.join(folder, fname)
        c.execute("SELECT 1 FROM originals WHERE filename = ?", (full_path,))
        if c.fetchone() is None:
            c.execute("INSERT INTO originals (filename) VALUES (?)", (full_path,))
            added += 1
    conn.commit()
    conn.close()
    print(f"[INFO] Inserted {added} clean originals into database.")

def analyze_clean_images(db_path: str):
    """
    Run statistical trace analysis on all clean images inserted into the database.

    Args:
        db_path (str): Path to the SQLite database.
    """
    StegoTraceAnalyzer.run_trace_analysis_for_clean_images(db_path)

def main():
    """
    Entry point for inserting and analyzing clean images for trace model training.
    """
    insert_clean_originals(DB_PATH, CLEAN_FOLDER)
    analyze_clean_images(DB_PATH)

if __name__ == "__main__":
    main()
