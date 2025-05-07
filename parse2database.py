


"""
Gather information from images and pairings from metadata


"""


import os
import pandas as pd
import sqlite3
from PIL import Image, ExifTags
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.util import view_as_blocks  # pip install scikit-image
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.stats import chisquare


# Directories and file paths
ORIGINAL_DIR = "./training/originals/"
STEGO_OPENSTEGA_DIR = "./training/openstego/"
STEGO_STEGHIDE_DIR = "./training/steghide/"
STEGO_OUTGUESS_DIR = "./training/outguess/"
STEGO_DIRS = {
        "openstego": STEGO_OPENSTEGA_DIR,
        "steghide": STEGO_STEGHIDE_DIR,
        "outguess": STEGO_OUTGUESS_DIR
    }
SECRET_MESSAGE_FILE = "./training/embeds/testfile.txt"
PASSPHRASE = "set"
STEGO_METADATA_FILE = "./training/stego_metadata.csv"
ANALYSIS_DB_FILE = "./training/database/steganalysis.db"
VERIFY_MESSAGE_PATH = "./training/extracted/verify.txt"


debug_create_database = True
debug_analyzedatabase = True
debug_stegodecisiontreeanalyzer = True
debug_stegotraceanalyzer = True


class CreateDatabase:
    def __init__(self, original_dir: str, stego_dirs: str, metadata_file: str):
        self.original_dir = original_dir
        self.stego_dirs = stego_dirs
        self.metadata_file = metadata_file
        self.data = []
        self.check_metadata()

    def check_metadata(self):
        # Add stego images from metadata CSV
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        df = pd.read_csv(self.metadata_file)
        for _, row in df.iterrows():
            original_file = row['original']
            tool = row['tool']
            fname = row['outfile']
            if tool not in self.stego_dirs:
                if debug_create_database:
                    print(f"Unknown tool '{tool}' in metadata")
                continue
            full_stego_path = os.path.join(self.stego_dirs[tool], fname)
            full_orig_path = os.path.join(self.original_dir, original_file)

            if os.path.isfile(full_stego_path):
                self.data.append((full_orig_path, full_stego_path, tool))

            else:
                if debug_create_database:
                    print(f"Stego file not print (metadata)found: {full_stego_path}")


    def create_db(self):
        db_path = ANALYSIS_DB_FILE
        if os.path.exists(db_path):
            if debug_create_database:
                print("[INFO] Database already exists.")
            return

        if debug_create_database:
            print("[INFO] Creating database...")

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Create tables
        c.execute("""
            CREATE TABLE originals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filetype TEXT,
            filesize INTEGER,
            exif_exists BOOLEAN,
            exif_offset INTEGER
        )

        """)

        c.execute("""
            CREATE TABLE stego_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER,
                filename TEXT,
                tool TEXT,
                filetype TEXT,
                filesize INTEGER,
                verified BOOLEAN,
                diff_mean REAL,
                diff_max INTEGER,
                shape_mismatch BOOLEAN,
                FOREIGN KEY(original_id) REFERENCES originals(id)
            )          

        """)

        c.execute("""
            CREATE TABLE trace_results (
                stego_id INTEGER PRIMARY KEY,
                kl_r REAL,
                kl_g REAL,
                kl_b REAL,
                dct_b_mean REAL,
                dct_b_max REAL,
                dct_g_mean REAL,
                dct_g_max REAL,
                dct_r_mean REAL,
                dct_r_max REAL,
                entropy_diff REAL,
                variance_diff REAL,
                chi2_pvalue REAL,
                lsb_b_changed_pct REAL,
                lsb_g_changed_pct REAL,
                lsb_r_changed_pct REAL,
                lsb_avg_changed_pct REAL,
                FOREIGN KEY(stego_id) REFERENCES stego_images(id)
            )
        """)

        # Process each data tuple
        for orig_path, stego_path, tool in self.data:
            # --- Analyze original ---
            orig_stat = os.stat(orig_path)
            orig_size = orig_stat.st_size
            orig_type = os.path.splitext(orig_path)[1][1:].lower()

            exif_exists = False
            exif_offset = None
            try:
                img = Image.open(orig_path)
                exif = img._getexif()
                if exif:
                    exif_exists = True
                    # Estimate EXIF offset (not always reliable)
                    with open(orig_path, 'rb') as f:
                        data = f.read()
                        offset = data.find(b'Exif')
                        if offset != -1:
                            exif_offset = offset
            except Exception:
                pass

            # Insert original
            c.execute("""
                INSERT INTO originals (filename, filetype, filesize, exif_exists, exif_offset)
                VALUES (?, ?, ?, ?, ?)
            """, (orig_path, orig_type, orig_size, exif_exists, exif_offset))

            original_id = c.lastrowid

            # --- Analyze stego image ---
            stego_stat = os.stat(stego_path)
            stego_size = stego_stat.st_size
            stego_type = os.path.splitext(stego_path)[1][1:].lower()

            # Placeholder diff metrics
            diff_mean = None
            diff_max = None

            # Placeholder verification flag
            verified = None

            # Insert stego
            c.execute("""
                INSERT INTO stego_images (original_id, filename, tool, filetype, filesize, verified, diff_mean, diff_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (original_id, stego_path, tool, stego_type, stego_size, verified, diff_mean, diff_max))

        conn.commit()
        conn.close()
        if debug_create_database:
            print("[INFO] Database created.")


class AnalyzeDatabase:

    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")


    def compute_diff_metrics(self):
        if debug_analyzedatabase:
            print("Starting analyze ... wait ...")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        shape_mismatch = False

        c.execute("""
            SELECT s.id, o.filename, s.filename
            FROM stego_images s
            JOIN originals o ON s.original_id = o.id
            WHERE s.diff_mean IS NULL OR s.diff_max IS NULL
        """)
        rows = c.fetchall()

        for sid, orig_path, stego_path in rows:
            try:
                orig = cv2.imread(orig_path)
                stego = cv2.imread(stego_path)

                if orig is None or stego is None or orig.shape != stego.shape:
                    shape_mismatch = True
                    if debug_analyzedatabase:
                        print(f"[NOTE] Shape mismatch: {orig_path} vs {stego_path}")
                    diff_mean = None
                    diff_max = None
                else:
                    diff = cv2.absdiff(orig, stego)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    diff_mean = float(np.mean(diff_gray))
                    diff_max = int(np.max(diff_gray))

                c.execute("""
                    UPDATE stego_images
                    SET diff_mean = ?, diff_max = ?, shape_mismatch = ?
                    WHERE id = ?
                """, (diff_mean, diff_max, shape_mismatch, sid))

            except Exception as e:
                print(f"[ERROR] {orig_path} vs {stego_path}: {e}")

        conn.commit()
        conn.close()
        if debug_analyzedatabase:
            print("[INFO] Diff metrics computed and saved.")

class StegoDecisionTreeAnalyzer:
    def __init__(self, db_path: str, max_depth: int = 3):
        self.db_path = db_path
        self.max_depth = max_depth
        self.model = None
        self.features = ['diff_mean', 'diff_max', 'shape_mismatch']
        self.df = None

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT tool, diff_mean, diff_max, shape_mismatch
            FROM stego_images
            WHERE diff_mean IS NOT NULL AND tool IS NOT NULL
        """
        self.df = pd.read_sql_query(query, conn)
        conn.close()

        self.df['shape_mismatch'] = self.df['shape_mismatch'].astype(int)

    def train_tree(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X = self.df[self.features]
        y = self.df['tool']

        clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        clf.fit(X, y)
        self.model = clf

    def visualize_tree(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_tree() first.")

        plt.figure(figsize=(12, 6))
        plot_tree(self.model,
                  feature_names=self.features,
                  class_names=self.model.classes_,
                  filled=True,
                  rounded=True)
        plt.title("Decision Tree: Tool Prediction")
        plt.tight_layout()
        plt.show()

    def run(self):
        if debug_stegodecisiontreeanalyzer:
            print("[INFO] Loading data...")
        self.load_data()
        if debug_stegodecisiontreeanalyzer:
            print("[INFO] Training decision tree...")
        self.train_tree()
        if debug_stegodecisiontreeanalyzer:
            print("[INFO] Visualizing tree...")
        self.visualize_tree()

class StegoTraceAnalyzer:
    def __init__(self, original_path: str, stego_path: str, db_path: str):
        self.original_path = original_path
        self.stego_path = stego_path
        self.db_path = db_path

        self.original = cv2.imread(self.original_path, cv2.IMREAD_COLOR)
        self.stego = cv2.imread(self.stego_path, cv2.IMREAD_COLOR)

        if self.original is None or self.stego is None:
            raise ValueError("One or both images could not be loaded.")

        if self.original.shape != self.stego.shape:
            raise ValueError("Image dimensions do not match.")

        self.diff = cv2.absdiff(self.original, self.stego)

    @staticmethod
    def from_database(db_path: str, stego_id: int):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("""
            SELECT o.filename, s.filename
            FROM stego_images s
            JOIN originals o ON s.original_id = o.id
            WHERE s.id = ?
        """, (stego_id,))
        row = c.fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"No image pair found for stego id {stego_id}.")

        orig_path, stego_path = row
        return StegoTraceAnalyzer(orig_path, stego_path, db_path)

    @staticmethod
    def random_pair(db_path: str):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT s.id, o.filename, s.filename
            FROM stego_images s
            JOIN originals o ON s.original_id = o.id
            LIMIT 1
        """, conn)
        conn.close()

        sid, orig_path, stego_path = df.iloc[0]
        if debug_stegotraceanalyzer:
            print(f"[INFO] Using pair from DB (stego id={sid}):\n  {orig_path}\n  {stego_path}")
        return StegoTraceAnalyzer(orig_path, stego_path)

    def histogram_kl_divergence(self, bins=256):
        kl_divs = []
        for ch in range(3):  # R, G, B
            orig_hist, _ = np.histogram(self.original[:, :, ch], bins=bins, range=(0, 256), density=True)
            stego_hist, _ = np.histogram(self.stego[:, :, ch], bins=bins, range=(0, 256), density=True)

            # Vältä 0 log 0 virheet
            orig_hist += 1e-10
            stego_hist += 1e-10

            kl = entropy(orig_hist, stego_hist)
            kl_divs.append(kl)

        return {
            "kl_r": kl_divs[2],  # OpenCV: BGR
            "kl_g": kl_divs[1],
            "kl_b": kl_divs[0]
        }

    def plot_histograms(self):
        color = ('b', 'g', 'r')
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(color):
            orig_hist = cv2.calcHist([self.original], [i], None, [256], [0, 256])
            stego_hist = cv2.calcHist([self.stego], [i], None, [256], [0, 256])

            plt.subplot(1, 3, i + 1)
            plt.plot(orig_hist, color=col, label='Original')
            plt.plot(stego_hist, color='k', linestyle='dashed', label='Stego')
            plt.title(f'Channel {col.upper()}')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def dct_difference_metrics(self, block_size=8):
        """Laskee DCT-koeffisienttien keskimääräisen erotuksen jokaisessa väri-kanavassa."""

        def blockwise_dct_diff(orig_ch, stego_ch):
            h, w = orig_ch.shape
            h = h - h % block_size
            w = w - w % block_size

            orig_ch = orig_ch[:h, :w]
            stego_ch = stego_ch[:h, :w]

            orig_blocks = view_as_blocks(orig_ch, (block_size, block_size)).reshape(-1, block_size, block_size)
            stego_blocks = view_as_blocks(stego_ch, (block_size, block_size)).reshape(-1, block_size, block_size)

            diffs = []
            for ob, sb in zip(orig_blocks, stego_blocks):
                dct_o = scipy.fftpack.dct(scipy.fftpack.dct(ob.T, norm='ortho').T, norm='ortho')
                dct_s = scipy.fftpack.dct(scipy.fftpack.dct(sb.T, norm='ortho').T, norm='ortho')
                diffs.append(np.abs(dct_o - dct_s))

            return np.mean(diffs, axis=0)

        results = {}
        channels = ['b', 'g', 'r']
        for i, ch in enumerate(channels):
            orig_ch = self.original[:, :, i].astype(np.float32)
            stego_ch = self.stego[:, :, i].astype(np.float32)
            avg_dct_diff = blockwise_dct_diff(orig_ch, stego_ch)
            results[f'dct_{ch}_mean'] = float(np.mean(avg_dct_diff))
            results[f'dct_{ch}_max'] = float(np.max(avg_dct_diff))

        return results

    def statistical_tests(self):
        """Laskee tilastollisia eroja alkuperäisen ja stegakuvan välillä."""
        results = {}

        # Entropia: keskiarvo RGB-kanavien entropioista
        def channel_entropy(channel):
            hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
            hist += 1e-10  # Vältä log(0)
            return entropy(hist)

        orig_entropy = np.mean([channel_entropy(self.original[:, :, i]) for i in range(3)])
        stego_entropy = np.mean([channel_entropy(self.stego[:, :, i]) for i in range(3)])
        results['entropy_diff'] = stego_entropy - orig_entropy

        # Varianssi: keskiarvo RGB-kanavien varianssieroista
        orig_var = np.var(self.original, axis=(0, 1))
        stego_var = np.var(self.stego, axis=(0, 1))
        results['variance_diff'] = float(np.mean(stego_var - orig_var))

        # Chi²-testi: kokonaishistogrammien ero RGB-kanavista yhdistettynä
        orig_flat = self.original.ravel()
        stego_flat = self.stego.ravel()
        orig_hist, _ = np.histogram(orig_flat, bins=256, range=(0, 256))
        stego_hist, _ = np.histogram(stego_flat, bins=256, range=(0, 256))

        # Vältä nollat
        orig_hist = orig_hist + 1
        stego_hist = stego_hist + 1

        chi2_stat, p_value = chisquare(f_obs=stego_hist, f_exp=orig_hist)
        results['chi2_pvalue'] = float(p_value)

        return results

    def lsb_difference(self):
        """Laskee LSB-tasolla tapahtuneet muutokset ja palauttaa prosenttiosuuden muuttuneista biteistä per kanava."""
        results = {}
        total_pixels = self.original.shape[0] * self.original.shape[1]

        changes = []
        for i, ch in enumerate(['b', 'g', 'r']):
            orig_lsb = self.original[:, :, i] & 1
            stego_lsb = self.stego[:, :, i] & 1
            xor_lsb = orig_lsb ^ stego_lsb

            changed = np.count_nonzero(xor_lsb)
            percent_changed = changed / total_pixels * 100

            results[f'lsb_{ch}_changed_pct'] = percent_changed
            changes.append(percent_changed)

        results['lsb_avg_changed_pct'] = np.mean(changes)
        return results


    def save_trace_results_to_db(self, stego_id, trace_results: dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        columns = ['stego_id'] + list(trace_results.keys())
        values = [stego_id] + [trace_results[k] for k in trace_results]

        placeholders = ','.join(['?'] * len(columns))
        sql = f"""
            INSERT OR REPLACE INTO trace_results ({','.join(columns)})
            VALUES ({placeholders})
        """
        c.execute(sql, values)
        conn.commit()
        conn.close()

    def run_all(self):
        """Suorittaa kaikki analyysit ja palauttaa yhdistetyt tulokset."""
        results = {}
        try:
            results.update(self.histogram_kl_divergence())
        except Exception as e:
            results['kl_error'] = str(e)

        try:
            results.update(self.dct_difference_metrics())
        except Exception as e:
            results['dct_error'] = str(e)

        try:
            results.update(self.statistical_tests())
        except Exception as e:
            results['stats_error'] = str(e)

        try:
            results.update(self.lsb_difference())
        except Exception as e:
            results['lsb_error'] = str(e)

        return results

    @staticmethod
    def run_trace_analysis_for_clean_images(db_path: str):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Hae kaikki original-kuvat, joita ei ole käytetty stego-kuvissa
        c.execute("""
            SELECT id, filename
            FROM originals
            WHERE id NOT IN (SELECT DISTINCT original_id FROM stego_images)
        """)
        clean_images = c.fetchall()
        conn.close()

        print(f"[INFO] Found {len(clean_images)} clean images to analyze.")

        for orig_id, orig_path in clean_images:
            try:
                if debug_stegotraceanalyzer:
                    print(f"[TRACE] Processing clean original_id = {orig_id}")
                # Käytetään samaa polkua sekä orig että stego – koska meillä ei ole vertailukuvaa
                trace = StegoTraceAnalyzer(orig_path, orig_path, db_path)
                results = trace.run_all()

                # Tallennetaan 'clean' tieto stego_images-tauluun ilman stegokuvaa
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                c.execute("""
                    INSERT INTO stego_images (original_id, filename, tool, filetype, shape_mismatch)
                    VALUES (?, ?, NULL, ?, 0)
                """, (orig_id, orig_path, os.path.splitext(orig_path)[1].lstrip('.').lower()))
                stego_id = c.lastrowid
                conn.commit()
                conn.close()

                trace.save_trace_results_to_db(stego_id, results)

            except Exception as e:
                print(f"[ERROR] Failed to analyze clean original_id {orig_id}: {e}")

    @staticmethod
    def run_trace_analysis_for_all(db_path: str):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM stego_images")
        stego_ids = [row[0] for row in c.fetchall()]
        conn.close()

        for sid in stego_ids:
            try:
                if debug_stegotraceanalyzer:
                    print(f"[TRACE] Processing stego_id = {sid} ... wait ... slow ...")
                trace = StegoTraceAnalyzer.from_database(db_path, stego_id=sid)
                summary = trace.run_all()
                trace.save_trace_results_to_db(sid, summary)
            except Exception as e:
                print(f"[ERROR] Failed to analyze stego_id {sid}: {e}")


def main():
    dbinstaller = CreateDatabase(ORIGINAL_DIR, STEGO_DIRS, STEGO_METADATA_FILE)
    dbinstaller.create_db()

    analyzer = AnalyzeDatabase(ANALYSIS_DB_FILE)
    analyzer.compute_diff_metrics()

    tree_analyzer = StegoDecisionTreeAnalyzer(ANALYSIS_DB_FILE)
    tree_analyzer.run()

    # ---- tests ---

    """
    trace = StegoTraceAnalyzer.from_database(ANALYSIS_DB_FILE, stego_id=3)
    kl = trace.histogram_kl_divergence()
    print("KL:", kl)
    trace.plot_histograms()

    trace = StegoTraceAnalyzer.from_database(ANALYSIS_DB_FILE, stego_id=1)
    dct = trace.dct_difference_metrics()
    print("DCT differences:", dct)


    trace = StegoTraceAnalyzer.from_database(ANALYSIS_DB_FILE, stego_id=2)
    stats = trace.statistical_tests()
    print("Statistical results:", stats)


    trace = StegoTraceAnalyzer.from_database(ANALYSIS_DB_FILE, stego_id=2)
    lsb = trace.lsb_difference()
    print("LSB diff %:", lsb)

    trace = StegoTraceAnalyzer.from_database(ANALYSIS_DB_FILE, stego_id=2)
    summary = trace.run_all()
    

    for k, v in summary.items():
        print(f"{k}: {v}")
                """

    # StegoTraceAnalyzer.run_trace_analysis_for_all(ANALYSIS_DB_FILE)
    StegoTraceAnalyzer.run_trace_analysis_for_clean_images(ANALYSIS_DB_FILE)


if __name__ == "__main__":
    main()
