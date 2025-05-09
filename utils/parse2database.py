import os
import pandas as pd
import sqlite3
from PIL import Image
import numpy as np
import cv2
from scipy.stats import entropy, chisquare
from skimage.util import view_as_blocks
import scipy.fftpack
from multiprocessing import Pool
import pywt
from multiprocessing import Value, Lock
from utils import settings

_counter = Value('i', 0)
_counter_lock = Lock()

STEGO_DIRS = {
    "openstego": settings.OPENSTEGO_DIR,
    "steghide": settings.STEGHIDE_DIR,
    "outguess": settings.OUTGUESS_DIR
}


# Debug flags
debug_create_database = True
debug_analyzedatabase = True
debug_stegotraceanalyzer = True


def process_stego_id_global(sid):
    try:
        trace = StegoTraceAnalyzer.from_database(settings.DATABASE_PATH, stego_id=sid)
        summary = trace.run_all()
        trace.save_trace_results_to_db(sid, summary)
    except Exception as e:
        print(f"[ERROR] Failed to analyze stego_id {sid}: {e}")
    finally:
        with _counter_lock:
            _counter.value += 1
            if _counter.value % 10 == 0:
                print(f"[INFO] Processed { _counter.value } stego images...")

def process_clean_image_global(orig_id_filename):
    orig_id, orig_path = orig_id_filename
    try:
        if debug_stegotraceanalyzer:
            print(f"[TRACE] Processing clean original_id = {orig_id}")
        trace = StegoTraceAnalyzer(orig_path, orig_path, settings.DATABASE_PATH)
        results = trace.run_all()
        results.update({
            "lsb_b_changed_pct": 0.0, "lsb_g_changed_pct": 0.0,
            "lsb_r_changed_pct": 0.0, "lsb_avg_changed_pct": 0.0,
            "kl_r": 0.0, "kl_g": 0.0, "kl_b": 0.0,
            "dct_b_mean": 0.0, "dct_b_max": 0.0,
            "dct_g_mean": 0.0, "dct_g_max": 0.0,
            "dct_r_mean": 0.0, "dct_r_max": 0.0
        })

        conn = sqlite3.connect(settings.DATABASE_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO stego_images (original_id, filename, tool, filetype, shape_mismatch, is_clean)
            VALUES (?, ?, NULL, ?, 0, 1)
        """, (orig_id, orig_path, os.path.splitext(orig_path)[1].lstrip('.').lower()))
        stego_id = c.lastrowid
        conn.commit()
        conn.close()

        trace.save_trace_results_to_db(stego_id, results)
    except Exception as e:
        print(f"[ERROR] Failed to analyze clean original_id {orig_id}: {e}")
    finally:
        with _counter_lock:
            _counter.value += 1
            if _counter.value % 10 == 0:
                print(f"[INFO] Processed { _counter.value } clean images...")



class CreateDatabase:
    def __init__(self, original_dir: str, stego_dirs: dict, metadata_file: str):
        self.original_dir = original_dir
        self.stego_dirs = stego_dirs
        self.metadata_file = metadata_file
        self.data = []
        self.check_metadata()

    def check_metadata(self):
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")

        df = pd.read_csv(self.metadata_file)
        for _, row in df.iterrows():
            original_file = row['original']
            tool = row['tool']
            fname = row['outfile']
            rate = row.get('rate', 'unknown')
            if tool not in self.stego_dirs:
                if debug_create_database:
                    print(f"Unknown tool '{tool}' in metadata")
                continue
            full_stego_path = os.path.join(self.stego_dirs[tool], fname)
            full_orig_path = os.path.join(self.original_dir, original_file)

            if os.path.isfile(full_stego_path):
                self.data.append((full_orig_path, full_stego_path, tool, rate))

    def create_db(self):
        db_path = settings.DATABASE_PATH
        if os.path.exists(db_path):
            if debug_create_database:
                print("[INFO] Database already exists.")
            return

        if debug_create_database:
            print("[INFO] Creating database...")

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

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
                is_clean BOOLEAN DEFAULT 0,
                rate TEXT,
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
                wavelet_diff_mean REAL,
                wavelet_diff_max REAL,
                srm_mean REAL,
                srm_max REAL,
                FOREIGN KEY(stego_id) REFERENCES stego_images(id)
            )
        """)

        for orig_path, stego_path, tool, rate in self.data:
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
                    with open(orig_path, 'rb') as f:
                        data = f.read()
                        offset = data.find(b'Exif')
                        if offset != -1:
                            exif_offset = offset
            except Exception:
                pass

            c.execute("""
                INSERT INTO originals (filename, filetype, filesize, exif_exists, exif_offset)
                VALUES (?, ?, ?, ?, ?)
            """, (orig_path, orig_type, orig_size, exif_exists, exif_offset))

            original_id = c.lastrowid

            stego_stat = os.stat(stego_path)
            stego_size = stego_stat.st_size
            stego_type = os.path.splitext(stego_path)[1][1].lower()

            diff_mean = None
            diff_max = None
            verified = None

            c.execute("""
                INSERT INTO stego_images (original_id, filename, tool, filetype, filesize, verified, diff_mean, diff_max, rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (original_id, stego_path, tool, stego_type, stego_size, verified, diff_mean, diff_max, rate))

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
                orig = cv2.resize(orig, (256, 256))
                stego = cv2.resize(stego, (256, 256))

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


def process_stego_id(sid):
    try:
        trace = StegoTraceAnalyzer.from_database(settings.DATABASE_PATH, stego_id=sid)
        summary = trace.run_all()
        trace.save_trace_results_to_db(sid, summary)
    except Exception as e:
        print(f"[ERROR] Failed to analyze stego_id {sid}: {e}")

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

    def histogram_kl_divergence(self, bins=256):
        kl_divs = []
        for ch in range(3):
            orig_hist, _ = np.histogram(self.original[:, :, ch], bins=bins, range=(0, 256), density=True)
            stego_hist, _ = np.histogram(self.stego[:, :, ch], bins=bins, range=(0, 256), density=True)
            orig_hist += 1e-10
            stego_hist += 1e-10
            kl = entropy(orig_hist, stego_hist)
            kl_divs.append(kl)
        return {
            "kl_r": kl_divs[2],
            "kl_g": kl_divs[1],
            "kl_b": kl_divs[0]
        }

    def dct_difference_metrics(self, block_size=8):
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
        results = {}
        def channel_entropy(channel):
            hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
            hist += 1e-10
            return entropy(hist)

        orig_entropy = np.mean([channel_entropy(self.original[:, :, i]) for i in range(3)])
        stego_entropy = np.mean([channel_entropy(self.stego[:, :, i]) for i in range(3)])
        results['entropy_diff'] = stego_entropy - orig_entropy

        orig_var = np.var(self.original, axis=(0, 1))
        stego_var = np.var(self.stego, axis=(0, 1))
        results['variance_diff'] = float(np.mean(stego_var - orig_var))

        orig_flat = self.original.ravel()
        stego_flat = self.stego.ravel()
        orig_hist, _ = np.histogram(orig_flat, bins=256, range=(0, 256))
        stego_hist, _ = np.histogram(stego_flat, bins=256, range=(0, 256))
        orig_hist = orig_hist + 1
        stego_hist = stego_hist + 1
        chi2_stat, p_value = chisquare(f_obs=stego_hist, f_exp=orig_hist)
        results['chi2_pvalue'] = float(p_value)
        return results

    def lsb_difference(self):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        total_pixels = self.original.shape[0] * self.original.shape[1]

        for i, ch in enumerate(['b', 'g', 'r']):
            orig_ch = torch.from_numpy(self.original[:, :, i]).to(device)
            stego_ch = torch.from_numpy(self.stego[:, :, i]).to(device)

            orig_lsb = orig_ch & 1
            stego_lsb = stego_ch & 1
            diff = (orig_lsb ^ stego_lsb).float()
            changed_pct = diff.sum().item() / total_pixels * 100

            results.append(changed_pct)

        return {
            'lsb_b_changed_pct': results[0],
            'lsb_g_changed_pct': results[1],
            'lsb_r_changed_pct': results[2],
            'lsb_avg_changed_pct': sum(results) / 3
        }

    def srm_features(self):
        import torch
        import torch.nn as nn

        class SRMFilter(nn.Module):
            def __init__(self):
                super(SRMFilter, self).__init__()
                srm_kernel = torch.tensor([[
                    [[-1, 2, -2, 2, -1],
                     [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2],
                     [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]]
                ]], dtype=torch.float32)
                self.conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
                self.conv.weight = nn.Parameter(srm_kernel, requires_grad=False)

            def forward(self, x):
                return self.conv(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        srm = SRMFilter().to(device)
        srm.eval()

        srm_outs = []
        for i in range(3):  # B, G, R
            ch = self.stego[:, :, i]
            ch_tensor = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                filtered = srm(ch_tensor).cpu().numpy().flatten()
                srm_outs.append(filtered)

        all_srm = np.concatenate(srm_outs)
        return {
            "srm_mean": float(np.mean(all_srm)),
            "srm_max": float(np.max(all_srm))
        }

    def wavelet_features(self):
        coeffs_orig = pywt.wavedec2(self.original, 'db1', level=1)
        coeffs_stego = pywt.wavedec2(self.stego, 'db1', level=1)
        diff = [np.mean(np.abs(co[0] - cs[0])) for co, cs in zip(coeffs_orig, coeffs_stego)]
        return {"wavelet_diff_mean": float(np.mean(diff)), "wavelet_diff_max": float(np.max(diff))}

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
        results = {}
        try:
            results.update(self.histogram_kl_divergence())
        except Exception as e:
            print(f"[ERROR] KL: {e}")
        try:
            results.update(self.dct_difference_metrics())
        except Exception as e:
            print(f"[ERROR] DCT: {e}")
        try:
            results.update(self.statistical_tests())
        except Exception as e:
            print(f"[ERROR] STATS: {e}")
        try:
            results.update(self.lsb_difference())
        except Exception as e:
            print(f"[ERROR] LSB: {e}")
        try:
            results.update(self.srm_features())
        except Exception as e:
            print(f"[ERROR] SRM: {e}")
        try:
            results.update(self.wavelet_features())
        except Exception as e:
            print(f"[ERROR] WAVELET: {e}")
        return results

    @staticmethod
    def run_trace_analysis_for_all(db_path: str):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM stego_images")
        stego_ids = [row[0] for row in c.fetchall()]
        conn.close()
        with Pool() as pool:
            pool.map(process_stego_id, stego_ids)

        print(f"[INFO] Running stego analysis for {len(stego_ids)} images...")
        with Pool() as pool:
            pool.map(process_stego_id_global, stego_ids)

    @staticmethod
    def run_trace_analysis_for_clean_images(db_path: str):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT id, filename FROM originals")
        clean_images = c.fetchall()
        conn.close()

        print(f"[INFO] Running clean image trace analysis for {len(clean_images)} images...")
        with Pool() as pool:
            pool.map(process_clean_image_global, clean_images)


def main():
    dbinstaller = CreateDatabase(settings.ORIGINALS_DIR, STEGO_DIRS, settings.METADATA_PATH)
    dbinstaller.create_db()

    analyzer = AnalyzeDatabase(settings.DATABASE_PATH)
    analyzer.compute_diff_metrics()

    # Comment out StegoDecisionTreeAnalyzer for now
    # tree_analyzer = StegoDecisionTreeAnalyzer(settings.DATABASE_PATH)
    # tree_analyzer.run()

    StegoTraceAnalyzer.run_trace_analysis_for_all(settings.DATABASE_PATH)
    # StegoTraceAnalyzer.run_trace_analysis_for_clean_images(settings.DATABASE_PATH)

if __name__ == "__main__":
    main()