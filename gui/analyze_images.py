from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QComboBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtCore import Qt
import sqlite3
import cv2
import numpy as np
import os
from utils import settings


class AnalyzeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyze Stego Images")
        self.setGeometry(200, 200, 1000, 600)

        self.layout = QVBoxLayout()
        self.original_combo = QComboBox()
        self.original_combo.currentIndexChanged.connect(self.load_tools_for_original)
        self.layout.addWidget(QLabel("Select Original Image:"))
        self.layout.addWidget(self.original_combo)

        self.tool_combo = QComboBox()
        self.tool_combo.currentIndexChanged.connect(self.load_versions_for_tool)
        self.layout.addWidget(QLabel("Select Stego Tool:"))
        self.layout.addWidget(self.tool_combo)

        self.version_combo = QComboBox()
        self.version_combo.currentIndexChanged.connect(self.display_analysis)
        self.layout.addWidget(QLabel("Select Message Length:"))
        self.layout.addWidget(self.version_combo)

        self.images_layout = QHBoxLayout()
        self.label_orig = QLabel("ORIGINAL")
        self.label_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_orig.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.label_orig.setStyleSheet("color: red;")
        self.label_stego = QLabel("STEGA")
        self.label_stego.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_stego.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.label_stego.setStyleSheet("color: blue;")
        self.orig_view = QGraphicsView()
        self.stego_info = QLabel("Stego Analysis:")
        self.stego_info.setWordWrap(True)
        self.orig_info = QLabel("Original Image:")
        self.orig_info.setWordWrap(True)

        self.stego_view = QGraphicsView()
        self.orig_scene = QGraphicsScene()
        self.stego_scene = QGraphicsScene()
        self.orig_view.setScene(self.orig_scene)
        self.stego_view.setScene(self.stego_scene)
        self.orig_view.setMinimumSize(256, 256)
        self.stego_view.setMinimumSize(256, 256)
        self.image_grid = QGridLayout()
        self.image_grid.addWidget(self.orig_view, 1, 0)
        self.image_grid.addWidget(self.stego_view, 1, 1)
        self.image_grid.addWidget(self.label_orig, 0, 0)
        self.image_grid.addWidget(self.label_stego, 0, 1)
        self.image_grid.addWidget(self.orig_view, 1, 0)
        self.image_grid.addWidget(self.stego_view, 1, 1)
        self.image_grid.addWidget(self.orig_info, 2, 0)
        self.image_grid.addWidget(self.stego_info, 2, 1)
        self.layout.addLayout(self.image_grid)

        self.details = QLabel("Select a stego variant to view analysis.")
        self.details.setWordWrap(True)
        self.layout.addWidget(self.details)

        self.lsb_button = QPushButton("Show LSB Diff Heatmap")
        self.lsb_button.clicked.connect(self.show_lsb_diff)
        self.layout.addWidget(self.lsb_button)

        self.srm_button = QPushButton("Show SRM Heatmap (placeholder)")
        self.srm_button.clicked.connect(lambda: self.details.setText(self.details.text() + "\n[SRM view coming soon]"))
        self.layout.addWidget(self.srm_button)

        self.dct_button = QPushButton("Show DCT Difference (placeholder)")
        self.dct_button.clicked.connect(lambda: self.details.setText(self.details.text() + "\n[DCT view coming soon]"))
        self.layout.addWidget(self.dct_button)

        self.setLayout(self.layout)
        self.db_path = settings.DATABASE_PATH
        self.filename_id_map = {}
        self.load_originals()

    def load_originals(self):
        self.original_combo.clear()
        self.filename_id_map = {}
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT DISTINCT o.filename, s.original_id
            FROM stego_images s
            JOIN originals o ON s.original_id = o.id
            WHERE s.is_clean = 0
        """)
        for filename, oid in c.fetchall():
            self.filename_id_map.setdefault(filename, []).append(oid)
        conn.close()
        for filename in sorted(self.filename_id_map.keys()):
            self.original_combo.addItem(os.path.basename(filename), filename)

    def load_tools_for_original(self):
        self.tool_combo.clear()
        self.version_combo.clear()
        self.orig_scene.clear()
        self.stego_scene.clear()
        filename = self.original_combo.currentData()
        if not filename:
            return
        id_list = self.filename_id_map[filename]
        placeholders = ",".join("?" * len(id_list))
        query = f"SELECT DISTINCT tool FROM stego_images WHERE original_id IN ({placeholders})"
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(query, id_list)
        tools = [row[0] for row in c.fetchall() if row[0]]
        conn.close()
        self.tool_combo.addItems(tools)

    def load_versions_for_tool(self):
        self.version_combo.clear()
        self.orig_scene.clear()
        self.stego_scene.clear()
        filename = self.original_combo.currentData()
        tool = self.tool_combo.currentText()
        if not filename or not tool:
            return
        id_list = self.filename_id_map[filename]
        placeholders = ",".join("?" * len(id_list))
        query = f"SELECT id, filename, rate FROM stego_images WHERE tool = ? AND original_id IN ({placeholders})"
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(query, [tool] + id_list)
        self.stegos = c.fetchall()
        for sid, path, rate in self.stegos:
            label = f"{rate if rate else 'unknown'}: {os.path.basename(path)}"
            self.version_combo.addItem(label, sid)
        conn.close()

    def display_analysis(self):
        self.orig_scene.clear()
        self.stego_scene.clear()
        idx = self.version_combo.currentIndex()
        if idx < 0:
            return

        sid = self.version_combo.currentData()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT o.filename, o.filesize, o.filetype,
                   s.filename, s.filesize, s.filetype,
                   s.tool, s.diff_mean, s.diff_max, s.shape_mismatch,
                   t.kl_r, t.kl_g, t.kl_b, t.lsb_avg_changed_pct, t.srm_mean, t.wavelet_diff_mean
            FROM stego_images s
            JOIN originals o ON s.original_id = o.id
            LEFT JOIN trace_results t ON s.id = t.stego_id
            WHERE s.id = ?
        """, (sid,))

        row = c.fetchone()
        conn.close()

        if row is None:
            self.details.setText("No data found.")
            return

        orig_path, orig_size, orig_type, steg_path, steg_size, steg_type, tool, diff_mean, diff_max, shape_mismatch, klr, klg, klb, lsbavg, srm, wav = row

        self.last_images = (orig_path, steg_path)

        self.load_image_to_scene(orig_path, self.orig_scene)
        self.load_image_to_scene(steg_path, self.stego_scene)

        self.orig_info.setText(
            f"Original File: {os.path.basename(orig_path)}\nSize: {orig_size} bytes | Type: {orig_type}")
        self.stego_info.setText(f"""<b>Tool:</b> {tool}<br>
        <b>Size:</b> {steg_size} bytes | <b>Type:</b> {steg_type}<br>
        <b>Shape mismatch:</b> {shape_mismatch}<br>
        <b>Diff mean:</b> {diff_mean:.2f} | <b>Diff max:</b> {diff_max}<br>
        <b>KL Divergence (R,G,B):</b> {klr:.4f}, {klg:.4f}, {klb:.4f}<br>
        <b>LSB Avg Changed %:</b> {lsbavg:.2f}%<br>
        <b>SRM Mean:</b> {srm:.4f} | <b>Wavelet Diff Mean:</b> {wav:.4f}<br>""")

    def show_lsb_diff(self):
        orig_path, steg_path = self.last_images
        if not os.path.exists(orig_path) or not os.path.exists(steg_path):
            self.details.setText("Image paths not available for LSB diff.")
            return
        orig = cv2.imread(orig_path)
        steg = cv2.imread(steg_path)
        if orig is None or steg is None or orig.shape != steg.shape:
            self.details.setText("Failed to load images or shape mismatch.")
            return
        diff_mask = np.zeros(orig.shape[:2], dtype=np.uint8)
        for ch in range(3):
            lsb_orig = orig[:, :, ch] & 1
            lsb_steg = steg[:, :, ch] & 1
            diff_mask |= (lsb_orig ^ lsb_steg).astype(np.uint8)
        heatmap = (diff_mask * 255).astype(np.uint8)
        cv2.imshow("LSB Difference Heatmap", heatmap)

    def load_image_to_scene(self, path, scene):
        if not os.path.isabs(path):
            path = os.path.join(settings.PROJECT_ROOT, path.lstrip("./"))
        print("Resolved path:", path)

        if not os.path.exists(path):
            print("[ERROR] File not found:", path)
            return

        img = cv2.imread(path)
        if img is None:
            print("OpenCV failed to read image.")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scene.clear()
        item = QGraphicsPixmapItem(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))
        scene.addItem(item)


