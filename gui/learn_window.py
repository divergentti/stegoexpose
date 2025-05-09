from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from utils import settings

class LearnWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learn Steganography")
        self.setGeometry(200, 200, 700, 500)

        layout = QVBoxLayout()
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.info_area.setHtml(self.get_info_html())

        layout.addWidget(self.info_area)
        self.setLayout(layout)

    def get_info_html(self):
        return """
        <h2>What is Steganography?</h2>
        <p>Steganography is the practice of hiding information within other non-secret text or data.
        In digital images, this typically means embedding data into the least significant bits (LSB) of pixels.</p>

        <h3>Common Tools</h3>
        <ul>
            <li><b>Steghide:</b> Supports BMP and JPEG, encrypts hidden data.</li>
            <li><b>OutGuess:</b> Primarily for JPEG, adjusts DCT coefficients.</li>
            <li><b>OpenStego:</b> User-friendly, supports watermarking and data hiding.</li>
        </ul>

        <h3>Challenges in Detection</h3>
        <p>Detecting steganography is difficult because modifications are subtle and designed to be invisible.
        Each tool leaves different statistical traces. Generalizing a detector across unknown tools or images is challenging.</p>

        <h3>About This Project</h3>
        <p>This tool was designed to create a full pipeline for detecting steganographic traces:
        <ul>
            <li>Generate stego images from clean sources</li>
            <li>Analyze statistical and structural changes</li>
            <li>Extract features into a database</li>
            <li>Train classifiers (e.g. Decision Trees, CNNs)</li>
        </ul>
        The ultimate goal is a hybrid system using both supervised and unsupervised learning.
        </p>
        """
