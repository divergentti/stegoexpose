from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox
from PyQt6.QtCore import QProcess
from utils import settings


class TrainingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Tools")
        self.setGeometry(150, 150, 600, 400)

        self.process = None

        self.layout = QVBoxLayout()

        self.instructions = QLabel("Place clean images into './training/originals'.\nStego versions will be generated automatically.")
        self.layout.addWidget(self.instructions)

        self.generate_button = QPushButton("Generate Stego Images")
        self.generate_button.clicked.connect(self.run_generate_stego)
        self.layout.addWidget(self.generate_button)

        self.parse_button = QPushButton("Parse to Database")
        self.parse_button.clicked.connect(self.run_parse2database)
        self.layout.addWidget(self.parse_button)

        self.reset_button = QPushButton("Reset Dataset")
        self.reset_button.clicked.connect(self.reset_dataset)
        self.layout.addWidget(self.reset_button)

        self.terminate_button = QPushButton("Terminate Current Process")
        self.terminate_button.clicked.connect(self.terminate_process)
        self.layout.addWidget(self.terminate_button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        self.setLayout(self.layout)

    def run_generate_stego(self):
        self.run_script("utils/generate_stego_images.py")

    def run_parse2database(self):
        self.run_script("utils/parse2database.py")

    def reset_dataset(self):
        reply = QMessageBox.question(self, "Confirm Reset", "This will backup and clear the current training set. Continue?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # Placeholder: implement actual reset logic
            self.output.append("[INFO] Dataset reset is not yet implemented.")

    def run_script(self, script_path):
        if self.process:
            self.output.append("[WARN] Another process is running.")
            return

        self.process = QProcess(self)
        self.process.setProgram("python3")
        self.process.setArguments([script_path])
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        self.output.append(f"[START] Running {script_path}...\n")
        self.process.start()

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.output.append(data)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode()
        self.output.append(f"[STDERR] {data}")

    def process_finished(self):
        self.output.append("\n[INFO] Process finished.\n")
        self.process = None

    def terminate_process(self):
        if self.process:
            self.process.kill()
            self.output.append("[INFO] Process terminated by user.\n")
            self.process = None
        else:
            self.output.append("[INFO] No process is running.")
