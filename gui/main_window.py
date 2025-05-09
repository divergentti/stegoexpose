from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMenuBar, QMenu, QMessageBox
from gui.training_window import TrainingWindow
from gui.learn_window import LearnWindow
from gui.analyze_images import AnalyzeWindow
import sys
from utils import settings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StegaExpose - Main Window")
        self.setGeometry(100, 100, 400, 300)

        self._createMenuBar()
        self._createMainLayout()

    def _createMenuBar(self):
        menu_bar = QMenuBar(self)

        about_menu = QMenu("About", self)
        about_menu.addAction("About This App", self.show_about_dialog)

        settings_menu = QMenu("Settings", self)
        settings_menu.addAction("Configure Paths", self.show_settings_dialog)

        menu_bar.addMenu(about_menu)
        menu_bar.addMenu(settings_menu)
        self.setMenuBar(menu_bar)

    def _createMainLayout(self):
        layout = QVBoxLayout()

        train_button = QPushButton("Training")
        train_button.clicked.connect(self.open_training_window)
        layout.addWidget(train_button)

        learn_button = QPushButton("Learn Steganography")
        learn_button.clicked.connect(self.open_learn_window)
        layout.addWidget(learn_button)

        analyze_button = QPushButton("Analyze Images")
        analyze_button.clicked.connect(self.open_analyze_window)
        layout.addWidget(analyze_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_training_window(self):
        self.training_window = TrainingWindow()
        self.training_window.show()

    def open_learn_window(self):
        self.learn_window = LearnWindow()
        self.learn_window.show()

    def open_analyze_window(self):
        self.analyze_window = AnalyzeWindow()


        self.analyze_window.show()

    def show_about_dialog(self):
        QMessageBox.information(self, "About", "StegaExpose is a steganography detection toolkit.")

    def show_settings_dialog(self):
        QMessageBox.information(self, "Settings", "Settings dialog placeholder (coming soon).")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
