"""Main Entry Point for Graal Mining Macro."""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.core.config import AppConfig
from app.core.logger import setup_logger
from app.gui.main_window import MainWindow

logger = setup_logger("Main")


def main():
    logger.info("Initializing Graal Mining Macro v1.0.0...")

    # Enable High-DPI scaling for crisp text on modern displays
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load configuration
    config_file = "config.json"
    config = AppConfig.load_from_file(config_file)
    logger.info(f"Loaded config: Window Pattern='{config.window.title_pattern}', Target FPS={config.capture.fps}")

    # Launch MainWindow
    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
