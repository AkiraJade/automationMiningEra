"""Unit tests for PySide6 GUI Window and Widgets."""

import pytest
from PySide6.QtWidgets import QApplication
from app.core.config import AppConfig
from app.gui.main_window import MainWindow
from app.input.safety import safety


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_main_window_instantiation(qapp):
    config = AppConfig()
    win = MainWindow(config)
    assert win.windowTitle() == "Graal Mining Macro"
    assert win.stacked_pages.count() == 5

    # Stop threads cleanly
    win.capture_worker.stop()
    safety.stop_listener()
    win.close()
