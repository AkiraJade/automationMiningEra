"""Main PySide6 Application Window for Graal Mining Macro."""

import sys
import numpy as np
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QPushButton, QLabel, QFrame, QStatusBar
)
from PySide6.QtGui import QFont, QIcon, QColor

if sys.platform.startswith("win"):
    import win32gui

from app.core.config import AppConfig
from app.core.events import events
from app.core.logger import setup_logger
from app.window.detector import WindowDetector
from app.window.models import WindowInfo
from app.capture.worker import CaptureWorkerThread
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.mining.mining_controller import MiningController
from app.mining.perception_worker import PerceptionWorkerThread
from app.input.safety import safety

from app.gui.dashboard import DashboardPage
from app.gui.mining_page import MiningPage
from app.gui.calibration import CalibrationPage
from app.gui.settings import SettingsPage
from app.gui.logs import LogsPage

logger = setup_logger("MainWindow")


class MainWindow(QMainWindow):
    """Primary GUI Application Window for Graal Mining Macro."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("Graal Mining Macro")
        self.resize(1280, 800)
        self.setStyleSheet("QMainWindow { background-color: #121216; color: #ffffff; }")

        # Domain Components
        self.detector = WindowDetector(
            title_pattern=self.config.window.title_pattern,
            target_executable=self.config.window.target_executable
        )
        self.capture_worker = CaptureWorkerThread(self.detector, target_fps=self.config.capture.fps)
        self.perception_engine = MiningPerceptionEngine(yolo_path=self.config.vision.yolo_model_path)
        self.mining_controller = MiningController(self.perception_engine)
        self.perception_worker = PerceptionWorkerThread(
            self.mining_controller,
            target_fps=self.config.vision.perception_fps
        )

        # Active Window Tracker & Metrics
        self.current_window_info: Optional[WindowInfo] = None
        self.current_fps: float = 0.0
        self.perception_fps: float = 0.0
        self.last_proc_time_ms: float = 0.0
        self.health_status: str = "HEALTHY"

        self.init_ui()
        self.setup_signals()

        # Start Foreground Diagnostics Timer (100ms interval)
        self.diag_timer = QTimer(self)
        self.diag_timer.setInterval(100)
        self.diag_timer.timeout.connect(self.update_foreground_diagnostic)
        self.diag_timer.start()

        # Start F12 Emergency Listener
        safety.start_listener()

        # Start Worker Threads
        self.perception_worker.start()
        self.capture_worker.start()

    def init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header Bar
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("QFrame { background-color: #1a1a22; border-bottom: 1px solid #2d2d38; }")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)

        app_title = QLabel("GRAAL MINING MACRO")
        app_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #00E5FF;")

        self.lbl_connection = QLabel("● SEARCHING...")
        self.lbl_connection.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_connection.setStyleSheet("color: #FFD600;")

        # Diagnostic Labels: Capture Source vs Foreground Window
        diag_layout = QVBoxLayout()
        diag_layout.setSpacing(2)

        self.lbl_win_info = QLabel("CAPTURE SOURCE: None | HWND: 0 | PID: 0 | Process: N/A")
        self.lbl_win_info.setStyleSheet("color: #00FF66; font-size: 11px; font-weight: bold;")

        self.lbl_foreground = QLabel("FOREGROUND: N/A")
        self.lbl_foreground.setStyleSheet("color: #888888; font-size: 10px;")

        diag_layout.addWidget(self.lbl_win_info)
        diag_layout.addWidget(self.lbl_foreground)

        header_layout.addWidget(app_title)
        header_layout.addWidget(self.lbl_connection)
        header_layout.addStretch()
        header_layout.addLayout(diag_layout)

        root_layout.addWidget(header)

        # Main Body (Left Sidebar Nav + Stacked Content Pages)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Left Sidebar Navigation
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("QFrame { background-color: #16161e; border-right: 1px solid #2d2d38; }")
        nav_layout = QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(8, 16, 8, 16)
        nav_layout.setSpacing(8)

        self.btn_nav_dashboard = self._create_nav_btn("Dashboard")
        self.btn_nav_mining = self._create_nav_btn("Mining")
        self.btn_nav_calibration = self._create_nav_btn("Calibration")
        self.btn_nav_settings = self._create_nav_btn("Settings")
        self.btn_nav_logs = self._create_nav_btn("Logs")

        nav_layout.addWidget(self.btn_nav_dashboard)
        nav_layout.addWidget(self.btn_nav_mining)
        nav_layout.addWidget(self.btn_nav_calibration)
        nav_layout.addWidget(self.btn_nav_settings)
        nav_layout.addWidget(self.btn_nav_logs)
        nav_layout.addStretch()

        body_layout.addWidget(sidebar)

        # Stacked Widget Pages
        self.stacked_pages = QStackedWidget()
        self.page_dashboard = DashboardPage()
        self.page_mining = MiningPage(self.config)
        self.page_calibration = CalibrationPage(self.config)
        self.page_settings = SettingsPage(self.config)
        self.page_logs = LogsPage()

        self.stacked_pages.addWidget(self.page_dashboard)
        self.stacked_pages.addWidget(self.page_mining)
        self.stacked_pages.addWidget(self.page_calibration)
        self.stacked_pages.addWidget(self.page_settings)
        self.stacked_pages.addWidget(self.page_logs)

        body_layout.addWidget(self.stacked_pages, stretch=1)
        root_layout.addLayout(body_layout, stretch=1)

        # Bottom Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar { background-color: #14141a; color: #888888; font-size: 11px; }")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | DryRun: ENABLED | Press F12 anytime to Emergency Stop")

    def _create_nav_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(40)
        btn.setFont(QFont("Segoe UI", 10))
        btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #aaaaaa; border-radius: 4px; text-align: left; padding-left: 16px; }"
            "QPushButton:hover { background-color: #242432; color: #ffffff; }"
            "QPushButton:checked { background-color: #00E5FF; color: #000000; font-weight: bold; }"
        )
        btn.setCheckable(True)
        return btn

    def setup_signals(self) -> None:
        # Navigation Button Clicks
        self.btn_nav_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.btn_nav_mining.clicked.connect(lambda: self.switch_page(1))
        self.btn_nav_calibration.clicked.connect(lambda: self.switch_page(2))
        self.btn_nav_settings.clicked.connect(lambda: self.switch_page(3))
        self.btn_nav_logs.clicked.connect(lambda: self.switch_page(4))
        self.btn_nav_dashboard.setChecked(True)

        # Capture Worker Signals
        self.capture_worker.frame_captured_signal.connect(self.on_frame_captured)
        self.capture_worker.status_changed_signal.connect(self.on_capture_status_changed)

        # Perception Worker Signals
        self.perception_worker.perception_complete_signal.connect(self.on_perception_complete)

        events.capture_fps_updated.connect(self.on_fps_updated)
        events.emergency_stop_triggered.connect(self.on_emergency_triggered)

    def update_foreground_diagnostic(self) -> None:
        """Updates diagnostic information showing the currently focused/foreground window."""
        if sys.platform.startswith("win"):
            try:
                fg_hwnd = win32gui.GetForegroundWindow()
                fg_title = win32gui.GetWindowText(fg_hwnd) or "Unknown Window"
                self.lbl_foreground.setText(f"FOREGROUND: {fg_title} (HWND: {fg_hwnd})")
            except Exception:
                pass

    def switch_page(self, index: int) -> None:
        self.stacked_pages.setCurrentIndex(index)
        buttons = [
            self.btn_nav_dashboard,
            self.btn_nav_mining,
            self.btn_nav_calibration,
            self.btn_nav_settings,
            self.btn_nav_logs,
        ]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

    def on_frame_captured(self, frame: np.ndarray, win_info: WindowInfo) -> None:
        self.current_window_info = win_info
        self.lbl_connection.setText("● CONNECTED")
        self.lbl_connection.setStyleSheet("color: #00FF66;")
        self.lbl_win_info.setText(
            f"CAPTURE SOURCE: {win_info.title} | HWND: {win_info.hwnd} | PID: {win_info.pid} ({win_info.process_name}) | Res: {win_info.client_width}x{win_info.client_height}"
        )

        try:
            # 1. Update live preview frame (GUI main thread, lightweight image conversion)
            self.page_dashboard.game_preview.update_frame(frame, win_info)
            self.page_calibration.set_current_frame(frame)

            # 2. Push frame reference into perception worker single-element buffer (O(1) latest frame wins)
            self.perception_worker.enqueue_frame(frame)
        except Exception as e:
            logger.error(f"Safety boundary caught error in on_frame_captured: {e}", exc_info=True)

    def on_perception_complete(
        self,
        perception: MiningPerceptionResult,
        proc_time_ms: float,
        perception_fps: float,
        health_status: str
    ) -> None:
        try:
            self.perception_fps = perception_fps
            self.last_proc_time_ms = proc_time_ms
            self.health_status = health_status

            metrics = self.perception_worker.get_metrics()
            dropped = metrics.get("dropped_frames", 0)
            dropped_fmt = metrics.get("dropped_formatted", str(dropped))

            # Update Dashboard perception display & health metrics
            self.page_dashboard.update_perception_display(
                p=perception,
                window_connected=True,
                state_str=self.mining_controller.current_state.value,
                health_status=health_status,
                proc_time_ms=proc_time_ms,
                perception_fps=perception_fps,
                dropped_frames=dropped_fmt,
            )

            # Update status bar
            self.status_bar.showMessage(
                f"Capture: {self.current_fps:.1f} FPS | Perception: {perception_fps:.1f} FPS ({proc_time_ms:.1f}ms) | Health: {health_status} | Dropped: {dropped_fmt} | State: {self.mining_controller.current_state.value} | DryRun: {self.config.safety.dry_run}"
            )
        except Exception as e:
            logger.error(f"Error in on_perception_complete handler: {e}", exc_info=True)

    def on_capture_status_changed(self, status_msg: str, win_info: Optional[WindowInfo]) -> None:
        if not win_info or not win_info.is_valid:
            self.lbl_connection.setText("● DISCONNECTED")
            self.lbl_connection.setStyleSheet("color: #FF1744;")
            self.lbl_win_info.setText("CAPTURE SOURCE: None | HWND: 0 | PID: 0 | Process: N/A")
            self.page_dashboard.game_preview.update_frame(None, None)

            # Reset perception metrics on dashboard
            empty_perception = MiningPerceptionResult()
            self.page_dashboard.update_perception_display(
                p=empty_perception,
                window_connected=False,
                state_str="WINDOW_NOT_FOUND",
                health_status="HEALTHY",
                proc_time_ms=0.0,
                perception_fps=0.0,
                dropped_frames=0,
            )

    def on_fps_updated(self, fps: float) -> None:
        self.current_fps = fps
        self.page_dashboard.game_preview.set_fps(fps)
        self.status_bar.showMessage(
            f"Capture: {self.current_fps:.1f} FPS | Perception: {self.perception_fps:.1f} FPS ({self.last_proc_time_ms:.1f}ms) | Health: {self.health_status} | State: {self.mining_controller.current_state.value} | DryRun: {self.config.safety.dry_run}"
        )

    def on_emergency_triggered(self, reason: str) -> None:
        self.lbl_connection.setText("🚨 EMERGENCY STOP")
        self.lbl_connection.setStyleSheet("color: #FF1744;")
        self.status_bar.showMessage(f"🚨 EMERGENCY STOP: {reason}")

    def closeEvent(self, event) -> None:
        logger.info("Application shutting down cleanly...")
        self.diag_timer.stop()
        self.capture_worker.stop()
        self.perception_worker.stop()
        safety.stop_listener()
        event.accept()
