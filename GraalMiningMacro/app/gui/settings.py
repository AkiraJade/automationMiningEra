"""Settings Page for Graal Mining Macro with Candidate Windows Debug Inspector."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLineEdit, QSpinBox, QCheckBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QFont, QColor
from app.core.config import AppConfig
from app.window.detector import WindowDetector


class SettingsPage(QWidget):
    """Application settings for window detection, capture FPS, safety, and Candidate Windows Inspector."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.detector = WindowDetector(
            title_pattern=self.config.window.title_pattern,
            target_executable=self.config.window.target_executable
        )
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title = QLabel("GLOBAL SETTINGS & DEBUG INSPECTOR")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF;")
        layout.addWidget(title)

        # Window Matching Box
        win_box = QGroupBox("Game Window Detection Settings")
        win_box.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        win_form = QFormLayout(win_box)

        self.txt_executable = QLineEdit(self.config.window.target_executable)
        self.txt_title = QLineEdit(self.config.window.title_pattern)
        self.chk_auto = QCheckBox("Auto Detect Game Window")
        self.chk_auto.setChecked(self.config.window.auto_detect)

        win_form.addRow("Game Executable (Process Name):", self.txt_executable)
        win_form.addRow("Game Window Title Pattern:", self.txt_title)
        win_form.addRow("", self.chk_auto)

        layout.addWidget(win_box)

        # Candidate Windows Debug Inspector Box
        inspector_box = QGroupBox("Candidate Windows Debug Inspector")
        inspector_box.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        inspector_layout = QVBoxLayout(inspector_box)

        insp_header = QHBoxLayout()
        insp_label = QLabel("Enumerates all running windows on your system to inspect detection scores & process rules:")
        insp_label.setStyleSheet("color: #aaaaaa;")
        btn_refresh = QPushButton("🔍 Refresh Candidates")
        btn_refresh.setFixedWidth(160)
        btn_refresh.setStyleSheet("background-color: #00E5FF; color: #000000; font-weight: bold; border-radius: 4px; padding: 4px;")
        btn_refresh.clicked.connect(self.refresh_candidates)

        insp_header.addWidget(insp_label)
        insp_header.addStretch()
        insp_header.addWidget(btn_refresh)
        inspector_layout.addLayout(insp_header)

        self.table_candidates = QTableWidget()
        self.table_candidates.setColumnCount(7)
        self.table_candidates.setHorizontalHeaderLabels([
            "HWND", "PID", "Process Name", "Window Title", "Resolution", "Score", "Status / Rejection Reason"
        ])
        self.table_candidates.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_candidates.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_candidates.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table_candidates.setStyleSheet(
            "QTableWidget { background-color: #101014; color: #ffffff; gridline-color: #2d2d38; }"
            "QHeaderView::section { background-color: #1a1a24; color: #00E5FF; font-weight: bold; }"
        )

        inspector_layout.addWidget(self.table_candidates)
        layout.addWidget(inspector_box, stretch=1)

        # Populate table on init
        self.refresh_candidates()

    def refresh_candidates(self) -> None:
        pattern = self.txt_title.text().strip() or "GraalOnline Era"
        exe = self.txt_executable.text().strip() or "Era.exe"
        self.detector.title_pattern = pattern
        self.detector.target_executable = exe
        candidates = self.detector.list_candidate_windows(override_pattern=pattern, override_exe=exe)

        self.table_candidates.setRowCount(len(candidates))

        for row, c in enumerate(candidates):
            self.table_candidates.setItem(row, 0, QTableWidgetItem(str(c.hwnd)))
            self.table_candidates.setItem(row, 1, QTableWidgetItem(str(c.pid)))
            self.table_candidates.setItem(row, 2, QTableWidgetItem(c.process_name))
            self.table_candidates.setItem(row, 3, QTableWidgetItem(c.title))
            self.table_candidates.setItem(row, 4, QTableWidgetItem(f"{c.client_width}x{c.client_height}"))
            self.table_candidates.setItem(row, 5, QTableWidgetItem(str(c.score)))

            status_text = "ACCEPTED (SELECTED TARGET)" if c.is_accepted else c.rejection_reason
            item_status = QTableWidgetItem(status_text)
            if c.is_accepted:
                item_status.setForeground(QColor("#00FF66"))
            else:
                item_status.setForeground(QColor("#FF1744" if "own macro" in status_text.lower() else "#888888"))
            self.table_candidates.setItem(row, 6, item_status)
