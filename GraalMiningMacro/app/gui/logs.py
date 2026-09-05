"""Real-time Log Viewer Page for Graal Mining Macro."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton, QLabel
from PySide6.QtGui import QFont, QColor, QTextCharFormat
from app.core.logger import get_log_emitter


class LogsPage(QWidget):
    """Real-time application log console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        get_log_emitter().log_emitted.connect(self.append_log)

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_layout = QHBoxLayout()
        title = QLabel("REAL-TIME LOGS")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF;")

        btn_clear = QPushButton("Clear Console")
        btn_clear.setFixedWidth(120)
        btn_clear.setStyleSheet("background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; border-radius: 4px; padding: 4px;")
        btn_clear.clicked.connect(self.clear_logs)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(btn_clear)

        layout.addLayout(header_layout)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background-color: #101014; color: #e0e0e0; border: 1px solid #2d2d35; border-radius: 4px; padding: 8px; }"
        )

        layout.addWidget(self.log_text, stretch=1)

    def append_log(self, timestamp: str, level: str, message: str) -> None:
        color = "#e0e0e0"
        if level == "WARNING":
            color = "#FFD600"
        elif level == "ERROR":
            color = "#FF3D00"
        elif level == "CRITICAL":
            color = "#FF1744"

        formatted_line = f"[{timestamp}] [{level}] {message}"
        self.log_text.appendHtml(f'<span style="color: {color};">{formatted_line}</span>')
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_logs(self) -> None:
        self.log_text.clear()
