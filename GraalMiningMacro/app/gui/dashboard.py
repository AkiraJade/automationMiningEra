"""Dashboard Page View for Graal Mining Macro."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QComboBox, QScrollArea
)
from PySide6.QtGui import QFont, QColor
from app.gui.game_preview import GamePreviewWidget
from app.mining.mining_perception import MiningPerceptionResult
from app.input.safety import safety


class DashboardPage(QWidget):
    """Main Dashboard with centered Live Game Display, sidebar stats, frame freeze, and control buttons."""

    start_clicked = Signal()
    stop_clicked = Signal()
    emergency_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Content Splitter (Left Stats Panel + Center Live Game Capture)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Left Sidebar Stats Panel
        stats_frame = QFrame()
        stats_frame.setObjectName("stats_frame")
        stats_frame.setStyleSheet(
            "QFrame#stats_frame { background-color: #1e1e24; border: 1px solid #2d2d35; border-radius: 6px; padding: 12px; }"
        )
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(8)

        stats_title = QLabel("SYSTEM MONITOR")
        stats_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        stats_title.setStyleSheet("QLabel { color: #00E5FF; background: transparent; border: none; }")
        stats_layout.addWidget(stats_title)

        # Metrics Grid (Responsive 2-column layout with explicit row minimum height)
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        def add_metric(row, label_text, default_val):
            lbl = QLabel(f"{label_text}:")
            lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
            lbl.setStyleSheet("QLabel { color: #aaaaaa; background: transparent; border: none; margin: 0; padding: 0; }")
            lbl.setMinimumHeight(22)

            val = QLabel(default_val)
            val.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            val.setStyleSheet("QLabel { color: #ffffff; background: transparent; border: none; margin: 0; padding: 0; }")
            val.setMinimumHeight(22)
            val.setToolTip(default_val)

            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            grid.setRowMinimumHeight(row, 22)
            return val

        self.val_game = add_metric(0, "GAME", "NOT CONNECTED")
        self.val_health = add_metric(1, "PERCEPTION", "HEALTHY")
        self.val_player = add_metric(2, "PLAYER", "SEARCHING")
        self.val_facing = add_metric(3, "FACING", "UNKNOWN")
        self.val_wall = add_metric(4, "WALL", "UNKNOWN")
        self.val_target_src = add_metric(5, "TARGET SRC", "WALL_CONTACT")
        self.val_target = add_metric(6, "TARGET", "SEARCHING")
        self.val_iteration = add_metric(7, "ITERATION", "0 / 3")
        self.val_rock = add_metric(8, "ROCK", "SEARCHING")
        self.val_mini_rock = add_metric(9, "MINI ROCK", "NONE")
        self.val_spider = add_metric(10, "SPIDER", "NONE")
        self.val_drill = add_metric(11, "DRILL", "EQUIPPED")
        self.val_battery = add_metric(12, "BATTERY", "OK")
        self.val_location = add_metric(13, "MINE", "INSIDE")
        self.val_state = add_metric(14, "STATE", "IDLE")
        self.val_conf = add_metric(15, "CONFIDENCE", "0%")
        self.val_perc_fps = add_metric(16, "PERC FPS", "0.0")
        self.val_proc_time = add_metric(17, "PROC TIME", "0 ms")
        self.val_dropped = add_metric(18, "DROPPED", "0 (0.0%)")

        stats_layout.addLayout(grid)

        # Inspection Controls
        insp_lbl = QLabel("INSPECTION CONTROLS:")
        insp_lbl.setStyleSheet("QLabel { color: #888888; background: transparent; border: none; font-weight: bold; font-size: 10px; }")
        stats_layout.addWidget(insp_lbl)

        btn_row = QHBoxLayout()
        self.btn_freeze = QPushButton("❄️ Freeze Frame")
        self.btn_freeze.setStyleSheet("background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; border-radius: 4px; padding: 4px;")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.toggled.connect(self._toggle_freeze)

        self.btn_save_frame = QPushButton("📷 Save Diagnostic")
        self.btn_save_frame.setStyleSheet("background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; border-radius: 4px; padding: 4px;")
        self.btn_save_frame.clicked.connect(self._save_diagnostic)

        btn_row.addWidget(self.btn_freeze)
        btn_row.addWidget(self.btn_save_frame)
        stats_layout.addLayout(btn_row)

        stats_layout.addStretch()

        # Automation Level Selector
        level_lbl = QLabel("AUTOMATION MODE:")
        level_lbl.setStyleSheet("QLabel { color: #888888; background: transparent; border: none; }")
        stats_layout.addWidget(level_lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet("QComboBox { background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; padding: 4px; }")
        self.mode_combo.addItems([
            "MODE 1 - OBSERVE (PASSIVE)",
            "MODE 2 - RECOMMEND",
            "MODE 3 - MOVEMENT",
            "MODE 4 - MINING",
            "MODE 5 - RECOVERY",
            "MODE 6 - FULL MINING"
        ])
        stats_layout.addWidget(self.mode_combo)

        # Scroll area wrapper for stats panel to handle full-screen & small-screen height changes gracefully
        stats_scroll = QScrollArea()
        stats_scroll.setObjectName("stats_scroll")
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMinimumWidth(340)
        stats_scroll.setMaximumWidth(380)
        stats_scroll.setFrameShape(QFrame.Shape.NoFrame)
        stats_scroll.setStyleSheet(
            "QScrollArea#stats_scroll { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 6px; background: #1e1e24; border-radius: 3px; }"
            "QScrollBar::handle:vertical { background: #3d3d4d; border-radius: 3px; }"
        )
        stats_scroll.setWidget(stats_frame)

        # Center Centerpiece Live Game Preview
        self.game_preview = GamePreviewWidget()

        content_layout.addWidget(stats_scroll)
        content_layout.addWidget(self.game_preview, stretch=1)
        main_layout.addLayout(content_layout, stretch=1)

        # Bottom Control Panel Bar
        control_frame = QFrame()
        control_frame.setFixedHeight(60)
        control_frame.setStyleSheet("QFrame { background-color: #1e1e24; border: 1px solid #2d2d35; border-radius: 6px; }")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(12, 6, 12, 6)

        self.btn_start = QPushButton("▶  START OBSERVATION")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #00C853; color: white; border-radius: 4px; padding: 0 20px; }"
            "QPushButton:hover { background-color: #00E676; }"
        )

        self.btn_stop = QPushButton("⏹  STOP")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #37474F; color: white; border-radius: 4px; padding: 0 20px; }"
            "QPushButton:hover { background-color: #455A64; }"
        )

        self.btn_emergency = QPushButton("🚨  EMERGENCY STOP (F12)")
        self.btn_emergency.setFixedHeight(40)
        self.btn_emergency.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_emergency.setStyleSheet(
            "QPushButton { background-color: #D50000; color: white; border-radius: 4px; padding: 0 20px; }"
            "QPushButton:hover { background-color: #FF1744; }"
        )

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_emergency)

        main_layout.addWidget(control_frame)

        # Connect signals
        self.btn_start.clicked.connect(self.start_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_emergency.clicked.connect(self._on_emergency_click)

    def _toggle_freeze(self, checked: bool) -> None:
        self.btn_freeze.setText("▶ Resume Frame" if checked else "❄️ Freeze Frame")
        self.game_preview.set_frozen(checked)

    def _save_diagnostic(self) -> None:
        self.game_preview.save_diagnostic_frame()

    def _on_emergency_click(self) -> None:
        safety.trigger_emergency_stop("GUI Emergency Button Clicked")
        self.emergency_clicked.emit()

    def _set_val(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setToolTip(text)

    def update_perception_display(
        self,
        p: MiningPerceptionResult,
        window_connected: bool,
        state_str: str,
        health_status: str = "HEALTHY",
        proc_time_ms: float = 0.0,
        perception_fps: float = 0.0,
        dropped_frames: object = 0,
    ) -> None:
        game_text = "CONNECTED" if window_connected else "NOT FOUND"
        self._set_val(self.val_game, game_text)
        self.val_game.setStyleSheet("QLabel { color: #00FF66; background: transparent; border: none; }" if window_connected else "QLabel { color: #FF1744; background: transparent; border: none; }")

        if health_status == "HEALTHY":
            self._set_val(self.val_health, "HEALTHY")
            self.val_health.setStyleSheet("QLabel { color: #00FF66; background: transparent; border: none; }")
        elif health_status == "SLOW":
            self._set_val(self.val_health, "SLOW")
            self.val_health.setStyleSheet("QLabel { color: #FFD600; background: transparent; border: none; }")
        else:
            self._set_val(self.val_health, "ERROR")
            self.val_health.setStyleSheet("QLabel { color: #FF1744; background: transparent; border: none; }")

        self._set_val(self.val_player, p.player.summary_text())
        facing_str = getattr(p.player, 'facing_direction', 'UNKNOWN')
        self._set_val(self.val_facing, facing_str)
        self._set_val(self.val_wall, p.wall.summary_text())
        self._set_val(self.val_target_src, getattr(p.target, 'target_source', 'WALL_CONTACT'))
        self._set_val(self.val_target, p.target.summary_text())
        self._set_val(self.val_iteration, f"{p.target.iteration} / 3")

        rock_text = "YELLOW COMPLETE" if p.yellow_glow.is_confirmed else ("MINING" if p.target.detected else "SEARCHING")
        self._set_val(self.val_rock, rock_text)
        self._set_val(self.val_mini_rock, p.mini_rock.summary_text())

        self._set_val(self.val_spider, p.spider.summary_text())
        self.val_spider.setStyleSheet("QLabel { color: #FF1744; background: transparent; border: none; }" if p.spider.detected else "QLabel { color: #ffffff; background: transparent; border: none; }")

        bat_text = p.status.battery_state.name.replace("BATTERY_", "").replace("_", " ").upper() if hasattr(p.status.battery_state, "name") else str(p.status.battery_state.value)
        self._set_val(self.val_battery, bat_text)

        loc_text = p.status.mine_state.name.replace("_", " ").upper() if hasattr(p.status.mine_state, "name") else str(p.status.mine_state.value)
        self._set_val(self.val_location, loc_text)

        clean_state = state_str.split(".")[-1].replace("_", " ").upper() if state_str else "IDLE"
        self._set_val(self.val_state, clean_state)

        self._set_val(self.val_conf, f"{p.overall_confidence * 100:.0f}%")
        self._set_val(self.val_perc_fps, f"{perception_fps:.1f} FPS")

        proc_str = f"{proc_time_ms:.1f} ms"
        if p.detector_timings:
            t = p.detector_timings
            breakdown_str = (
                f"PROC TIME: {proc_time_ms:.1f} ms\n"
                f"• Player: {t.get('player', 0.0):.1f} ms\n"
                f"• Spider: {t.get('spider', 0.0):.1f} ms\n"
                f"• Yellow Rock: {t.get('yellow_rock', 0.0):.1f} ms\n"
                f"• Wall: {t.get('wall', 0.0):.1f} ms\n"
                f"• Target: {t.get('target', 0.0):.1f} ms\n"
                f"• Message: {t.get('message', 0.0):.1f} ms\n"
                f"• Status: {t.get('status', 0.0):.1f} ms\n"
                f"• Reference Matcher: {t.get('reference_matcher_total', 0.0):.1f} ms"
            )
            self.val_proc_time.setText(proc_str)
            self.val_proc_time.setToolTip(breakdown_str)
        else:
            self._set_val(self.val_proc_time, proc_str)

        if isinstance(dropped_frames, str):
            dropped_str = dropped_frames
        else:
            dropped_str = f"{dropped_frames:,}"
        self._set_val(self.val_dropped, dropped_str)

        self.game_preview.update_perception(p)
