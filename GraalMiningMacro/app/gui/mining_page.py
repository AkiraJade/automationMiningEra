"""Mining Settings & Keybinding Page for Graal Mining Macro."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLabel
)
from PySide6.QtGui import QFont
from app.core.config import AppConfig


class MiningPage(QWidget):
    """Page for configuring mining parameters, keybindings, and thresholds."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title = QLabel("MINING CONFIGURATION")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF;")
        layout.addWidget(title)

        # Keybindings Box
        key_box = QGroupBox("Keybindings")
        key_box.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        key_form = QFormLayout(key_box)

        self.key_drill = QLineEdit(self.config.keys.drill_equip_key)
        self.key_spider = QLineEdit(self.config.keys.spider_combat_key)
        self.key_attack = QLineEdit(self.config.keys.attack_key)

        key_form.addRow("Drill Equip Key:", self.key_drill)
        key_form.addRow("Spider Combat Key:", self.key_spider)
        key_form.addRow("Attack Key:", self.key_attack)

        layout.addWidget(key_box)

        # Mining Timing Box
        timing_box = QGroupBox("Mining Timing & Level")
        timing_box.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        timing_form = QFormLayout(timing_box)

        self.spin_level = QSpinBox()
        self.spin_level.setRange(0, 10)
        self.spin_level.setValue(self.config.mining_level)

        self.spin_action_int = QDoubleSpinBox()
        self.spin_action_int.setRange(0.1, 5.0)
        self.spin_action_int.setValue(self.config.timing.mining_action_interval)

        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(1.0, 60.0)
        self.spin_cooldown.setValue(self.config.timing.nothing_to_mine_cooldown)

        timing_form.addRow("Mining Level:", self.spin_level)
        timing_form.addRow("Action Interval (s):", self.spin_action_int)
        timing_form.addRow("Nothing-to-Mine Cooldown (s):", self.spin_cooldown)

        layout.addWidget(timing_box)
        layout.addStretch()
