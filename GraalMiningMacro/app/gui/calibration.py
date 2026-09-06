"""Visual Calibration Page & Reference Library Manager for Graal Mining Macro."""

import os
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QSlider, QLabel, QPushButton,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog, QDialog,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QFont, QPixmap, QImage, QColor, QPen, QPainter

from app.core.config import AppConfig
from app.vision.reference import (
    ReferenceManager, ReferenceImage, ReferenceMatchResult, CATEGORIES, SUBCATEGORIES, DEFAULT_THRESHOLDS
)
from app.core.logger import setup_logger

logger = setup_logger("CalibrationPage")


class CropLabelWidget(QLabel):
    """Custom image display widget supporting interactive rectangle drag-cropping."""

    crop_changed = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_original: Optional[QPixmap] = None
        self._start_pos: Optional[QPoint] = None
        self._current_pos: Optional[QPoint] = None
        self._is_selecting: bool = False
        self.selected_rect: Optional[QRect] = None

    def set_frame_image(self, bgr_img: np.ndarray) -> None:
        if bgr_img is None or bgr_img.size == 0:
            return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap_original = QPixmap.fromImage(qimg)
        self.setPixmap(self._pixmap_original)
        self.selected_rect = None
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_original:
            self._start_pos = event.position().toPoint()
            self._current_pos = self._start_pos
            self._is_selecting = True
            self.selected_rect = None
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._is_selecting:
            self._current_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_selecting:
            self._current_pos = event.position().toPoint()
            self._is_selecting = False
            if self._start_pos and self._current_pos:
                self.selected_rect = QRect(self._start_pos, self._current_pos).normalized()
                self.crop_changed.emit(self.selected_rect)
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._start_pos and self._current_pos:
            painter = QPainter(self)
            rect = QRect(self._start_pos, self._current_pos).normalized()
            painter.setPen(QPen(QColor("#00E5FF"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(0, 229, 255, 40))
            painter.drawRect(rect)


class AddReferenceDialog(QDialog):
    """Dialog allowing users to upload a reference image from disk and specify metadata."""

    def __init__(self, parent=None, default_category: str = "player"):
        super().__init__(parent)
        self.setWindowTitle("Add Reference Image From Disk")
        self.resize(480, 420)
        self.setStyleSheet("background-color: #1a1a24; color: #ffffff;")

        self.selected_file_path: str = ""
        self.default_category = default_category
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # File Picker Header
        file_layout = QHBoxLayout()
        self.lbl_filepath = QLabel("No file selected...")
        self.lbl_filepath.setStyleSheet("color: #888888; font-size: 11px;")
        btn_browse = QPushButton("📁 Browse Image...")
        btn_browse.setStyleSheet("background-color: #00E5FF; color: #000000; font-weight: bold; border-radius: 4px; padding: 6px 12px;")
        btn_browse.clicked.connect(self._browse_file)

        file_layout.addWidget(self.lbl_filepath, stretch=1)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)

        # Preview Thumbnail
        self.lbl_preview = QLabel("Preview Thumbnail")
        self.lbl_preview.setFixedSize(160, 120)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("border: 1px dashed #444455; background-color: #101014;")
        layout.addWidget(self.lbl_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # Metadata Form
        form = QFormLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("e.g. player_right")

        self.combo_category = QComboBox()
        self.combo_category.addItems(CATEGORIES)
        idx = self.combo_category.findText(self.default_category)
        if idx >= 0:
            self.combo_category.setCurrentIndex(idx)
        self.combo_category.currentTextChanged.connect(self._on_category_changed)

        self.combo_subcategory = QComboBox()
        self._update_subcategories(self.default_category)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.50, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(DEFAULT_THRESHOLDS.get(self.default_category, 0.80))

        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText("Optional notes...")

        form.addRow("Reference Name:", self.txt_name)
        form.addRow("Category:", self.combo_category)
        form.addRow("Subcategory:", self.combo_subcategory)
        form.addRow("Confidence Threshold:", self.spin_threshold)
        form.addRow("Notes:", self.txt_notes)
        layout.addLayout(form)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_save = QPushButton("💾 Save Reference")
        btn_save.setStyleSheet("background-color: #00C853; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        btn_save.clicked.connect(self.accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #37474F; color: white; padding: 6px 16px; border-radius: 4px;")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(btn_save)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "", "Image Files (*.png *.jpg *.jpeg *.webp)"
        )
        if path and os.path.exists(path):
            self.selected_file_path = path
            self.lbl_filepath.setText(os.path.basename(path))
            if not self.txt_name.text():
                base_name = os.path.splitext(os.path.basename(path))[0]
                self.txt_name.setText(base_name)

            pix = QPixmap(path)
            if not pix.isNull():
                self.lbl_preview.setPixmap(pix.scaled(160, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_category_changed(self, cat: str) -> None:
        self._update_subcategories(cat)
        self.spin_threshold.setValue(DEFAULT_THRESHOLDS.get(cat, 0.80))

    def _update_subcategories(self, cat: str) -> None:
        self.combo_subcategory.clear()
        subs = SUBCATEGORIES.get(cat, ["custom"])
        self.combo_subcategory.addItems(subs)


class GameCropDialog(QDialog):
    """Dialog allowing users to select and crop a reference image directly from the live Era game frame."""

    def __init__(self, current_frame_bgr: np.ndarray, parent=None, default_category: str = "player"):
        super().__init__(parent)
        self.setWindowTitle("Capture & Crop Reference From Live Game Frame")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #1a1a24; color: #ffffff;")

        self.current_frame = current_frame_bgr
        self.cropped_image: Optional[np.ndarray] = None
        self.default_category = default_category

        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)

        instruction = QLabel("Click and drag a rectangle over the target object in the game view to crop it:")
        instruction.setStyleSheet("color: #00E5FF; font-weight: bold;")
        layout.addWidget(instruction)

        # Image Crop Widget
        self.crop_widget = CropLabelWidget()
        self.crop_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_widget.setStyleSheet("border: 1px solid #2d2d38; background-color: #101014;")
        self.crop_widget.set_frame_image(self.current_frame)
        self.crop_widget.crop_changed.connect(self._on_crop_selected)

        scroll = QScrollArea()
        scroll.setWidget(self.crop_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        # Crop Preview & Metadata Form
        meta_layout = QHBoxLayout()

        self.lbl_crop_preview = QLabel("Crop Preview")
        self.lbl_crop_preview.setFixedSize(120, 90)
        self.lbl_crop_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_crop_preview.setStyleSheet("border: 1px dashed #555566; background-color: #101014;")
        meta_layout.addWidget(self.lbl_crop_preview)

        form = QFormLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("e.g. yellow_rock_complete")

        self.combo_category = QComboBox()
        self.combo_category.addItems(CATEGORIES)
        idx = self.combo_category.findText(self.default_category)
        if idx >= 0:
            self.combo_category.setCurrentIndex(idx)

        self.combo_subcategory = QComboBox()
        self.combo_subcategory.addItems(SUBCATEGORIES.get(self.default_category, ["custom"]))

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.50, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(DEFAULT_THRESHOLDS.get(self.default_category, 0.80))

        form.addRow("Name:", self.txt_name)
        form.addRow("Category:", self.combo_category)
        form.addRow("Subcategory:", self.combo_subcategory)
        form.addRow("Threshold:", self.spin_threshold)
        meta_layout.addLayout(form, stretch=1)

        layout.addLayout(meta_layout)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_save = QPushButton("✂️ Crop & Save Reference")
        btn_save.setStyleSheet("background-color: #00C853; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        btn_save.clicked.connect(self._on_save_clicked)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #37474F; color: white; padding: 6px 16px; border-radius: 4px;")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(btn_save)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def _on_crop_selected(self, rect: QRect) -> None:
        if self.current_frame is None or rect.width() < 5 or rect.height() < 5:
            return

        x = max(0, rect.x())
        y = max(0, rect.y())
        w = min(rect.width(), self.current_frame.shape[1] - x)
        h = min(rect.height(), self.current_frame.shape[0] - y)

        if w > 0 and h > 0:
            self.cropped_image = self.current_frame[y:y+h, x:x+w].copy()

            # Show crop preview
            rgb = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            ch = rgb.shape[2]
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.lbl_crop_preview.setPixmap(pix.scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_save_clicked(self) -> None:
        if self.cropped_image is None or self.cropped_image.size == 0:
            QMessageBox.warning(self, "Crop Required", "Please drag a rectangle on the image to crop an object before saving.")
            return
        if not self.txt_name.text().strip():
            QMessageBox.warning(self, "Name Required", "Please enter a reference name.")
            return
        self.accept()


class ReferenceTestDialog(QDialog):
    """Diagnostic Modal Dialog testing all enabled references against current live frame."""

    def __init__(self, frame_bgr: np.ndarray, reference_manager: ReferenceManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference Library Diagnostic Test Results")
        self.resize(900, 520)
        self.setStyleSheet("background-color: #1a1a24; color: #ffffff;")

        self.frame_bgr = frame_bgr
        self.manager = reference_manager
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("REFERENCE TEST DIAGNOSTIC SUMMARY")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF;")
        layout.addWidget(title)

        table = QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            "Category", "Reference Name", "Match Result", "Raw Score", "Final Conf", "Bounding Box", "Scale", "Rejection Reason"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        table.setStyleSheet(
            "QTableWidget { background-color: #101014; color: #ffffff; gridline-color: #2d2d38; }"
            "QHeaderView::section { background-color: #1a1a24; color: #00E5FF; font-weight: bold; }"
        )

        all_refs = self.manager.registry.get_all()
        table.setRowCount(len(all_refs))

        for row, ref in enumerate(all_refs):
            match_res = self.manager.matcher.match_single(self.frame_bgr, ref) if ref.enabled else None

            table.setItem(row, 0, QTableWidgetItem(ref.category.upper()))
            table.setItem(row, 1, QTableWidgetItem(ref.name))

            if not ref.enabled:
                item_res = QTableWidgetItem("DISABLED")
                item_res.setForeground(QColor("#888888"))
                table.setItem(row, 2, item_res)
                table.setItem(row, 3, QTableWidgetItem("N/A"))
                table.setItem(row, 4, QTableWidgetItem("N/A"))
                table.setItem(row, 5, QTableWidgetItem("N/A"))
                table.setItem(row, 6, QTableWidgetItem("1.00x"))
                table.setItem(row, 7, QTableWidgetItem("Reference disabled"))
            elif match_res and match_res.found:
                item_res = QTableWidgetItem("MATCH PASSED ✓")
                item_res.setForeground(QColor("#00FF66"))
                table.setItem(row, 2, item_res)
                table.setItem(row, 3, QTableWidgetItem(f"{match_res.raw_score:.2f}"))
                table.setItem(row, 4, QTableWidgetItem(f"{match_res.confidence * 100:.1f}%"))
                bbox_str = str(match_res.bbox) if match_res.bbox else "N/A"
                table.setItem(row, 5, QTableWidgetItem(bbox_str))
                table.setItem(row, 6, QTableWidgetItem(f"{match_res.scale:.2f}x"))
                table.setItem(row, 7, QTableWidgetItem("PASSED"))
            else:
                raw_str = f"{match_res.raw_score:.2f}" if match_res else "0.00"
                conf_str = "0.0%"
                item_res = QTableWidgetItem("NO MATCH ✗")
                item_res.setForeground(QColor("#FF1744"))
                table.setItem(row, 2, item_res)
                table.setItem(row, 3, QTableWidgetItem(raw_str))
                table.setItem(row, 4, QTableWidgetItem(conf_str))
                bbox_str = str(match_res.bbox) if (match_res and match_res.bbox) else "N/A"
                table.setItem(row, 5, QTableWidgetItem(bbox_str))
                scale_str = f"{match_res.scale:.2f}x" if match_res else "1.00x"
                table.setItem(row, 6, QTableWidgetItem(scale_str))
                rej_reason = match_res.rejection_reason if match_res else "Unknown"
                table.setItem(row, 7, QTableWidgetItem(rej_reason))

        layout.addWidget(table, stretch=1)

        btn_close = QPushButton("Close Test Results")
        btn_close.setStyleSheet("background-color: #37474F; color: white; padding: 6px 16px; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)


class CalibrationPage(QWidget):
    """Visual calibration page & Reference Library manager for Graal Mining Macro."""

    calibration_changed = Signal(dict)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.reference_manager = ReferenceManager()
        self._current_live_frame: Optional[np.ndarray] = None
        self.init_ui()

    def set_current_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        self._current_live_frame = frame_bgr

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("VISUAL CALIBRATION & REFERENCE LIBRARY")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00E5FF;")

        btn_test_all = QPushButton("🔍 Test References")
        btn_test_all.setStyleSheet("background-color: #00E5FF; color: black; font-weight: bold; border-radius: 4px; padding: 6px 12px;")
        btn_test_all.clicked.connect(self._run_reference_test)

        btn_save = QPushButton("💾 Save Calibration")
        btn_save.setStyleSheet("background-color: #00C853; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px;")
        btn_save.clicked.connect(self.save_calibration)

        btn_reset = QPushButton("🔄 Reset Defaults")
        btn_reset.setStyleSheet("background-color: #37474F; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px;")
        btn_reset.clicked.connect(self.reset_defaults)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(btn_test_all)
        header_layout.addWidget(btn_save)
        header_layout.addWidget(btn_reset)
        layout.addLayout(header_layout)

        # Tabbed Layout: Tab 1 = Reference Library, Tab 2 = Color / HSV Calibration
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #2d2d38; background-color: #14141a; }"
            "QTabBar::tab { background-color: #1a1a24; color: #aaaaaa; padding: 8px 16px; font-weight: bold; }"
            "QTabBar::tab:selected { background-color: #00E5FF; color: #000000; }"
        )

        # TAB 1: REFERENCE LIBRARY
        tab_ref = QWidget()
        tab_ref_layout = QVBoxLayout(tab_ref)

        self.ref_tab_widget = QTabWidget()
        self.ref_tab_widget.setStyleSheet("QTabBar::tab { padding: 6px 12px; }")

        for cat in CATEGORIES:
            cat_page = self._build_category_page(cat)
            self.ref_tab_widget.addTab(cat_page, cat.upper())

        tab_ref_layout.addWidget(self.ref_tab_widget)
        self.tabs.addTab(tab_ref, "REFERENCE LIBRARY")

        # TAB 2: HSV COLOR CALIBRATION
        tab_hsv = QWidget()
        tab_hsv_layout = QVBoxLayout(tab_hsv)

        hsv_box = QGroupBox("Yellow Glow Rock Detector & Temporal Confirmation")
        hsv_box.setStyleSheet("QGroupBox { color: #ffffff; font-weight: bold; }")
        hsv_form = QFormLayout(hsv_box)

        self.slider_h_min = self._make_slider(0, 180, self.config.vision.yellow_glow_hsv_min[0])
        self.slider_s_min = self._make_slider(0, 255, self.config.vision.yellow_glow_hsv_min[1])
        self.slider_v_min = self._make_slider(0, 255, self.config.vision.yellow_glow_hsv_min[2])

        self.slider_h_max = self._make_slider(0, 180, self.config.vision.yellow_glow_hsv_max[0])
        self.slider_s_max = self._make_slider(0, 255, self.config.vision.yellow_glow_hsv_max[1])
        self.slider_v_max = self._make_slider(0, 255, self.config.vision.yellow_glow_hsv_max[2])

        self.spin_required_frames = QSpinBox()
        self.spin_required_frames.setRange(1, 10)
        self.spin_required_frames.setValue(3)

        hsv_form.addRow("Hue Min (H):", self.slider_h_min)
        hsv_form.addRow("Saturation Min (S):", self.slider_s_min)
        hsv_form.addRow("Value Min (V):", self.slider_v_min)
        hsv_form.addRow("Hue Max (H):", self.slider_h_max)
        hsv_form.addRow("Saturation Max (S):", self.slider_s_max)
        hsv_form.addRow("Value Max (V):", self.slider_v_max)
        hsv_form.addRow("Temporal Confirmation Frames (1-10):", self.spin_required_frames)

        tab_hsv_layout.addWidget(hsv_box)
        tab_hsv_layout.addStretch()

        self.tabs.addTab(tab_hsv, "HSV / DETECTOR TUNING")

        layout.addWidget(self.tabs, stretch=1)

    def _build_category_page(self, category: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        # Action Buttons Header
        btn_bar = QHBoxLayout()
        lbl_cat = QLabel(f"CATEGORY: {category.upper()}")
        lbl_cat.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_cat.setStyleSheet("color: #00E5FF;")

        btn_add = QPushButton("➕ Add Reference From Disk")
        btn_add.setStyleSheet("background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; padding: 4px 10px; border-radius: 4px;")
        btn_add.clicked.connect(lambda _, c=category: self._open_add_reference_dialog(c))

        btn_capture = QPushButton("📷 Capture From Game")
        btn_capture.setStyleSheet("background-color: #2b2b36; color: #ffffff; border: 1px solid #3d3d4d; padding: 4px 10px; border-radius: 4px;")
        btn_capture.clicked.connect(lambda _, c=category: self._open_game_crop_dialog(c))

        btn_bar.addWidget(lbl_cat)
        btn_bar.addStretch()
        btn_bar.addWidget(btn_add)
        btn_bar.addWidget(btn_capture)
        layout.addLayout(btn_bar)

        # Scrollable Cards Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #2d2d38; background-color: #101014; }")

        cards_container = QWidget()
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setSpacing(8)

        refs = self.reference_manager.registry.get_by_category(category)
        if not refs:
            lbl_empty = QLabel(f"No reference images registered for category '{category.upper()}'. Click '+ Add Reference' or 'Capture From Game' above.")
            lbl_empty.setStyleSheet("color: #666677; font-style: italic; padding: 16px;")
            cards_layout.addWidget(lbl_empty)
        else:
            for ref in refs:
                card = self._build_reference_card(ref)
                cards_layout.addWidget(card)

        cards_layout.addStretch()
        scroll.setWidget(cards_container)
        layout.addWidget(scroll, stretch=1)

        return page

    def _build_reference_card(self, ref: ReferenceImage) -> QFrame:
        card = QFrame()
        card.setFixedHeight(80)
        card.setStyleSheet("QFrame { background-color: #1e1e28; border: 1px solid #2d2d38; border-radius: 4px; padding: 4px; }")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)

        # Thumbnail
        lbl_thumb = QLabel()
        lbl_thumb.setFixedSize(70, 60)
        lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_thumb.setStyleSheet("background-color: #101014; border: 1px solid #333344;")

        if ref.file_path and os.path.exists(ref.file_path):
            pix = QPixmap(ref.file_path)
            if not pix.isNull():
                lbl_thumb.setPixmap(pix.scaled(70, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                lbl_thumb.setText("CORRUPT")
                lbl_thumb.setStyleSheet("color: #FF1744; font-size: 9px;")
        else:
            lbl_thumb.setText("MISSING")
            lbl_thumb.setStyleSheet("color: #FF1744; font-size: 9px;")

        # Enabled Checkbox
        chk_enable = QCheckBox()
        chk_enable.setChecked(ref.enabled)
        chk_enable.toggled.connect(lambda checked, r_id=ref.id: self.reference_manager.registry.toggle_reference(r_id, checked))

        # Title / Subcategory info
        info_layout = QVBoxLayout()
        lbl_name = QLabel(ref.name)
        lbl_name.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lbl_name.setStyleSheet("color: #ffffff;")

        lbl_sub = QLabel(f"Subcategory: {ref.subcategory} | ID: {ref.id}")
        lbl_sub.setStyleSheet("color: #888888; font-size: 10px;")

        info_layout.addWidget(lbl_name)
        info_layout.addWidget(lbl_sub)

        # Threshold Spinbox
        spin_thresh = QDoubleSpinBox()
        spin_thresh.setRange(0.50, 0.99)
        spin_thresh.setSingleStep(0.05)
        spin_thresh.setValue(ref.threshold)
        spin_thresh.setFixedWidth(70)
        spin_thresh.valueChanged.connect(lambda val, r_id=ref.id: self.reference_manager.registry.update_threshold(r_id, val))

        # Delete Button
        btn_del = QPushButton("🗑️")
        btn_del.setFixedSize(32, 32)
        btn_del.setStyleSheet("background-color: #3a1c1c; color: #FF1744; border: 1px solid #552222; border-radius: 4px;")
        btn_del.clicked.connect(lambda _, r_id=ref.id: self._delete_reference_item(r_id))

        layout.addWidget(chk_enable)
        layout.addWidget(lbl_thumb)
        layout.addLayout(info_layout, stretch=1)
        layout.addWidget(QLabel("Threshold:"))
        layout.addWidget(spin_thresh)
        layout.addWidget(btn_del)

        return card

    def _open_add_reference_dialog(self, category: str) -> None:
        dlg = AddReferenceDialog(self, default_category=category)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if not dlg.selected_file_path or not os.path.exists(dlg.selected_file_path):
                QMessageBox.warning(self, "Invalid File", "Please select a valid image file.")
                return

            name = dlg.txt_name.text().strip() or "Unnamed Reference"
            cat = dlg.combo_category.currentText()
            sub = dlg.combo_subcategory.currentText()
            thresh = dlg.spin_threshold.value()
            notes = dlg.txt_notes.text().strip()

            ref = self.reference_manager.registry.add_reference(
                name=name,
                category=cat,
                subcategory=sub,
                source_file_or_image=dlg.selected_file_path,
                threshold=thresh,
                notes=notes
            )

            if ref:
                self.refresh_ui()

    def _open_game_crop_dialog(self, category: str) -> None:
        if self._current_live_frame is None or self._current_live_frame.size == 0:
            QMessageBox.warning(self, "No Game Frame", "No live GraalOnline Era game frame available to crop from.")
            return

        dlg = GameCropDialog(self._current_live_frame, self, default_category=category)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.cropped_image is not None:
            name = dlg.txt_name.text().strip() or "Cropped Reference"
            cat = dlg.combo_category.currentText()
            sub = dlg.combo_subcategory.currentText()
            thresh = dlg.spin_threshold.value()

            ref = self.reference_manager.registry.add_reference(
                name=name,
                category=cat,
                subcategory=sub,
                source_file_or_image=dlg.cropped_image,
                threshold=thresh,
            )

            if ref:
                self.refresh_ui()

    def _delete_reference_item(self, ref_id: str) -> None:
        if self.reference_manager.registry.delete_reference(ref_id):
            self.refresh_ui()

    def _run_reference_test(self) -> None:
        if self._current_live_frame is None or self._current_live_frame.size == 0:
            QMessageBox.warning(self, "No Active Frame", "No live game frame available to run reference test.")
            return

        dlg = ReferenceTestDialog(self._current_live_frame, self.reference_manager, self)
        dlg.exec()

    def refresh_ui(self) -> None:
        """Rebuilds the category tab pages after adding/deleting references."""
        cur_idx = self.ref_tab_widget.currentIndex()
        self.ref_tab_widget.clear()

        for cat in CATEGORIES:
            cat_page = self._build_category_page(cat)
            self.ref_tab_widget.addTab(cat_page, cat.upper())

        if 0 <= cur_idx < self.ref_tab_widget.count():
            self.ref_tab_widget.setCurrentIndex(cur_idx)

    def _make_slider(self, min_v: int, max_v: int, cur_v: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(cur_v)
        return slider

    def save_calibration(self) -> None:
        self.config.vision.yellow_glow_hsv_min = [
            self.slider_h_min.value(),
            self.slider_s_min.value(),
            self.slider_v_min.value(),
        ]
        self.config.vision.yellow_glow_hsv_max = [
            self.slider_h_max.value(),
            self.slider_s_max.value(),
            self.slider_v_max.value(),
        ]
        self.config.save_to_file()
        self.reference_manager.registry.save()
        self.calibration_changed.emit(self.config.to_dict())

    def reset_defaults(self) -> None:
        self.slider_h_min.setValue(15)
        self.slider_s_min.setValue(120)
        self.slider_v_min.setValue(150)
        self.slider_h_max.setValue(35)
        self.slider_s_max.setValue(255)
        self.slider_v_max.setValue(255)
        self.spin_required_frames.setValue(3)
        self.save_calibration()
