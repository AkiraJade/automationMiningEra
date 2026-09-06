"""Live Game Preview Widget with Enhanced Debug Overlay & Frame Freeze for Graal Mining Macro."""

import os
import time
import cv2
import numpy as np
from typing import Optional
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor, QPen, QFont
from app.capture.screen_capture import ScreenCaptureEngine
from app.mining.mining_perception import MiningPerceptionResult
from app.window.models import WindowInfo
from app.core.logger import setup_logger

logger = setup_logger("GamePreviewWidget")


class GamePreviewWidget(QWidget):
    """Centerpiece Live Game Display Widget with aspect ratio scaling, debug visual overlay, and frame freeze."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background-color: #121212; border: 1px solid #333333; border-radius: 4px;")

        self._current_frame: Optional[np.ndarray] = None
        self._current_qimage: Optional[QImage] = None
        self._window_info: Optional[WindowInfo] = None
        self._perception_result: Optional[MiningPerceptionResult] = None

        self.debug_overlay_enabled: bool = True
        self.is_frozen: bool = False
        self.current_fps: float = 0.0

    def set_frozen(self, frozen: bool) -> None:
        self.is_frozen = frozen
        self.update()

    def save_diagnostic_frame(self, filepath: Optional[str] = None) -> str:
        """Saves diagnostic bundle: raw.png, debug.png, perception.json, and timings.json to scratch directory."""
        if self._current_frame is None:
            logger.warning("Cannot save diagnostic frame: No active frame captured.")
            return ""

        try:
            import json
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            folder = os.path.abspath(f"scratch/diagnostic_{timestamp}")
            os.makedirs(folder, exist_ok=True)

            raw_path = os.path.join(folder, "raw.png")
            cv2.imwrite(raw_path, self._current_frame)

            # Render debug overlay on copy
            debug_img = self._current_frame.copy()
            p = self._perception_result
            if p is not None:
                if p.player.detected and p.player.bbox:
                    x, y, w, h = p.player.bbox
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 229, 0), 2)
                    cv2.putText(debug_img, f"PLAYER ({p.player.player_state})", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 229, 0), 1)
                if p.wall.detected and p.wall.bbox:
                    x, y, w, h = p.wall.bbox
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (3, 255, 118), 2)
                    cv2.putText(debug_img, f"WALL: {p.wall.direction}", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 255, 118), 1)
                if p.yellow_glow.is_confirmed and p.yellow_glow.bbox:
                    x, y, w, h = p.yellow_glow.bbox
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 234, 255), 3)
                    cv2.putText(debug_img, "ROCK COMPLETED", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 234, 255), 2)
                elif p.target.detected and p.target.bbox:
                    x, y, w, h = p.target.bbox
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 145, 255), 2)
                    cv2.putText(debug_img, f"TARGET ({p.target.target_state})", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 145, 255), 1)
                if p.spider.detected and p.spider.bbox:
                    x, y, w, h = p.spider.bbox
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (68, 23, 255), 3)
                    cv2.putText(debug_img, "SPIDER", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (68, 23, 255), 2)

            debug_path = os.path.join(folder, "debug.png")
            cv2.imwrite(debug_path, debug_img)

            perc_json_path = os.path.join(folder, "perception.json")
            perc_dict = {
                "player": {
                    "state": getattr(p.player, "player_state", "UNKNOWN") if p else "UNKNOWN",
                    "confidence": p.player.confidence if p else 0.0,
                    "bbox": p.player.bbox if p else None,
                    "center": p.player.center if p else None,
                    "source": p.player.player_source if p else "NONE",
                },
                "wall": {
                    "detected": p.wall.detected if p else False,
                    "direction": p.wall.direction if p else "UNKNOWN",
                    "distance_px": p.wall.distance_px if p else 0.0,
                    "bbox": p.wall.bbox if p else None,
                },
                "drill": {
                    "state": p.status.drill_state.name if (p and hasattr(p.status.drill_state, "name")) else "UNKNOWN",
                    "confidence": p.status.drill_confidence if p else 0.0,
                },
                "target": {
                    "detected": p.target.detected if p else False,
                    "state": getattr(p.target, "target_state", "NO_TARGET") if p else "NO_TARGET",
                    "iteration": p.target.iteration if p else 0,
                    "bbox": p.target.bbox if p else None,
                },
                "yellow_rock": {
                    "is_confirmed": p.yellow_glow.is_confirmed if p else False,
                    "confidence": p.yellow_glow.confidence if p else 0.0,
                    "bbox": p.yellow_glow.bbox if p else None,
                },
                "spider": {
                    "detected": p.spider.detected if p else False,
                    "confidence": p.spider.confidence if p else 0.0,
                    "bbox": p.spider.bbox if p else None,
                },
                "message": {
                    "nothing_to_mine_detected": p.message.nothing_to_mine_detected if p else False,
                    "cooldown_remaining": p.message.cooldown_remaining if p else 0.0,
                }
            }
            with open(perc_json_path, "w") as f:
                json.dump(perc_dict, f, indent=2)

            timings_path = os.path.join(folder, "timings.json")
            with open(timings_path, "w") as f:
                json.dump(p.detector_timings if p else {}, f, indent=2)

            logger.info(f"Saved diagnostic bundle to '{folder}'")
            return folder
        except Exception as e:
            logger.error(f"Failed to save diagnostic bundle: {e}")
            return ""

    def update_frame(self, frame_bgr: Optional[np.ndarray], window_info: Optional[WindowInfo] = None) -> None:
        if self.is_frozen:
            return  # Retain existing frozen frame

        self._current_frame = frame_bgr
        self._window_info = window_info
        if frame_bgr is not None and window_info is not None and window_info.is_valid:
            self._current_qimage = ScreenCaptureEngine.bgr_to_qimage(frame_bgr)
        else:
            self._current_qimage = None
        self.update()

    def update_perception(self, perception: MiningPerceptionResult) -> None:
        if self.is_frozen:
            return  # Retain frozen perception
        self._perception_result = perception
        self.update()

    def set_fps(self, fps: float) -> None:
        self.current_fps = fps
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w = self.width()
        widget_h = self.height()

        # Handle Minimized Window case
        if self._window_info is not None and self._window_info.is_minimized:
            painter.fillRect(self.rect(), QColor("#101014"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FFD600"))
            painter.drawEllipse(widget_w // 2 - 8, widget_h // 2 - 50, 16, 16)

            painter.setPen(QColor("#FFD600"))
            font_title = QFont("Segoe UI", 13, QFont.Weight.Bold)
            painter.setFont(font_title)
            painter.drawText(
                QRect(0, widget_h // 2 - 20, widget_w, 30),
                Qt.AlignmentFlag.AlignCenter,
                "CAPTURE UNAVAILABLE — WINDOW MINIMIZED"
            )

            painter.setPen(QColor("#888888"))
            font_sub = QFont("Segoe UI", 10)
            painter.setFont(font_sub)
            painter.drawText(
                QRect(0, widget_h // 2 + 15, widget_w, 30),
                Qt.AlignmentFlag.AlignCenter,
                "[ Restore GraalOnline Era window from taskbar to resume capture ]"
            )
            return

        # Handle Disconnected / No Frame case
        if self._current_qimage is None or self._current_qimage.isNull() or self._window_info is None or not self._window_info.is_valid:
            painter.fillRect(self.rect(), QColor("#101014"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FF1744"))
            painter.drawEllipse(widget_w // 2 - 8, widget_h // 2 - 50, 16, 16)

            painter.setPen(QColor("#FF1744"))
            font_title = QFont("Segoe UI", 13, QFont.Weight.Bold)
            painter.setFont(font_title)
            painter.drawText(
                QRect(0, widget_h // 2 - 20, widget_w, 30),
                Qt.AlignmentFlag.AlignCenter,
                "GAME DISCONNECTED — NOT FOUND"
            )

            painter.setPen(QColor("#888888"))
            font_sub = QFont("Segoe UI", 10)
            painter.setFont(font_sub)
            painter.drawText(
                QRect(0, widget_h // 2 + 15, widget_w, 30),
                Qt.AlignmentFlag.AlignCenter,
                "[ Ensure Graal Online Era window is open & matching pattern in Settings ]"
            )
            return

        # Calculate aspect ratio scaling
        img_w = self._current_qimage.width()
        img_h = self._current_qimage.height()

        scale = min(widget_w / img_w, widget_h / img_h)
        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        target_x = (widget_w - target_w) // 2
        target_y = (widget_h - target_h) // 2

        target_rect = QRect(target_x, target_y, target_w, target_h)

        # Draw black background padding
        painter.fillRect(self.rect(), QColor("#000000"))

        # Draw captured game image scaled cleanly
        painter.drawImage(target_rect, self._current_qimage)

        # Freeze Frame Banner Watermark
        if self.is_frozen:
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor("#00E5FF"))
            painter.drawText(game_rect.left() + 15, game_rect.top() + 25, "❄️ FRAME FROZEN (INSPECTION MODE)")

        # Draw Debug Overlay if enabled
        if self.debug_overlay_enabled:
            self._draw_debug_overlay(painter, target_rect, scale)

    def _draw_debug_overlay(self, painter: QPainter, game_rect: QRect, scale: float) -> None:
        gx = game_rect.x()
        gy = game_rect.y()

        # Draw FPS watermark top-right
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#00FF66"))
        fps_text = f"FPS: {self.current_fps:.1f}"
        painter.drawText(game_rect.right() - 100, game_rect.top() + 25, fps_text)

        p = self._perception_result
        if p is None:
            return

        # 1. Player Detection (Cyan for TEMPLATE match, Dark Cyan/Dash for HEURISTIC)
        if p.player.detected and p.player.bbox and p.player.center:
            px, py, pw, ph = p.player.bbox
            sx = int(gx + px * scale)
            sy = int(gy + py * scale)
            sw = int(pw * scale)
            sh = int(ph * scale)

            method_str = getattr(p.player, "detection_method", "TEMPLATE")
            if method_str == "HEURISTIC":
                painter.setPen(QPen(QColor("#00B0FF"), 1, Qt.PenStyle.DashLine))
                tag = "PLAYER (HEURISTIC)"
            else:
                painter.setPen(QPen(QColor("#00E5FF"), 2, Qt.PenStyle.SolidLine))
                ref_name = getattr(p.player, "matched_reference_name", "")
                tag = f"PLAYER ({p.player.confidence * 100:.0f}%) REF:{ref_name} RAW:{p.player.raw_score:.2f}"

            painter.drawRect(sx, sy, sw, sh)
            painter.drawText(sx, sy - 5, f"{tag} X:{p.player.center[0]} Y:{p.player.center[1]}")

        # 2. Wall Detection (Green Outline + Arrow)
        if p.wall.detected and p.wall.bbox:
            wx, wy, ww, wh = p.wall.bbox
            sx = int(gx + wx * scale)
            sy = int(gy + wy * scale)
            sw = int(ww * scale)
            sh = int(wh * scale)

            painter.setPen(QPen(QColor("#76FF03"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(sx, sy, sw, sh)
            painter.drawText(sx, sy - 5, f"WALL: {p.wall.direction} ({p.wall.distance_px:.0f}px)")

        # 3. Yellow Glow Completed Rock (Bright Yellow)
        if p.yellow_glow.is_confirmed and p.yellow_glow.bbox:
            tx, ty, tw, th = p.yellow_glow.bbox
            sx = int(gx + tx * scale)
            sy = int(gy + ty * scale)
            sw = int(tw * scale)
            sh = int(th * scale)

            painter.setPen(QPen(QColor("#FFEA00"), 3, Qt.PenStyle.SolidLine))
            painter.drawRect(sx, sy, sw, sh)
            ref_name = getattr(p.yellow_glow, "matched_reference_name", "")
            ref_tag = f" REF:{ref_name}" if ref_name else ""
            painter.drawText(sx, sy - 5, f"★ ROCK COMPLETED ({p.yellow_glow.confidence * 100:.0f}%){ref_tag} RAW:{p.yellow_glow.raw_score:.2f}")

        # 4. Target Rock (Orange)
        elif p.target.detected and p.target.bbox:
            tx, ty, tw, th = p.target.bbox
            sx = int(gx + tx * scale)
            sy = int(gy + ty * scale)
            sw = int(tw * scale)
            sh = int(th * scale)

            painter.setPen(QPen(QColor("#FF9100"), 2))
            painter.drawRect(sx, sy, sw, sh)
            painter.drawText(sx, sy - 5, f"TARGET (ITER {p.target.iteration}/3)")

        # 5. Spider Threat (Bright Red if confirmed, Orange/Dash if candidate)
        if p.spider.bbox:
            sx = int(gx + p.spider.bbox[0] * scale)
            sy = int(gy + p.spider.bbox[1] * scale)
            sw = int(p.spider.bbox[2] * scale)
            sh = int(p.spider.bbox[3] * scale)
            ref_name = getattr(p.spider, "matched_reference_name", "")
            ref_tag = f" REF:{ref_name}" if ref_name else ""

            if p.spider.detected:
                painter.setPen(QPen(QColor("#FF1744"), 3, Qt.PenStyle.SolidLine))
                painter.drawRect(sx, sy, sw, sh)
                painter.drawText(sx, sy - 5, f"🚨 SPIDER DETECTED ({p.spider.confidence * 100:.0f}%){ref_tag} RAW:{p.spider.raw_score:.2f}")
            elif getattr(p.spider, "is_candidate", False):
                painter.setPen(QPen(QColor("#FF9100"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(sx, sy, sw, sh)
                painter.drawText(sx, sy - 5, f"SPIDER CANDIDATE ({p.spider.consecutive_frames}/{p.spider.required_frames}){ref_tag} RAW:{p.spider.raw_score:.2f}")

        # 6. Nothing to Mine Message Banner
        if p.message.nothing_to_mine_detected:
            painter.setPen(QColor("#FFD600"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            ref_tag = f" REF: {p.message.matched_reference_name}" if p.message.matched_reference_name else ""
            painter.drawText(game_rect.left() + 20, game_rect.bottom() - 30, f"⚠️ 'NOTHING TO MINE HERE'{ref_tag} (COOLDOWN: {p.message.cooldown_remaining:.1f}s)")
