"""Unified Screen Capture Engine for Graal Mining Macro."""

import numpy as np
from typing import Optional
from PySide6.QtGui import QImage
from app.window.models import WindowInfo
from app.capture.capture_target import CaptureTarget
from app.capture.window_graphics_capture import WindowGraphicsCaptureBackend
from app.core.logger import setup_logger

logger = setup_logger("ScreenCaptureEngine")


class ScreenCaptureEngine:
    """Unified Capture Engine using WindowGraphicsCaptureBackend (Direct HWND PrintWindow)."""

    def __init__(self):
        self.backend = WindowGraphicsCaptureBackend()

    def capture_window_frame(self, window_info: WindowInfo) -> Optional[np.ndarray]:
        if not window_info or not window_info.is_valid:
            return None

        target = CaptureTarget.from_window_info(window_info)
        if not target:
            return None

        return self.backend.capture_target_frame(target)

    @staticmethod
    def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
        """Converts an OpenCV BGR numpy frame to PySide6 QImage cleanly."""
        height, width, channel = frame_bgr.shape
        bytes_per_line = channel * width
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        return qimg.copy()

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
