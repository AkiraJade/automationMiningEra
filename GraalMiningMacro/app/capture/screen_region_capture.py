"""Legacy MSS Screen Region Capture Backend for testing."""

import os
import mss
import numpy as np
from typing import Optional
from app.capture.base_backend import BaseCaptureBackend
from app.capture.capture_target import CaptureTarget
from app.core.logger import setup_logger

logger = setup_logger("ScreenRegionCaptureBackend")


class ScreenRegionCaptureBackend(BaseCaptureBackend):
    """Legacy screen region capture backend using MSS desktop coordinates."""

    def __init__(self):
        self._sct: Optional[mss.mss] = None
        self.own_pid = os.getpid()

    def _ensure_sct(self) -> None:
        if self._sct is None:
            self._sct = mss.mss()

    def capture_target_frame(self, target: CaptureTarget) -> Optional[np.ndarray]:
        if not target or not target.is_valid or target.is_minimized:
            return None

        if target.pid == self.own_pid or "graal mining macro" in target.title.lower():
            return None

        self._ensure_sct()
        left, top, right, bottom = target.client_rect
        w = right - left
        h = bottom - top

        if w <= 0 or h <= 0:
            return None

        monitor = {"top": int(top), "left": int(left), "width": int(w), "height": int(h)}

        try:
            sct_img = self._sct.grab(monitor)
            frame_bgra = np.array(sct_img, dtype=np.uint8)
            return frame_bgra[:, :, :3]
        except Exception as e:
            logger.error(f"MSS capture error: {e}")
            return None

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
