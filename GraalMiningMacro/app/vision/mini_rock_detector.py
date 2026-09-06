"""Mini-Rock Drop Detector Module for Graal Mining Macro."""

import cv2
import numpy as np
from typing import Tuple, Optional
from app.mining.scene_model import MiniRockDetection
from app.core.logger import setup_logger

logger = setup_logger("MiniRockDetector")


class MiniRockDetector:
    """Detects small rock/pebble drops behind or adjacent to the player upon successful hits."""

    def __init__(self, confidence_threshold: float = 0.50, confirm_frames: int = 2):
        self.confidence_threshold = confidence_threshold
        self.confirm_frames = confirm_frames
        self._consecutive_count: int = 0
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None

    def detect(
        self,
        frame: np.ndarray,
        player_center: Optional[Tuple[int, int]] = None,
        facing_direction: str = "UNKNOWN",
        world_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> MiniRockDetection:
        if frame is None or frame.size == 0 or not player_center:
            self._consecutive_count = 0
            return MiniRockDetection(detected=False, state="MINI_ROCK_NONE")

        px, py = player_center
        h, w = frame.shape[:2]

        # Calculate search ROI behind the player relative to facing direction
        offset_x, offset_y = 0, 0
        if facing_direction == "LEFT":
            offset_x, offset_y = 20, -10  # Behind player (to the right)
        elif facing_direction == "RIGHT":
            offset_x, offset_y = -45, -10 # Behind player (to the left)
        elif facing_direction == "UP":
            offset_x, offset_y = -15, 20  # Behind player (below)
        elif facing_direction == "DOWN":
            offset_x, offset_y = -15, -45 # Behind player (above)

        if offset_x == 0 and offset_y == 0:
            # Fallback search ring surrounding player
            rx1 = max(0, px - 40)
            ry1 = max(0, py - 40)
            rw = min(w - rx1, 80)
            rh = min(h - ry1, 80)
        else:
            rx1 = max(0, px + offset_x)
            ry1 = max(0, py + offset_y)
            rw = min(w - rx1, 35)
            rh = min(h - ry1, 35)

        crop = frame[ry1:ry1+rh, rx1:rx1+rw]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            self._consecutive_count = 0
            return MiniRockDetection(detected=False, state="MINI_ROCK_NONE")

        # Scan for small high-contrast greyish pebble contours (size 4x4 to 16x16)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_bbox = None
        found_conf = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if 8 <= area <= 160:
                bx, by, bw, bh = cv2.boundingRect(c)
                if 3 <= bw <= 20 and 3 <= bh <= 20:
                    found_bbox = (rx1 + bx, ry1 + by, bw, bh)
                    found_conf = min(0.90, 0.40 + area / 150.0)
                    break

        if not found_bbox or found_conf < self.confidence_threshold:
            self._consecutive_count = max(0, self._consecutive_count - 1)
            if self._consecutive_count > 0 and self._last_bbox:
                return MiniRockDetection(
                    detected=False,
                    is_candidate=True,
                    consecutive_frames=self._consecutive_count,
                    bbox=self._last_bbox,
                    center=(self._last_bbox[0] + self._last_bbox[2] // 2, self._last_bbox[1] + self._last_bbox[3] // 2),
                    confidence=0.40,
                    state="MINI_ROCK_CANDIDATE"
                )
            return MiniRockDetection(detected=False, state="MINI_ROCK_NONE")

        self._consecutive_count += 1
        self._last_bbox = found_bbox
        center = (found_bbox[0] + found_bbox[2] // 2, found_bbox[1] + found_bbox[3] // 2)

        is_confirmed = (self._consecutive_count >= self.confirm_frames)
        state_str = "MINI_ROCK_CONFIRMED" if is_confirmed else "MINI_ROCK_CANDIDATE"

        return MiniRockDetection(
            detected=is_confirmed,
            is_candidate=not is_confirmed,
            consecutive_frames=self._consecutive_count,
            bbox=found_bbox,
            center=center,
            confidence=found_conf if is_confirmed else 0.45,
            state=state_str
        )
