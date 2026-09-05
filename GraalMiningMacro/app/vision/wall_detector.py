"""Mining Wall Detection Module for Graal Mining Macro."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class WallDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    direction: str = "UNKNOWN"  # "LEFT", "RIGHT", "TOP", "BOTTOM", "UNKNOWN"
    distance_px: float = 0.0
    confidence: float = 0.0

    def summary_text(self) -> str:
        if not self.detected:
            return "WALL: UNKNOWN"
        return f"WALL: {self.direction} ({self.distance_px:.0f}px)"


class WallDetector:
    """Detects mine wall regions, relative orientation to player, and pixel distance."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray, player_center: Optional[Tuple[int, int]] = None) -> WallDetection:
        if frame is None or frame.size == 0 or not player_center:
            return WallDetection(detected=False, direction="UNKNOWN", confidence=0.0)

        height, width = frame.shape[:2]
        px, py = player_center

        # Convert to grayscale and threshold dark rock wall contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Mine wall dark tile thresholding
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_wall_bbox = None
        min_dist = float("inf")
        best_direction = "UNKNOWN"
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 200:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2

                dx = cx - px
                dy = cy - py
                dist = float(np.sqrt(dx * dx + dy * dy))

                # Determine relative direction
                if dist < min_dist and dist > 10:
                    min_dist = dist
                    best_wall_bbox = (x, y, w, h)
                    best_conf = min(1.0, area / 1000.0)

                    if abs(dx) > abs(dy):
                        best_direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        best_direction = "BOTTOM" if dy > 0 else "TOP"

        if best_conf < self.confidence_threshold or not best_wall_bbox:
            return WallDetection(detected=False, direction="UNKNOWN", confidence=0.0)

        return WallDetection(
            detected=True,
            bbox=best_wall_bbox,
            direction=best_direction,
            distance_px=min_dist,
            confidence=best_conf,
        )
