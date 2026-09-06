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

    def __init__(self, confidence_threshold: float = 0.50, max_player_distance: float = 120.0):
        self.confidence_threshold = confidence_threshold
        self.max_player_distance = max_player_distance

    def detect(
        self,
        frame: np.ndarray,
        player_center: Optional[Tuple[int, int]] = None,
        world_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> WallDetection:
        if frame is None or frame.size == 0 or not player_center:
            return WallDetection(detected=False, direction="UNKNOWN", confidence=0.0)

        height, width = frame.shape[:2]
        px, py = player_center

        # Define vertical and horizontal boundary safety margins (ignore top/bottom window borders and HUD)
        min_y = int(height * 0.05) if world_roi is None else world_roi[1]
        max_y = int(height * 0.95) if world_roi is None else (world_roi[1] + world_roi[3])
        min_x = int(width * 0.02) if world_roi is None else world_roi[0]
        max_x = int(width * 0.98) if world_roi is None else (world_roi[0] + world_roi[2])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        # Mine wall dark tile thresholding
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_wall_bbox = None
        min_dist = float("inf")
        best_direction = "UNKNOWN"
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 <= area <= 12000:
                x, y, w, h = cv2.boundingRect(cnt)

                # Exclude contours that touch top/bottom window borders or HUD lines
                if y < min_y or (y + h) > max_y or x < min_x or (x + w) > max_x:
                    continue

                cx = x + w // 2
                cy = y + h // 2

                dx = cx - px
                dy = cy - py
                dist = float(np.sqrt(dx * dx + dy * dy))

                # Determine relative direction within player proximity limit
                if dist <= self.max_player_distance and dist < min_dist and dist > 10:
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
