"""Spider Threat Detector Module for Graal Mining Macro."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class SpiderDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    distance_from_player: float = 0.0
    matched_reference_name: str = ""

    def summary_text(self) -> str:
        if not self.detected:
            return "SPIDER: NONE"
        ref_text = f" REF:{self.matched_reference_name}" if self.matched_reference_name else ""
        return f"🚨 SPIDER DETECTED ({self.confidence * 100:.0f}%){ref_text} DIST: {self.distance_from_player:.0f}px"


class SpiderDetector:
    """Detects spider threats using reference template matching, spatial distance validation, and color fallback."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        frame: np.ndarray,
        player_center: Optional[Tuple[int, int]] = None,
        yolo_detections: Optional[List] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> SpiderDetection:
        if frame is None or frame.size == 0:
            return SpiderDetection(detected=False)

        # 1. Reference Template Matching
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="spider", gray_image=gray_image, match_cache=match_cache
            )
            if ref_match and ref_match.found and ref_match.confidence >= self.confidence_threshold:
                center = ref_match.center
                dist = self._calc_dist(center, player_center)
                return SpiderDetection(
                    detected=True,
                    bbox=ref_match.bbox,
                    center=center,
                    confidence=ref_match.confidence,
                    distance_from_player=dist,
                    matched_reference_name=ref_match.reference_name,
                )

        # 2. Check YOLO Detections first if available
        if yolo_detections:
            for det in yolo_detections:
                if getattr(det, "class_name", "") == "spider" and det.confidence >= self.confidence_threshold:
                    center = det.center
                    dist = self._calc_dist(center, player_center)
                    return SpiderDetection(
                        detected=True,
                        bbox=det.bbox,
                        center=center,
                        confidence=det.confidence,
                        distance_from_player=dist,
                    )

        # 3. Color / Feature Fallback Spider Detector (Dark reddish/black small fast contour)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Spider red/dark mask
        lower_red = np.array([0, 100, 50], dtype=np.uint8)
        upper_red = np.array([10, 255, 200], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 <= area <= 300:
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w // 2, y + h // 2)
                dist = self._calc_dist(center, player_center)
                conf = min(1.0, area / 150.0)

                if conf >= self.confidence_threshold:
                    return SpiderDetection(
                        detected=True,
                        bbox=(x, y, w, h),
                        center=center,
                        confidence=conf,
                        distance_from_player=dist,
                    )

        return SpiderDetection(detected=False)

    @staticmethod
    def _calc_dist(spider_center: Tuple[int, int], player_center: Optional[Tuple[int, int]]) -> float:
        if not player_center:
            return 0.0
        dx = spider_center[0] - player_center[0]
        dy = spider_center[1] - player_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
