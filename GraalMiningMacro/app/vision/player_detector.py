"""Player Detection Module for Graal Mining Macro."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PlayerDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None         # (x, y)
    confidence: float = 0.0
    movement_delta: float = 0.0                      # Distance moved since last frame (px)
    matched_reference_name: str = ""
    matched_reference_confidence: float = 0.0

    def summary_text(self) -> str:
        if not self.detected or not self.center:
            return "PLAYER: UNKNOWN"
        if self.matched_reference_name:
            return f"PLAYER ({self.confidence * 100:.0f}%) REF: {self.matched_reference_name} X:{self.center[0]} Y:{self.center[1]}"
        return f"PLAYER ({self.confidence * 100:.0f}%) X:{self.center[0]} Y:{self.center[1]}"


class PlayerDetector:
    """Detects player character using reference library template matching with fallback to YOLO/Center heuristics."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self._last_center: Optional[Tuple[int, int]] = None

    def detect(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[list] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> PlayerDetection:
        if frame is None or frame.size == 0:
            self._last_center = None
            return PlayerDetection(detected=False, confidence=0.0)

        height, width = frame.shape[:2]

        # 1. Reference Template Matching
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="player", gray_image=gray_image, match_cache=match_cache
            )
            if ref_match and ref_match.found and ref_match.confidence >= self.confidence_threshold:
                center = ref_match.center
                delta = self._calc_delta(center)
                self._last_center = center
                return PlayerDetection(
                    detected=True,
                    bbox=ref_match.bbox,
                    center=center,
                    confidence=ref_match.confidence,
                    movement_delta=delta,
                    matched_reference_name=ref_match.reference_name,
                    matched_reference_confidence=ref_match.confidence,
                )

        # 2. Check YOLO Detections if available
        if yolo_detections:
            for det in yolo_detections:
                if getattr(det, "class_name", "") == "player" and det.confidence >= self.confidence_threshold:
                    center = det.center
                    delta = self._calc_delta(center)
                    self._last_center = center
                    return PlayerDetection(
                        detected=True,
                        bbox=det.bbox,
                        center=center,
                        confidence=det.confidence,
                        movement_delta=delta,
                    )

        # 3. Heuristic Game Center Player Detection (Standard Graal behavior: player centered)
        cx, cy = width // 2, height // 2
        pw, ph = 36, 48
        bbox = (cx - pw // 2, cy - ph // 2, pw, ph)
        center = (cx, cy)
        confidence = 0.85

        if confidence < self.confidence_threshold:
            self._last_center = None
            return PlayerDetection(detected=False, confidence=confidence)

        delta = self._calc_delta(center)
        self._last_center = center

        return PlayerDetection(
            detected=True,
            bbox=bbox,
            center=center,
            confidence=confidence,
            movement_delta=delta,
        )

    def _calc_delta(self, current_center: Tuple[int, int]) -> float:
        if not self._last_center:
            return 0.0
        dx = current_center[0] - self._last_center[0]
        dy = current_center[1] - self._last_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
