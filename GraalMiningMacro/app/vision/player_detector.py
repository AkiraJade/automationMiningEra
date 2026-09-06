"""Player Detection Module for Graal Mining Macro."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PlayerDetection:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    center: Optional[Tuple[int, int]] = None         # (x, y)
    raw_score: float = 0.0                           # Raw template match score
    confidence: float = 0.0                          # Final validated confidence
    movement_delta: float = 0.0                      # Distance moved since last frame (px)
    matched_reference_name: str = ""
    matched_subcategory: str = ""
    matched_reference_confidence: float = 0.0
    detection_method: str = "TEMPLATE"               # "TEMPLATE", "YOLO", "HEURISTIC"
    player_source: str = "NONE"                      # "REFERENCE", "YOLO", "HEURISTIC", "NONE"
    is_heuristic: bool = False
    rejection_reason: str = ""

    def summary_text(self) -> str:
        if not self.detected or not self.center:
            return "PLAYER: UNKNOWN"
        if self.is_heuristic or self.player_source == "HEURISTIC":
            return f"PLAYER (HEURISTIC) X:{self.center[0]} Y:{self.center[1]}"
        if self.matched_reference_name:
            pose_str = f" [{self.matched_subcategory.upper()}]" if self.matched_subcategory else ""
            return f"PLAYER (REF {self.confidence * 100:.0f}%) REF: {self.matched_reference_name}{pose_str} X:{self.center[0]} Y:{self.center[1]}"
        return f"PLAYER ({self.confidence * 100:.0f}%) [SOURCE:{self.player_source}] X:{self.center[0]} Y:{self.center[1]}"


class PlayerDetector:
    """Detects player character using reference library template matching with fallback to YOLO/Center heuristics."""

    def __init__(self, confidence_threshold: float = 0.65, allow_heuristic_fallback: bool = True):
        self.confidence_threshold = confidence_threshold
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self._last_center: Optional[Tuple[int, int]] = None

    def detect(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[list] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        world_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> PlayerDetection:
        if frame is None or frame.size == 0:
            self._last_center = None
            return PlayerDetection(detected=False, confidence=0.0)

        height, width = frame.shape[:2]

        # 1. REFERENCE TEMPLATE MATCHING (PRIMARY DETECTOR)
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="player", roi=world_roi, gray_image=gray_image, match_cache=match_cache
            )
            if ref_match and ref_match.found and ref_match.confidence >= self.confidence_threshold:
                center = ref_match.center
                delta = self._calc_delta(center)
                self._last_center = center
                return PlayerDetection(
                    detected=True,
                    bbox=ref_match.bbox,
                    center=center,
                    raw_score=getattr(ref_match, "raw_score", ref_match.confidence),
                    confidence=ref_match.confidence,
                    movement_delta=delta,
                    matched_reference_name=ref_match.reference_name,
                    matched_subcategory=ref_match.subcategory,
                    matched_reference_confidence=ref_match.confidence,
                    detection_method="TEMPLATE",
                    player_source="REFERENCE",
                    is_heuristic=False,
                )

        # 2. YOLO DETECTOR FALLBACK
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
                        raw_score=det.confidence,
                        confidence=det.confidence,
                        movement_delta=delta,
                        detection_method="YOLO",
                        player_source="YOLO",
                        is_heuristic=False,
                    )

        # 3. EXPLICIT CENTER HEURISTIC FALLBACK (Labeled clearly as HEURISTIC with low confidence 0.30)
        if self.allow_heuristic_fallback:
            cx, cy = width // 2, height // 2
            pw, ph = 36, 48
            bbox = (cx - pw // 2, cy - ph // 2, pw, ph)
            center = (cx, cy)
            heuristic_conf = 0.30  # Low confidence marker for unconfirmed heuristic fallback

            delta = self._calc_delta(center)
            self._last_center = center

            return PlayerDetection(
                detected=True,
                bbox=bbox,
                center=center,
                raw_score=0.30,
                confidence=heuristic_conf,
                movement_delta=delta,
                detection_method="HEURISTIC",
                player_source="HEURISTIC",
                is_heuristic=True,
            )

        self._last_center = None
        return PlayerDetection(detected=False, confidence=0.0, player_source="NONE")

    def _calc_delta(self, current_center: Tuple[int, int]) -> float:
        if not self._last_center:
            return 0.0
        dx = current_center[0] - self._last_center[0]
        dy = current_center[1] - self._last_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
