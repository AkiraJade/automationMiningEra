"""Spider Threat Detector Module for Graal Mining Macro."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class SpiderDetection:
    detected: bool = False
    is_candidate: bool = False
    consecutive_frames: int = 0
    required_frames: int = 2
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    raw_score: float = 0.0
    confidence: float = 0.0                          # Final validated confidence
    distance_from_player: float = 0.0
    has_player_distance: bool = False
    matched_reference_name: str = ""
    rejection_reason: str = ""
    detection_method: str = "TEMPLATE"               # "TEMPLATE", "YOLO", "COLOR"

    def summary_text(self) -> str:
        if not self.detected and not self.is_candidate:
            return "SPIDER: NONE"
        ref_text = f" REF:{self.matched_reference_name}" if self.matched_reference_name else ""
        dist_text = f" DIST:{self.distance_from_player:.0f}px" if self.has_player_distance else " DIST:UNKNOWN"

        if self.is_candidate and not self.detected:
            return f"SPIDER CANDIDATE ({self.consecutive_frames}/{self.required_frames}){ref_text}{dist_text}"
        return f"🚨 SPIDER DETECTED ({self.confidence * 100:.0f}%){ref_text}{dist_text}"


class SpiderDetector:
    """Detects spider threats using multi-factor template validation, temporal confirmation, and spatial distance checks."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        confirm_frames: int = 2,
        min_size: Tuple[int, int] = (12, 12),
        max_size: Tuple[int, int] = (120, 120),
        max_distance: float = 800.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.confirm_frames = confirm_frames
        self.min_size = min_size
        self.max_size = max_size
        self.max_distance = max_distance

        self._consecutive_count: int = 0
        self._last_candidate_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_candidate_center: Optional[Tuple[int, int]] = None

    def detect(
        self,
        frame: np.ndarray,
        player_center: Optional[Tuple[int, int]] = None,
        yolo_detections: Optional[List] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        world_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> SpiderDetection:
        if frame is None or frame.size == 0:
            self._consecutive_count = 0
            return SpiderDetection(detected=False)

        raw_match = None
        cand_bbox = None
        cand_center = None
        cand_raw_score = 0.0
        ref_name = ""
        method = "TEMPLATE"
        rejection_reason = ""

        # 1. Reference Template Matching
        if reference_manager is not None:
            ref_match = reference_manager.find_best_match(
                frame, category="spider", roi=world_roi, gray_image=gray_image, match_cache=match_cache
            )
            if ref_match and ref_match.found:
                raw_match = ref_match
                cand_raw_score = getattr(ref_match, "raw_score", ref_match.confidence)
                ref_name = ref_match.reference_name
                cand_bbox = ref_match.bbox
                cand_center = ref_match.center
                method = "TEMPLATE"

        # 2. YOLO Detections Fallback
        if not cand_bbox and yolo_detections:
            for det in yolo_detections:
                if getattr(det, "class_name", "") == "spider" and det.confidence >= self.confidence_threshold:
                    cand_raw_score = det.confidence
                    cand_bbox = det.bbox
                    cand_center = det.center
                    method = "YOLO"
                    break

        # 3. HSV Color / Feature Fallback (Strict bounds & size check)
        if not cand_bbox:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 120, 60], dtype=np.uint8)
            upper_red = np.array([10, 255, 200], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_red, upper_red)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 20 <= area <= 250:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if self.min_size[0] <= w <= self.max_size[0] and self.min_size[1] <= h <= self.max_size[1]:
                        cand_bbox = (x, y, w, h)
                        cand_center = (x + w // 2, y + h // 2)
                        cand_raw_score = min(0.95, area / 150.0)
                        method = "COLOR"
                        break

        # Multi-factor validation checks
        if not cand_bbox or cand_raw_score < self.confidence_threshold:
            self._consecutive_count = 0
            return SpiderDetection(detected=False, rejection_reason="LOW_CONFIDENCE" if cand_raw_score > 0 else "")

        # Size validation
        bw, bh = cand_bbox[2], cand_bbox[3]
        if not (self.min_size[0] <= bw <= self.max_size[0] and self.min_size[1] <= bh <= self.max_size[1]):
            self._consecutive_count = 0
            return SpiderDetection(detected=False, raw_score=cand_raw_score, rejection_reason="SIZE_INVALID")

        # Spatial distance to player validation
        has_dist = False
        dist = 0.0
        if player_center and player_center[0] > 0 and player_center[1] > 0:
            dist = self._calc_dist(cand_center, player_center)
            has_dist = True
            if dist > self.max_distance:
                self._consecutive_count = 0
                return SpiderDetection(detected=False, raw_score=cand_raw_score, rejection_reason="DISTANCE_TOO_FAR")

        # Candidate confirmed for current frame -> increment temporal count
        self._consecutive_count += 1
        self._last_candidate_bbox = cand_bbox
        self._last_candidate_center = cand_center

        is_confirmed = (self._consecutive_count >= self.confirm_frames)
        # Compute final confidence from raw score & temporal stability
        final_conf = min(1.0, cand_raw_score * (0.85 + 0.15 * min(1.0, self._consecutive_count / float(self.confirm_frames))))

        return SpiderDetection(
            detected=is_confirmed,
            is_candidate=not is_confirmed,
            consecutive_frames=self._consecutive_count,
            required_frames=self.confirm_frames,
            bbox=cand_bbox,
            center=cand_center,
            raw_score=cand_raw_score,
            confidence=final_conf,
            distance_from_player=dist,
            has_player_distance=has_dist,
            matched_reference_name=ref_name,
            detection_method=method,
        )

    @staticmethod
    def _calc_dist(spider_center: Tuple[int, int], player_center: Optional[Tuple[int, int]]) -> float:
        if not player_center:
            return 0.0
        dx = spider_center[0] - player_center[0]
        dy = spider_center[1] - player_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
