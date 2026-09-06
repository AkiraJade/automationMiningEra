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
    drill_equipped: Optional[bool] = None            # True if equipped, False if unequipped, None if unknown
    detection_method: str = "TEMPLATE"               # "TEMPLATE", "YOLO", "HEURISTIC"
    player_source: str = "NONE"                      # "REFERENCE", "YOLO", "HEURISTIC", "NONE"
    player_state: str = "PLAYER_NOT_FOUND"           # "PLAYER_CONFIRMED", "PLAYER_UNCERTAIN", "PLAYER_NOT_FOUND"
    is_heuristic: bool = False
    facing_direction: str = "UNKNOWN"                # "LEFT", "RIGHT", "UP", "DOWN", "UNKNOWN"
    rejection_reason: str = ""

    def summary_text(self) -> str:
        if not self.detected or not self.center or self.player_state == "PLAYER_NOT_FOUND":
            return "PLAYER: UNKNOWN"
        if self.is_heuristic or self.player_source == "HEURISTIC":
            return f"PLAYER (HEURISTIC) X:{self.center[0]} Y:{self.center[1]}"
        if self.matched_reference_name:
            pose_str = f" [{self.matched_subcategory.upper()}]" if self.matched_subcategory else ""
            return f"PLAYER (REF {self.confidence * 100:.0f}%) REF: {self.matched_reference_name}{pose_str} X:{self.center[0]} Y:{self.center[1]}"
        return f"PLAYER ({self.confidence * 100:.0f}%) [SOURCE:{self.player_source}] X:{self.center[0]} Y:{self.center[1]}"


class PlayerDetector:
    """Detects player character using reference library template matching with fallback to YOLO/Center heuristics."""

    PLAYER_MIN_MATCH_SIZE: int = 12
    PLAYER_MAX_MATCH_SIZE: int = 120

    def __init__(self, confidence_threshold: float = 0.58, allow_heuristic_fallback: bool = True):
        self.confidence_threshold = confidence_threshold
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self._last_center: Optional[Tuple[int, int]] = None
        self._locked_scale: float = 1.0

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
            return PlayerDetection(detected=False, confidence=0.0, player_state="PLAYER_NOT_FOUND")

        height, width = frame.shape[:2]

        # 1. REFERENCE TEMPLATE MATCHING (PRIMARY DETECTOR)
        if reference_manager is not None:
            ref_match = None

            # Fast local tracking window if player was previously detected
            if self._last_center is not None:
                lcx, lcy = self._last_center
                track_w, track_h = 320, 320
                track_x = max(0, min(lcx - track_w // 2, width - track_w))
                track_y = max(0, min(lcy - track_h // 2, height - track_h))
                track_roi = (track_x, track_y, min(width - track_x, track_w), min(height - track_y, track_h))
                cand_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=track_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=[self._locked_scale],
                    use_core=True,
                )
                if cand_match and cand_match.found and cand_match.confidence >= self.confidence_threshold:
                    ref_match = cand_match

            # Central viewport search prior (Graal camera keeps player centered in cave)
            if ref_match is None:
                cx, cy = width // 2, height // 2
                cw, ch = int(width * 0.50), int(height * 0.55)
                central_roi = (max(0, cx - cw // 2), max(0, cy - ch // 2), min(width, cw), min(height, ch))

                # Primary scales: locked scale and native 1.0
                primary_scales = [self._locked_scale] if abs(self._locked_scale - 1.0) < 1e-3 else [self._locked_scale, 1.0]

                cand_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=central_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=primary_scales,
                    use_core=True,
                )
                if cand_match and cand_match.found and cand_match.confidence >= self.confidence_threshold:
                    ref_match = cand_match
                    if getattr(cand_match, "scale", 1.0) > 0.1:
                        self._locked_scale = float(cand_match.scale)
                elif self._last_center is None:
                    # Secondary fallback scales if primary missed and lost
                    secondary_scales = [1.05, 0.95]
                    cand_match2 = reference_manager.find_best_match(
                        frame,
                        category="player",
                        roi=central_roi,
                        gray_image=gray_image,
                        match_cache=match_cache,
                        candidate_scales=secondary_scales,
                        use_core=True,
                    )
                    if cand_match2 and cand_match2.found and cand_match2.confidence >= self.confidence_threshold:
                        ref_match = cand_match2
                        if getattr(cand_match2, "scale", 1.0) > 0.1:
                            self._locked_scale = float(cand_match2.scale)

            # Fall back to full world_roi search if tracking missed or lost
            if ref_match is None and world_roi is not None:
                ref_match = reference_manager.find_best_match(
                    frame,
                    category="player",
                    roi=world_roi,
                    gray_image=gray_image,
                    match_cache=match_cache,
                    candidate_scales=[self._locked_scale, 1.0],
                    use_core=True,
                )

            if ref_match and ref_match.found and ref_match.confidence >= self.confidence_threshold:
                bw, bh = ref_match.bbox[2], ref_match.bbox[3] if ref_match.bbox else (0, 0)
                if self.PLAYER_MIN_MATCH_SIZE <= bw <= self.PLAYER_MAX_MATCH_SIZE and self.PLAYER_MIN_MATCH_SIZE <= bh <= self.PLAYER_MAX_MATCH_SIZE:
                    center = ref_match.center
                    delta = self._calc_delta(center)
                    self._last_center = center

                    # Determine drill equipment evidence & facing direction directly from player reference template
                    ref_name_lower = ref_match.reference_name.lower()
                    sub_lower = ref_match.subcategory.lower()

                    drill_equipped = None
                    if "drillequiped" in ref_name_lower or ref_match.subcategory == "mining":
                        drill_equipped = True
                    elif ref_match.reference_name in ["LEFT", "DOWN", "RIGHT", "UP"] or ref_match.subcategory in ["left", "down", "right", "up"]:
                        drill_equipped = False

                    facing = "UNKNOWN"
                    if "left" in ref_name_lower or "left" in sub_lower:
                        facing = "LEFT"
                    elif "right" in ref_name_lower or "right" in sub_lower:
                        facing = "RIGHT"
                    elif "up" in ref_name_lower or "up" in sub_lower:
                        facing = "UP"
                    elif "down" in ref_name_lower or "down" in sub_lower:
                        facing = "DOWN"

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
                        drill_equipped=drill_equipped,
                        facing_direction=facing,
                        detection_method="TEMPLATE",
                        player_source="REFERENCE",
                        player_state="PLAYER_CONFIRMED",
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
                        player_state="PLAYER_CONFIRMED",
                        is_heuristic=False,
                    )

        # 3. EXPLICIT CENTER HEURISTIC FALLBACK (Labeled clearly as HEURISTIC with low confidence 0.30, NOT detected)
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
                player_state="PLAYER_UNCERTAIN",
                is_heuristic=True,
            )

        self._last_center = None
        return PlayerDetection(detected=False, confidence=0.0, player_source="NONE", player_state="PLAYER_NOT_FOUND")

    def _calc_delta(self, current_center: Tuple[int, int]) -> float:
        if not self._last_center:
            return 0.0
        dx = current_center[0] - self._last_center[0]
        dy = current_center[1] - self._last_center[1]
        return float(np.sqrt(dx * dx + dy * dy))
