"""Game Status Infrastructure Detectors (Drill, Battery, Mine Location)."""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DrillState(Enum):
    EQUIPPED = "EQUIPPED"
    UNEQUIPPED = "UNEQUIPPED"
    UNKNOWN = "UNKNOWN"


class BatteryState(Enum):
    BATTERY_OK = "BATTERY_OK"
    BATTERY_LOW = "BATTERY_LOW"
    BATTERY_EMPTY = "BATTERY_EMPTY"
    BATTERY_UNKNOWN = "BATTERY_UNKNOWN"


class MineLocationState(Enum):
    INSIDE = "INSIDE"
    SURFACE = "SURFACE"
    ENTERING = "ENTERING"
    EXITING = "EXITING"
    COLLAPSED = "COLLAPSED"
    UNKNOWN = "UNKNOWN"


@dataclass
class StatusDetectionResult:
    drill_state: DrillState = DrillState.UNKNOWN
    battery_state: BatteryState = BatteryState.BATTERY_UNKNOWN
    mine_state: MineLocationState = MineLocationState.UNKNOWN
    drill_confidence: float = 0.0
    battery_confidence: float = 0.0
    mine_confidence: float = 0.0
    matched_reference_name: str = ""


from app.vision.temporal_filter import TemporalStateFilter


class StatusDetector:
    """Perception infrastructure for Drill, Battery, and Mine Location states with reference template matching."""

    def __init__(self, drill_margin: float = 0.10, confirm_frames: int = 3):
        self.drill_state_margin = drill_margin
        self.drill_filter = TemporalStateFilter[DrillState](required_frames=confirm_frames, default_state=DrillState.UNKNOWN)
        self.battery_filter = TemporalStateFilter[BatteryState](required_frames=confirm_frames, default_state=BatteryState.BATTERY_OK)
        self.mine_filter = TemporalStateFilter[MineLocationState](required_frames=confirm_frames, default_state=MineLocationState.UNKNOWN)

    def detect(
        self,
        frame: np.ndarray,
        player_detected: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
        reference_manager: Optional[object] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> StatusDetectionResult:
        if frame is None or frame.size == 0:
            return StatusDetectionResult()

        raw_drill_state = DrillState.UNKNOWN
        drill_conf = 0.0
        battery_state = BatteryState.BATTERY_OK
        battery_conf = 0.80
        mine_state = MineLocationState.UNKNOWN
        mine_conf = 0.0
        matched_ref_name = ""

        # 1. Reference Template Matching for STATUS Category
        if reference_manager is not None:
            # Mutually-exclusive drill state evaluation
            eq_ref = reference_manager.find_best_match(
                frame, category="status", subcategory="drill_equipped", roi=roi, gray_image=gray_image, match_cache=match_cache
            )
            uneq_ref = reference_manager.find_best_match(
                frame, category="status", subcategory="drill_unequipped", roi=roi, gray_image=gray_image, match_cache=match_cache
            )

            eq_score = eq_ref.raw_score if (eq_ref and eq_ref.found) else 0.0
            uneq_score = uneq_ref.raw_score if (uneq_ref and uneq_ref.found) else 0.0

            thresh = 0.70
            if eq_score >= thresh and eq_score > uneq_score + self.drill_state_margin:
                raw_drill_state = DrillState.EQUIPPED
                drill_conf = eq_score
                matched_ref_name = eq_ref.reference_name if eq_ref else ""
            elif uneq_score >= thresh and uneq_score > eq_score + self.drill_state_margin:
                raw_drill_state = DrillState.UNEQUIPPED
                drill_conf = uneq_score
                matched_ref_name = uneq_ref.reference_name if uneq_ref else ""
            else:
                raw_drill_state = DrillState.UNKNOWN
                drill_conf = 0.0

            # Check battery empty references
            bat_ref = reference_manager.find_best_match(
                frame, category="status", subcategory="battery_empty", roi=roi, gray_image=gray_image, match_cache=match_cache
            )
            if bat_ref and bat_ref.found:
                if not matched_ref_name:
                    matched_ref_name = bat_ref.reference_name
                battery_state = BatteryState.BATTERY_EMPTY
                battery_conf = bat_ref.confidence

            # Check mine state references (inside, surface, entrance, collapsed)
            mine_ref = reference_manager.find_best_match(
                frame, category="mine", roi=roi, gray_image=gray_image, match_cache=match_cache
            )
            if mine_ref and mine_ref.found:
                if not matched_ref_name:
                    matched_ref_name = mine_ref.reference_name
                mine_conf = mine_ref.confidence
                sub = (mine_ref.subcategory + mine_ref.reference_name).lower()
                if "collapsed" in sub:
                    mine_state = MineLocationState.COLLAPSED
                elif "surface" in sub:
                    mine_state = MineLocationState.SURFACE
                elif "entrance" in sub:
                    mine_state = MineLocationState.ENTERING
                elif "inside" in sub:
                    mine_state = MineLocationState.INSIDE

        # Fallback Mine Location Color Palette Analysis
        if mine_conf <= 0.5:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else frame
            avg_v = float(np.mean(hsv[:, :, 2])) if len(hsv.shape) == 3 else float(np.mean(hsv))
            if avg_v < 100:
                mine_state = MineLocationState.INSIDE
                mine_conf = 0.85
            else:
                mine_state = MineLocationState.SURFACE
                mine_conf = 0.75

        # Temporal State Filtering for Drill
        confirmed_drill, is_confirmed, _ = self.drill_filter.update(raw_drill_state)
        final_drill_state = confirmed_drill if confirmed_drill is not None else DrillState.UNKNOWN

        return StatusDetectionResult(
            drill_state=final_drill_state,
            battery_state=battery_state,
            mine_state=mine_state,
            drill_confidence=drill_conf,
            battery_confidence=battery_conf,
            mine_confidence=mine_conf,
            matched_reference_name=matched_ref_name,
        )
