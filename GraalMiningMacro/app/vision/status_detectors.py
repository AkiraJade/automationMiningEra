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


class StatusDetector:
    """Perception infrastructure for Drill, Battery, and Mine Location states with reference template matching."""

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

        drill_state = DrillState.EQUIPPED if player_detected else DrillState.UNKNOWN
        drill_conf = 0.85 if player_detected else 0.0
        battery_state = BatteryState.BATTERY_OK
        battery_conf = 0.80
        mine_state = MineLocationState.UNKNOWN
        mine_conf = 0.0
        matched_ref_name = ""

        # 1. Reference Template Matching for STATUS Category
        if reference_manager is not None:
            # Check drill state references
            drill_ref = reference_manager.find_best_match(
                frame, category="status", subcategory="drill_equipped", roi=roi, gray_image=gray_image, match_cache=match_cache
            )
            if not drill_ref or not drill_ref.found:
                drill_ref = reference_manager.find_best_match(
                    frame, category="status", subcategory="drill_unequipped", roi=roi, gray_image=gray_image, match_cache=match_cache
                )

            if drill_ref and drill_ref.found:
                matched_ref_name = drill_ref.reference_name
                drill_conf = drill_ref.confidence
                if "unequipped" in (drill_ref.subcategory + drill_ref.reference_name).lower():
                    drill_state = DrillState.UNEQUIPPED
                else:
                    drill_state = DrillState.EQUIPPED

            # Check battery empty references
            bat_ref = reference_manager.find_best_match(
                frame, category="status", subcategory="battery_empty", roi=roi, gray_image=gray_image, match_cache=match_cache
            )
            if not bat_ref or not bat_ref.found:
                # Check generic status references with battery in name
                for ref_res in reference_manager.find_all_matches(
                    frame, category="status", roi=roi, gray_image=gray_image, match_cache=match_cache
                ):
                    if "battery" in (ref_res.subcategory + ref_res.reference_name).lower():
                        bat_ref = ref_res
                        break

            if bat_ref and bat_ref.found:
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

        # 2. Fallback Mine Location Color Palette Analysis
        if mine_conf <= 0.5:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_v = float(np.mean(hsv[:, :, 2]))
            if avg_v < 100:
                mine_state = MineLocationState.INSIDE
                mine_conf = 0.85
            else:
                mine_state = MineLocationState.SURFACE
                mine_conf = 0.75

        return StatusDetectionResult(
            drill_state=drill_state,
            battery_state=battery_state,
            mine_state=mine_state,
            drill_confidence=drill_conf,
            battery_confidence=battery_conf,
            mine_confidence=mine_conf,
            matched_reference_name=matched_ref_name,
        )
