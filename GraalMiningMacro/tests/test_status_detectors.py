"""Unit tests for Status Infrastructure Detectors."""

import numpy as np
import pytest
from app.vision.status_detectors import StatusDetector, StatusDetectionResult, DrillState, BatteryState, MineLocationState


def test_status_detector_inside_mine():
    detector = StatusDetector()
    # Dark floor frame (inside mine)
    dark_frame = np.full((200, 200, 3), 30, dtype=np.uint8)

    res = detector.detect(dark_frame, player_detected=True)
    assert isinstance(res, StatusDetectionResult)
    assert res.battery_state == BatteryState.BATTERY_OK
    assert res.mine_state == MineLocationState.INSIDE


def test_status_detector_surface():
    detector = StatusDetector()
    # Bright surface outdoor frame
    bright_frame = np.full((200, 200, 3), 180, dtype=np.uint8)

    res = detector.detect(bright_frame, player_detected=False)
    assert res.mine_state == MineLocationState.SURFACE
    assert res.drill_state == DrillState.UNKNOWN


def test_drill_mutually_exclusive_detection_and_temporal_confirmation(tmp_path):
    """Verifies drill detector requires score margin and 3-frame confirmation, never defaulting to EQUIPPED."""
    from app.vision.reference import ReferenceManager

    manager = ReferenceManager(base_dir=str(tmp_path))
    detector = StatusDetector(drill_margin=0.10, confirm_frames=3)

    rng = np.random.RandomState(555)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    eq_patch = rng.randint(180, 255, (20, 20, 3), dtype=np.uint8)
    uneq_patch = rng.randint(180, 255, (20, 20, 3), dtype=np.uint8)

    # Place unequipped drill icon in frame
    frame[10:30, 10:30] = uneq_patch

    manager.registry.add_reference(
        name="drill_equipped_ref",
        category="status",
        subcategory="drill_equipped",
        source_file_or_image=eq_patch,
        threshold=0.70
    )
    manager.registry.add_reference(
        name="drill_unequipped_ref",
        category="status",
        subcategory="drill_unequipped",
        source_file_or_image=uneq_patch,
        threshold=0.70
    )

    # Frame 1: Candidate UNEQUIPPED (1/3)
    res1 = detector.detect(frame, player_detected=True, reference_manager=manager)
    assert res1.drill_state == DrillState.UNKNOWN

    # Frame 2: Candidate UNEQUIPPED (2/3)
    res2 = detector.detect(frame, player_detected=True, reference_manager=manager)
    assert res2.drill_state == DrillState.UNKNOWN

    # Frame 3: Confirmed UNEQUIPPED (3/3)
    res3 = detector.detect(frame, player_detected=True, reference_manager=manager)
    assert res3.drill_state == DrillState.UNEQUIPPED
    assert res3.matched_reference_name == "drill_unequipped_ref"
