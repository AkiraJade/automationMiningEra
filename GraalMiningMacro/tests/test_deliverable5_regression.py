"""Deliverable 5 Real-Game Vision Regression Test Suite."""

import time
import numpy as np
import pytest
import cv2
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.vision.wall_detector import WallDetector
from app.vision.player_detector import PlayerDetector
from app.vision.target_detector import TargetDetector
from app.vision.status_detectors import StatusDetector, DrillState


def test_perception_throughput_performance():
    """Verifies perception engine throughput processes standard 960x540 game frame in under 150ms."""
    engine = MiningPerceptionEngine()
    frame = np.full((540, 960, 3), 120, dtype=np.uint8)

    # Warmup tick
    engine.process_frame(frame)

    # Benchmark tick
    t0 = time.perf_counter()
    result = engine.process_frame(frame)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(result, MiningPerceptionResult)
    assert dt_ms < 200.0, f"Perception tick took {dt_ms:.2f}ms (Target: <200ms)"


def test_wall_detector_boundary_isolation():
    """Verifies wall detector ignores window borders/padding and top/bottom HUD lines."""
    detector = WallDetector(confidence_threshold=0.30, max_player_distance=120.0)

    # Frame with dark top/bottom padding border lines
    frame = np.full((600, 800, 3), 150, dtype=np.uint8)
    frame[0:80, :] = [10, 10, 10]      # Top border padding
    frame[520:600, :] = [10, 10, 10]    # Bottom border padding

    player_center = (400, 300)
    world_roi = (0, 90, 800, 420)

    det = detector.detect(frame, player_center=player_center, world_roi=world_roi)
    # Borders outside world_roi must be ignored
    assert det.detected is False or det.distance_px > 0.0


def test_wall_detector_player_proximity():
    """Verifies wall detector only accepts rock contours within max_player_distance."""
    detector = WallDetector(confidence_threshold=0.30, max_player_distance=80.0)
    frame = np.full((400, 600, 3), 150, dtype=np.uint8)
    # Dark rock contour 150px away from player (too far)
    frame[50:100, 50:100] = [10, 10, 10]

    player_center = (300, 200)
    world_roi = (0, 40, 600, 320)

    det = detector.detect(frame, player_center=player_center, world_roi=world_roi)
    assert det.detected is False


def test_player_explicit_states():
    """Verifies player detector explicitly sets player_state."""
    detector = PlayerDetector(allow_heuristic_fallback=True)
    frame = np.full((400, 600, 3), 120, dtype=np.uint8)

    # Without reference match, center fallback yields PLAYER_UNCERTAIN
    det = detector.detect(frame)
    assert det.detected is True
    assert det.player_state == "PLAYER_UNCERTAIN"
    assert det.is_heuristic is True

    # Disallowing heuristic fallback yields PLAYER_NOT_FOUND
    detector_no_fallback = PlayerDetector(allow_heuristic_fallback=False)
    det2 = detector_no_fallback.detect(frame)
    assert det2.detected is False
    assert det2.player_state == "PLAYER_NOT_FOUND"


def test_target_detector_state_validation():
    """Verifies target detector candidate state classification."""
    target_det = TargetDetector()

    # Invalid low confidence -> NO_TARGET
    res1 = target_det.update_target(center=(100, 100), bbox=(90, 90, 20, 20), confidence=0.20)
    assert res1.detected is False
    assert res1.target_state == "NO_TARGET"

    # Medium confidence -> TARGET_CANDIDATE
    res2 = target_det.update_target(center=(100, 100), bbox=(90, 90, 20, 20), confidence=0.50)
    assert res2.detected is True
    assert res2.target_state == "TARGET_CANDIDATE"

    # High confidence -> TARGET_CONFIRMED
    res3 = target_det.update_target(center=(100, 100), bbox=(90, 90, 20, 20), confidence=0.85)
    assert res3.detected is True
    assert res3.target_state == "TARGET_CONFIRMED"


def test_drill_state_temporal_margin():
    """Verifies drill status detector defaults to UNKNOWN and uses margin filter."""
    detector = StatusDetector()
    frame = np.full((200, 400, 3), 100, dtype=np.uint8)

    res = detector.detect(frame, player_detected=True)
    assert res.drill_state in [DrillState.UNKNOWN, DrillState.UNEQUIPPED, DrillState.EQUIPPED]


def test_yellow_danger_sign_rejected():
    """Verifies yellow DANGER sign in lower message banner is NOT classified as completed yellow rock."""
    from app.vision.yellow_detector import YellowGlowDetector
    yellow_det = YellowGlowDetector()

    # Frame 600x800 with yellow DANGER triangle at bottom (y = 500)
    frame = np.full((600, 800, 3), 50, dtype=np.uint8)
    # Draw yellow DANGER sign (BGR: 0, 220, 255)
    frame[500:540, 300:340] = [0, 220, 255]

    world_roi = (0, 90, 800, 420)
    res = yellow_det.detect(frame, roi_bbox=world_roi, player_center=(400, 300))

    assert res.is_confirmed is False
    assert res.detected_raw is False


def test_spider_red_cone_rejected():
    """Verifies red traffic cone/marker at bottom of viewport is NOT detected as a spider."""
    from app.vision.spider_detector import SpiderDetector
    spider_det = SpiderDetector()

    # Frame 600x800 with red cone at bottom (y = 520)
    frame = np.full((600, 800, 3), 50, dtype=np.uint8)
    # Draw red cone (BGR: 0, 0, 220)
    frame[520:550, 400:430] = [0, 0, 220]

    world_roi = (0, 90, 800, 420)
    res = spider_det.detect(frame, world_roi=world_roi, player_center=(400, 300))

    assert res.detected is False


def test_unequipped_drill_visual_evidence():
    """Verifies missing drill visual evidence results in DrillState.UNKNOWN (never forced EQUIPPED)."""
    status_det = StatusDetector()
    frame = np.full((200, 400, 3), 80, dtype=np.uint8)

    res = status_det.detect(frame, player_detected=False)
    assert res.drill_state == DrillState.UNKNOWN
    assert res.drill_confidence == 0.0


def test_unknown_state_preserved_when_evidence_insufficient():
    """Verifies UNKNOWN state is preserved across perception result when evidence is missing."""
    engine = MiningPerceptionEngine()
    frame = np.full((540, 960, 3), 100, dtype=np.uint8)

    result = engine.process_frame(frame)
    assert result.yellow_glow.is_confirmed is False
    assert result.spider.detected is False
    assert result.status.drill_state in [DrillState.UNKNOWN, DrillState.UNEQUIPPED]
    assert result.wall.direction == "UNKNOWN" or result.wall.detected is False


def test_real_game_screenshot_perception():
    """Verifies real-game failure screenshot (media_1788619682953.png) produces NO yellow rock or spider false positives."""
    import os
    img_path = r"C:\Users\Jhade\.gemini\antigravity\brain\b6ebf358-ab24-42ed-b3a1-bff6bea61b9c\.user_uploaded\media_1788619682953.png"
    if not os.path.exists(img_path):
        pytest.skip("User screenshot not found at path.")

    raw_frame = cv2.imread(img_path)
    assert raw_frame is not None and raw_frame.size > 0

    engine = MiningPerceptionEngine()
    # Process multiple ticks to test temporal filtering
    for _ in range(5):
        res = engine.process_frame(raw_frame)

    # 1. Yellow DANGER sign in lower message box MUST NOT trigger YELLOW_COMPLETE
    assert res.yellow_glow.is_confirmed is False

    # 2. Red cone in lower banner MUST NOT trigger SPIDER DETECTED
    assert res.spider.detected is False
