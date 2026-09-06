"""Unit Tests for MiningSceneState, MiniRockDetector, and Red-Free Spider Verification."""

import pytest
import numpy as np
from app.mining.scene_model import MiningSceneState, MiniRockDetection
from app.vision.mini_rock_detector import MiniRockDetector
from app.vision.spider_detector import SpiderDetector
from app.vision.yellow_detector import YellowGlowDetector
from app.mining.mining_perception import MiningPerceptionEngine


def test_mining_scene_state_initialization():
    """Verifies MiningSceneState initializes correctly and formats summary text."""
    state = MiningSceneState()
    assert state.facing_direction == "UNKNOWN"
    assert state.target_source == "NONE"
    assert isinstance(state.mini_rock, MiniRockDetection)
    assert "PLAYER: UNKNOWN" in state.summary_text()


def test_spider_detector_rejects_red_objects_without_reference():
    """Verifies that red traffic cones, red buttons, or red objects NEVER trigger spider detection without reference match."""
    detector = SpiderDetector()
    frame = np.full((600, 800, 3), 50, dtype=np.uint8)

    # Draw bright red rectangle (BGR: 0, 0, 255)
    frame[300:350, 400:450] = [0, 0, 255]

    res = detector.detect(frame, player_center=(400, 400))
    assert res.detected is False
    assert res.is_candidate is False
    assert res.matched_reference_name == ""


def test_mini_rock_detector_facing_direction():
    """Verifies MiniRockDetector calculates search ROI behind player based on facing direction."""
    detector = MiniRockDetector(confirm_frames=1)
    frame = np.full((400, 600, 3), 40, dtype=np.uint8)

    # Place small high-contrast pebble patch behind player (player facing LEFT, pebble to the right at 330, 200)
    frame[195:205, 325:335] = [220, 220, 220]

    player_center = (300, 200)
    res = detector.detect(frame, player_center=player_center, facing_direction="LEFT")

    assert res.detected is True
    assert res.state == "MINI_ROCK_CONFIRMED"
    assert res.center is not None


def test_yellow_detector_rejects_yellow_outside_target_roi():
    """Verifies YellowGlowDetector rejects yellow DANGER sign outside target ROI."""
    detector = YellowGlowDetector()
    frame = np.full((600, 800, 3), 40, dtype=np.uint8)

    # Yellow DANGER triangle at bottom y=520
    frame[520:550, 400:430] = [0, 220, 255]

    player_center = (400, 300)
    target_center = (360, 300) # Wall contact 40px to left

    res = detector.detect(frame, player_center=player_center, target_center=target_center)
    assert res.is_confirmed is False
    assert res.detected_raw is False


def test_mining_perception_engine_produces_scene_state():
    """Verifies MiningPerceptionEngine populates result.scene_state correctly."""
    engine = MiningPerceptionEngine()
    frame = np.full((540, 960, 3), 100, dtype=np.uint8)

    res = engine.process_frame(frame)
    assert res.scene_state is not None
    assert isinstance(res.scene_state, MiningSceneState)
    assert res.scene_state.timestamp == res.timestamp
