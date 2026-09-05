"""Unit tests for Perception Pipeline Aggregator."""

import numpy as np
import pytest
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult


def test_perception_pipeline_processing():
    engine = MiningPerceptionEngine()
    frame = np.full((200, 200, 3), 50, dtype=np.uint8)

    result = engine.process_frame(frame)
    assert isinstance(result, MiningPerceptionResult)
    assert result.player.detected is True
    assert result.status.drill_state.value == "EQUIPPED"
    assert result.status.battery_state.value == "BATTERY_OK"
    assert result.overall_confidence > 0.0


def test_perception_pipeline_empty_frame():
    engine = MiningPerceptionEngine()
    result = engine.process_frame(None)
    assert result.player.detected is False
    assert result.overall_confidence == 0.0
