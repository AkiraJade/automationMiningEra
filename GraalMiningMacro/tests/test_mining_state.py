"""Unit tests for Mining State Machine."""

import pytest
from app.mining.mining_state import MiningState, STATE_PRIORITY
from app.mining.mining_target import MiningTarget, TargetMemoryBank, TargetState


def test_state_priority_ordering():
    assert STATE_PRIORITY[MiningState.EMERGENCY_STOP] > STATE_PRIORITY[MiningState.SPIDER_DETECTED]
    assert STATE_PRIORITY[MiningState.SPIDER_DETECTED] > STATE_PRIORITY[MiningState.MINE_COLLAPSE_DETECTED]
    assert STATE_PRIORITY[MiningState.MINE_COLLAPSE_DETECTED] > STATE_PRIORITY[MiningState.BATTERY_EMPTY]
    assert STATE_PRIORITY[MiningState.BATTERY_EMPTY] > STATE_PRIORITY[MiningState.MINING]


def test_target_memory_bank():
    bank = TargetMemoryBank(match_tolerance_px=30)
    target1 = bank.get_or_create(center=(100, 100), bbox=(80, 80, 40, 40), confidence=0.9)
    assert target1.state == TargetState.AVAILABLE

    # Query same position within tolerance
    target2 = bank.get_or_create(center=(105, 102), bbox=(80, 80, 40, 40), confidence=0.95)
    assert target1.target_id == target2.target_id

    # Query position outside tolerance
    target3 = bank.get_or_create(center=(300, 300), bbox=(280, 280, 40, 40), confidence=0.85)
    assert target1.target_id != target3.target_id
