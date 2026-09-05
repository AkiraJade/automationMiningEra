"""Unit tests for Input Safety and F12 Emergency Stop system."""

import pytest
from app.input.safety import SafetyManager
from app.input.keyboard import KeyboardController
from app.input.mouse import MouseController


def test_dry_run_blocks_real_input():
    mgr = SafetyManager(dry_run=True)
    assert mgr.can_dispatch_input() is False


def test_emergency_stop_blocks_input():
    mgr = SafetyManager(dry_run=False)
    assert mgr.can_dispatch_input() is True

    mgr.trigger_emergency_stop("Test Emergency Stop")
    assert mgr.emergency_stopped is True
    assert mgr.can_dispatch_input() is False


def test_keyboard_and_mouse_controllers_respect_safety():
    kbd = KeyboardController()
    mouse = MouseController()

    # When dry_run is active (default), controllers return False and do not dispatch OS inputs
    assert kbd.press_key("space") is False
    assert mouse.click(100, 100) is False
