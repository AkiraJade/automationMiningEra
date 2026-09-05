"""Global Event Signals Bus for Graal Mining Macro."""

from PySide6.QtCore import QObject, Signal


class EventBus(QObject):
    """Central signal hub for cross-module decoupled communication."""

    # Window detection signals
    window_found = Signal(object)        # WindowInfo
    window_lost = Signal()

    # Capture signals
    frame_captured = Signal(object)     # numpy array (BGR frame)
    capture_fps_updated = Signal(float)  # current FPS float
    capture_error = Signal(str)         # error message

    # Emergency & Safety signals
    emergency_stop_triggered = Signal(str) # reason
    safety_state_changed = Signal(bool)    # dry_run state

    # Perception & Mining signals
    perception_updated = Signal(object)   # MiningPerceptionResult
    state_changed = Signal(str, str)      # old_state, new_state
    target_updated = Signal(object)       # MiningTarget
    action_dispatched = Signal(str)       # description of action intended/performed


# Singleton instance
events = EventBus()
