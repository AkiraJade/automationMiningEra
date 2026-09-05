"""Main Mining Automation State Machine Controller for Graal Mining Macro."""

import time
from typing import Optional
from app.mining.mining_state import MiningState, STATE_PRIORITY
from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.vision.status_detectors import DrillState, BatteryState, MineLocationState
from app.input.safety import safety
from app.input.keyboard import KeyboardController
from app.core.events import events
from app.core.logger import setup_logger

logger = setup_logger("MiningController")


class MiningController:
    """Orchestrates perception, state transitions, actions, and emergency interrupts."""

    def __init__(self, perception_engine: MiningPerceptionEngine):
        self.perception_engine = perception_engine
        self.keyboard = KeyboardController()
        self.current_state = MiningState.IDLE
        self.automation_level = 1  # 1 = OBSERVE mode
        self.last_perception: Optional[MiningPerceptionResult] = None

        # Perception error debouncing state
        self._last_perception_error_time: float = 0.0
        self._perception_error_debounce_sec: float = 5.0
        self._last_error_message: str = ""

        # Subscribe to emergency stop event
        events.emergency_stop_triggered.connect(self._on_emergency_stop)

    def _on_emergency_stop(self, reason: str) -> None:
        self.set_state(MiningState.EMERGENCY_STOP, f"Emergency Stop: {reason}")

    def set_state(self, new_state: MiningState, reason: str = "") -> None:
        if self.current_state != new_state:
            old_state_str = self.current_state.value
            new_state_str = new_state.value
            self.current_state = new_state
            logger.info(f"State transition: {old_state_str} ➔ {new_state_str} ({reason})")
            events.state_changed.emit(old_state_str, new_state_str)

    def process_tick(self, frame) -> MiningPerceptionResult:
        """Processes frame through perception engine and updates state machine safely."""
        # If emergency stopped, lock state
        if safety.emergency_stopped:
            self.set_state(MiningState.EMERGENCY_STOP, "F12 Active")
            return self.last_perception or MiningPerceptionResult()

        try:
            perception = self.perception_engine.process_frame(frame)
        except Exception as e:
            now = time.time()
            err_msg = str(e)
            if now - self._last_perception_error_time > self._perception_error_debounce_sec or err_msg != self._last_error_message:
                logger.error(f"[PERCEPTION_ERROR] Exception during perception frame processing: {e}", exc_info=True)
                self._last_perception_error_time = now
                self._last_error_message = err_msg

            self.set_state(MiningState.PERCEPTION_ERROR, f"Perception exception: {e}")
            fallback = MiningPerceptionResult()
            self.last_perception = fallback
            events.perception_updated.emit(fallback)
            return fallback

        self.last_perception = perception
        events.perception_updated.emit(perception)

        # Enforce State Priority Evaluation using canonical schema attributes
        if perception.spider.detected:
            self.set_state(MiningState.SPIDER_DETECTED, "Spider visually detected")
        elif perception.status.mine_state == MineLocationState.COLLAPSED:
            self.set_state(MiningState.MINE_COLLAPSE_DETECTED, "Mine collapse detected")
        elif perception.status.battery_state == BatteryState.BATTERY_EMPTY:
            self.set_state(MiningState.BATTERY_EMPTY, "Battery empty")
        elif perception.status.drill_state == DrillState.UNEQUIPPED:
            self.set_state(MiningState.CHECKING_DRILL, "Drill unequipped")
        elif perception.message.nothing_to_mine_detected:
            self.set_state(MiningState.NOTHING_TO_MINE, "'Nothing to Mine Here' detected")
        elif perception.yellow_glow.is_confirmed:
            self.set_state(MiningState.TARGET_COMPLETED, "Yellow glowing rock completed")
        elif perception.target.detected:
            if self.automation_level == 1:
                self.set_state(MiningState.OBSERVATION, "Observation Mode active")
            else:
                self.set_state(MiningState.MINING, "Target rock active")
        else:
            if self.automation_level == 1:
                self.set_state(MiningState.OBSERVATION, "Observation Mode active")
            else:
                self.set_state(MiningState.SEARCHING_FOR_TARGET, "Searching for wall target")

        return perception
