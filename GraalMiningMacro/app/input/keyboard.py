"""DirectInput Keyboard Controller for Graal Mining Macro."""

import time
import pydirectinput
from app.input.safety import safety
from app.core.events import events
from app.core.logger import setup_logger

logger = setup_logger("KeyboardController")

# Disable PyDirectInput pause delay for fast response
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.01


class KeyboardController:
    """Safe keyboard controller guarded by F12 Emergency Stop and DryRun mode."""

    def press_key(self, key: str, duration: float = 0.05) -> bool:
        key_name = key.lower()
        action_desc = f"Press key '{key_name}' (duration: {duration:.2f}s)"

        if not safety.can_dispatch_input():
            logger.info(f"[DRY_RUN / SAFE] Intended action: {action_desc}")
            events.action_dispatched.emit(f"[SIMULATED] {action_desc}")
            return False

        try:
            logger.debug(f"Executing: {action_desc}")
            safety.track_key_down(key_name)
            pydirectinput.keyDown(key_name)
            time.sleep(duration)
            pydirectinput.keyUp(key_name)
            safety.track_key_up(key_name)
            events.action_dispatched.emit(f"[DISPATCHED] {action_desc}")
            return True
        except Exception as e:
            logger.error(f"Failed to press key '{key_name}': {e}")
            safety.release_all_held_keys()
            return False

    def key_down(self, key: str) -> bool:
        key_name = key.lower()
        action_desc = f"Key down '{key_name}'"

        if not safety.can_dispatch_input():
            logger.info(f"[DRY_RUN / SAFE] Intended action: {action_desc}")
            events.action_dispatched.emit(f"[SIMULATED] {action_desc}")
            return False

        try:
            safety.track_key_down(key_name)
            pydirectinput.keyDown(key_name)
            events.action_dispatched.emit(f"[DISPATCHED] {action_desc}")
            return True
        except Exception as e:
            logger.error(f"Failed key_down '{key_name}': {e}")
            return False

    def key_up(self, key: str) -> bool:
        key_name = key.lower()
        action_desc = f"Key up '{key_name}'"

        if not safety.can_dispatch_input():
            logger.info(f"[DRY_RUN / SAFE] Intended action: {action_desc}")
            events.action_dispatched.emit(f"[SIMULATED] {action_desc}")
            return False

        try:
            pydirectinput.keyUp(key_name)
            safety.track_key_up(key_name)
            events.action_dispatched.emit(f"[DISPATCHED] {action_desc}")
            return True
        except Exception as e:
            logger.error(f"Failed key_up '{key_name}': {e}")
            return False
