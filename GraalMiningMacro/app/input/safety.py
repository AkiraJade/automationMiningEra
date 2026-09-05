"""Global F12 Emergency Stop and Safety System for Graal Mining Macro."""

import time
from typing import Set, Callable, Optional
from pynput import keyboard
from app.core.events import events
from app.core.logger import setup_logger

logger = setup_logger("InputSafety")


class SafetyManager:
    """Manages F12 emergency stop, key release guarantees, and DryRun guard."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.emergency_stopped = False
        self.held_keys: Set[str] = set()
        self._listener: Optional[keyboard.Listener] = None

    def start_listener(self) -> None:
        """Starts global low-level F12 key listener."""
        if self._listener is not None:
            return

        def on_press(key):
            try:
                if key == keyboard.Key.f12:
                    self.trigger_emergency_stop("F12 Hotkey Pressed")
            except Exception as e:
                logger.error(f"Error in keyboard listener: {e}")

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()
        logger.info("Global F12 Emergency Stop listener activated.")

    def stop_listener(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
                self._listener.join(timeout=0.5)
            except Exception:
                pass
            self._listener = None

    def trigger_emergency_stop(self, reason: str = "Emergency Stop Triggered") -> None:
        """Immediately halts all automation, releases held keys, and locks input."""
        self.emergency_stopped = True
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")

        # Release all tracked held keys
        self.release_all_held_keys()

        # Dispatch global events
        events.emergency_stop_triggered.emit(reason)

    def reset_emergency_stop(self) -> None:
        self.emergency_stopped = False
        logger.info("Emergency Stop reset.")

    def track_key_down(self, key: str) -> None:
        self.held_keys.add(key.lower())

    def track_key_up(self, key: str) -> None:
        self.held_keys.discard(key.lower())

    def release_all_held_keys(self) -> None:
        if not self.held_keys:
            return

        logger.info(f"Releasing held keys: {list(self.held_keys)}")
        # If real keys were sent via pydirectinput, release them
        import pydirectinput
        for key in list(self.held_keys):
            try:
                pydirectinput.keyUp(key)
            except Exception as e:
                logger.error(f"Failed to release key '{key}': {e}")
        self.held_keys.clear()

    def can_dispatch_input(self) -> bool:
        """Returns True only if automation is NOT emergency stopped and NOT in DryRun mode."""
        if self.emergency_stopped:
            logger.warning("Input blocked: Application is in EMERGENCY_STOP state.")
            return False
        if self.dry_run:
            logger.debug("Input simulated: DryRun mode is ENABLED.")
            return False
        return True


# Global Safety Instance
safety = SafetyManager()
