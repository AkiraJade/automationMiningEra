"""DirectInput Mouse Controller for Graal Mining Macro."""

import pydirectinput
from app.input.safety import safety
from app.core.events import events
from app.core.logger import setup_logger

logger = setup_logger("MouseController")


class MouseController:
    """Safe mouse controller guarded by F12 Emergency Stop and DryRun mode."""

    def click(self, screen_x: int, screen_y: int, button: str = "left") -> bool:
        action_desc = f"Mouse click {button} at screen ({screen_x}, {screen_y})"

        if not safety.can_dispatch_input():
            logger.info(f"[DRY_RUN / SAFE] Intended action: {action_desc}")
            events.action_dispatched.emit(f"[SIMULATED] {action_desc}")
            return False

        try:
            pydirectinput.click(x=screen_x, y=screen_y, button=button)
            events.action_dispatched.emit(f"[DISPATCHED] {action_desc}")
            return True
        except Exception as e:
            logger.error(f"Failed mouse click: {e}")
            return False

    def move_to(self, screen_x: int, screen_y: int) -> bool:
        action_desc = f"Mouse move to screen ({screen_x}, {screen_y})"

        if not safety.can_dispatch_input():
            logger.info(f"[DRY_RUN / SAFE] Intended action: {action_desc}")
            events.action_dispatched.emit(f"[SIMULATED] {action_desc}")
            return False

        try:
            pydirectinput.moveTo(x=screen_x, y=screen_y)
            events.action_dispatched.emit(f"[DISPATCHED] {action_desc}")
            return True
        except Exception as e:
            logger.error(f"Failed mouse move: {e}")
            return False
