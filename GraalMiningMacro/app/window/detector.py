"""Windows Game-Window Detector for Graal Mining Macro."""

import os
import sys
import re
from typing import List, Optional, Tuple
from app.window.models import WindowInfo, CandidateWindowInfo
from app.core.logger import setup_logger

logger = setup_logger("WindowDetector")

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import win32gui
    import win32process
    import win32con

try:
    import psutil
except ImportError:
    psutil = None


class WindowDetector:
    """Detects and monitors the Graal Online Era game window with strict self-process exclusion and focus-independent target locking."""

    KNOWN_GRAAL_PROCESSES = {
        "era.exe",
        "graalonline.exe",
        "graalera.exe",
        "graal.exe",
        "flashplayer.exe",
    }

    MACRO_TITLE_KEYWORDS = {
        "graal mining macro",
        "graal multi macro",
        "graal multi tool",
    }

    def __init__(
        self,
        title_pattern: str = "GraalOnline Era",
        target_executable: str = "Era.exe",
        match_exact: bool = False
    ):
        self.title_pattern = title_pattern
        self.target_executable = target_executable
        self.match_exact = match_exact
        self._target_hwnd: Optional[int] = None
        self._current_info: Optional[WindowInfo] = None
        self.own_pid = os.getpid()

    def get_process_info(self, hwnd: int) -> Tuple[int, str]:
        """Resolves Process ID and executable process name for an HWND."""
        if not IS_WINDOWS or not hwnd:
            return (0, "Unknown")

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = "Unknown"
            if pid > 0 and psutil is not None:
                try:
                    proc = psutil.Process(pid)
                    process_name = proc.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            return (pid, process_name)
        except Exception:
            return (0, "Unknown")

    @staticmethod
    def _normalize_string(s: str) -> str:
        """Removes spaces, punctuation, and converts to lowercase for flexible string matching."""
        return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

    def list_candidate_windows(
        self, override_pattern: Optional[str] = None, override_exe: Optional[str] = None
    ) -> List[CandidateWindowInfo]:
        """Enumerates top-level windows and computes candidacy, match score, and rejection reasons.
        NOTE: Window focus / active window (GetForegroundWindow) is COMPLETELY IGNORED.
        """
        if not IS_WINDOWS:
            return []

        pattern = (override_pattern or self.title_pattern).lower()
        pattern_norm = self._normalize_string(pattern)
        target_exe = (override_exe or self.target_executable).lower()

        candidates: List[CandidateWindowInfo] = []

        def enum_windows_callback(hwnd: int, extra: None) -> bool:
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True

            is_visible = bool(win32gui.IsWindowVisible(hwnd))
            is_minimized = bool(win32gui.IsIconic(hwnd))

            # Client geometry
            try:
                c_left, c_top, c_right, c_bottom = win32gui.GetClientRect(hwnd)
                client_w = c_right - c_left
                client_h = c_bottom - c_top
            except Exception:
                client_w, client_h = 0, 0

            pid, process_name = self.get_process_info(hwnd)
            proc_name_lower = process_name.lower()
            is_own_proc = (pid == self.own_pid)

            # Evaluate hard rejection rules
            title_lower = title.lower()
            title_norm = self._normalize_string(title)
            rejection_reason = None

            if is_own_proc:
                rejection_reason = f"Excluded: Belongs to own macro process ID ({pid})"
            elif any(kw in title_lower for kw in self.MACRO_TITLE_KEYWORDS):
                rejection_reason = "Excluded: Matches macro application title"
            elif not is_visible:
                rejection_reason = "Excluded: Window is invisible"
            elif is_minimized:
                rejection_reason = "Excluded: Window is minimized"
            elif client_w < 100 or client_h < 100:
                rejection_reason = f"Excluded: Client resolution too small ({client_w}x{client_h})"

            # Calculate deterministic match score (NO DEPENDENCY ON GetForegroundWindow)
            score = 0
            if rejection_reason is None:
                # Priority 1: Executable Identity (Era.exe or known Graal executables)
                if proc_name_lower == target_exe or proc_name_lower in self.KNOWN_GRAAL_PROCESSES:
                    score += 500

                # Priority 2: Title Matching (Flexible normalized comparison)
                if self.match_exact:
                    if title_lower == pattern or title_norm == pattern_norm:
                        score += 300
                    elif score == 0:
                        rejection_reason = f"Excluded: Title '{title}' does not match exact pattern '{pattern}'"
                else:
                    if pattern_norm in title_norm or title_norm in pattern_norm:
                        score += 300
                    elif "graal" in title_norm and "era" in title_norm:
                        score += 200
                    elif "graal" in title_norm or "era" in title_norm:
                        score += 100
                    elif score == 0:
                        rejection_reason = f"Excluded: Neither process '{process_name}' nor title '{title}' matches Graal identifiers"

                if score > 0:
                    score += min(100, int((client_w * client_h) / 10000))

            candidates.append(
                CandidateWindowInfo(
                    hwnd=hwnd,
                    title=title,
                    pid=pid,
                    process_name=process_name,
                    client_width=client_w,
                    client_height=client_h,
                    is_visible=is_visible,
                    is_minimized=is_minimized,
                    is_own_process=is_own_proc,
                    score=score,
                    rejection_reason=rejection_reason,
                )
            )
            return True

        try:
            win32gui.EnumWindows(enum_windows_callback, None)
        except Exception as e:
            logger.error(f"Error enumerating windows: {e}")

        return candidates

    def find_window(self, override_title: Optional[str] = None, override_hwnd: Optional[int] = None) -> Optional[WindowInfo]:
        if not IS_WINDOWS:
            return None

        if override_hwnd:
            pid, proc_name = self.get_process_info(override_hwnd)
            if pid == self.own_pid:
                logger.warning(f"Explicit HWND {override_hwnd} rejected because it belongs to own process PID {pid}.")
                self._target_hwnd = None
                self._current_info = None
                return None
            self._target_hwnd = override_hwnd
            return self.get_window_info_by_hwnd(override_hwnd)

        all_candidates = self.list_candidate_windows(override_pattern=override_title)
        accepted_candidates = [c for c in all_candidates if c.is_accepted and c.score > 0]

        if not accepted_candidates:
            logger.debug("Candidate search completed: No valid Graal Online Era window found. Status: DISCONNECTED")
            self._target_hwnd = None
            self._current_info = None
            return None

        # Deterministic sorting: highest score -> largest client area -> lowest HWND
        accepted_candidates.sort(key=lambda c: (c.score, c.client_width * c.client_height, -c.hwnd), reverse=True)
        best = accepted_candidates[0]

        self._target_hwnd = best.hwnd
        logger.info(
            f"GAME WINDOW DETECTED | Title: '{best.title}' | HWND: {best.hwnd} | "
            f"PID: {best.pid} | Process: {best.process_name} | Resolution: {best.client_width}x{best.client_height} (Score: {best.score})"
        )
        self._current_info = self.get_window_info_by_hwnd(best.hwnd)
        return self._current_info

    def get_window_info_by_hwnd(self, hwnd: int) -> Optional[WindowInfo]:
        if not IS_WINDOWS or not hwnd or not win32gui.IsWindow(hwnd):
            return None

        pid, process_name = self.get_process_info(hwnd)

        # STRICT SAFETY GUARD: Reject own application process
        if pid == self.own_pid:
            logger.error(f"STRICT SAFETY CHECK: Rejecting HWND {hwnd} (PID {pid}) because it belongs to macro process!")
            return None

        try:
            title = win32gui.GetWindowText(hwnd)
            is_visible = bool(win32gui.IsWindowVisible(hwnd))
            is_iconic = bool(win32gui.IsIconic(hwnd))

            # Outer rectangle
            outer_rect = win32gui.GetWindowRect(hwnd)

            # Client rectangle relative to screen
            c_left, c_top, c_right, c_bottom = win32gui.GetClientRect(hwnd)
            client_w = c_right - c_left
            client_h = c_bottom - c_top

            # Map client top-left (0,0) to screen coordinates
            screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = (screen_left, screen_top, screen_left + client_w, screen_top + client_h)

            info = WindowInfo(
                hwnd=hwnd,
                title=title,
                pid=pid,
                process_name=process_name,
                outer_rect=outer_rect,
                client_rect=client_rect,
                client_width=client_w,
                client_height=client_h,
                is_visible=is_visible,
                is_minimized=is_iconic,
            )
            return info

        except Exception as e:
            logger.error(f"Failed to query HWND {hwnd}: {e}")
            return None

    def refresh(self) -> Optional[WindowInfo]:
        """Refreshes geometry for locked target HWND. Re-detects ONLY if target HWND becomes invalid or closed."""
        if self._target_hwnd:
            updated = self.get_window_info_by_hwnd(self._target_hwnd)
            if updated and updated.is_valid:
                self._current_info = updated
                return updated
            else:
                logger.info(f"Target HWND {self._target_hwnd} is no longer valid. Triggering re-detection...")
                self._target_hwnd = None
                self._current_info = None

        return self.find_window()

    @property
    def target_hwnd(self) -> Optional[int]:
        return self._target_hwnd

    @property
    def current_info(self) -> Optional[WindowInfo]:
        return self._current_info
