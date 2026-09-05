"""Direct HWND Window-Content Capture Backend for Graal Mining Macro."""

import os
import sys
import ctypes
import numpy as np
from typing import Optional
from app.capture.base_backend import BaseCaptureBackend
from app.capture.capture_target import CaptureTarget
from app.core.logger import setup_logger

logger = setup_logger("WindowGraphicsCaptureBackend")

IS_WINDOWS = sys.platform.startswith("win")

if IS_WINDOWS:
    import win32gui
    import win32ui
    import win32con


class WindowGraphicsCaptureBackend(BaseCaptureBackend):
    """Direct HWND Window-Content Capture Engine using Win32 PrintWindow API.
    Captures window content directly from HWND memory device context, independent of desktop focus or covering windows.
    """

    PW_CLIENTONLY = 0x00000001
    PW_RENDERFULLCONTENT = 0x00000002

    def __init__(self):
        self.own_pid = os.getpid()

    def capture_target_frame(self, target: CaptureTarget) -> Optional[np.ndarray]:
        if not IS_WINDOWS or not target or not target.is_valid:
            return None

        # HARD SAFETY GUARD 1: Minimized Window Check
        if target.is_minimized:
            logger.debug(f"Capture skipped for HWND {target.hwnd}: Window is minimized.")
            return None

        # HARD SAFETY GUARD 2: Own Process Exclusion
        if target.pid == self.own_pid or "graal mining macro" in target.title.lower():
            logger.critical(
                f"🚨 STRICT SAFETY BREACH REJECTED: Refusing to capture HWND {target.hwnd} "
                f"(PID {target.pid}, Title '{target.title}') matching macro process!"
            )
            return None

        hwnd = target.hwnd
        w = target.width
        h = target.height

        if w <= 0 or h <= 0 or not win32gui.IsWindow(hwnd):
            return None

        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        save_bitmap = None

        try:
            # Retrieve HWND window device context
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            if not hwnd_dc:
                return None

            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(save_bitmap)

            # Render HWND content directly via PrintWindow (PW_RENDERFULLCONTENT = 2)
            # This captures window surface directly regardless of overlapping windows on desktop
            result = ctypes.windll.user32.PrintWindow(
                hwnd,
                save_dc.GetSafeHdc(),
                self.PW_RENDERFULLCONTENT
            )

            if result != 1:
                # Fall back to PW_CLIENTONLY if PW_RENDERFULLCONTENT fails
                result = ctypes.windll.user32.PrintWindow(
                    hwnd,
                    save_dc.GetSafeHdc(),
                    self.PW_CLIENTONLY
                )

            if result == 1:
                bmp_info = save_bitmap.GetInfo()
                bmp_str = save_bitmap.GetBitmapBits(True)
                # Convert raw BGRA bitmap buffer to numpy array (BGR format)
                frame_bgra = np.frombuffer(bmp_str, dtype=np.uint8).reshape((h, w, 4))
                frame_bgr = frame_bgra[:, :, :3]
                return frame_bgr
            else:
                logger.warning(f"PrintWindow returned 0 for HWND {hwnd}")
                return None

        except Exception as e:
            logger.error(f"Direct window capture error for HWND {hwnd}: {e}")
            return None
        finally:
            # Clean up Win32 GDI handles cleanly
            if save_bitmap is not None:
                try:
                    win32gui.DeleteObject(save_bitmap.GetHandle())
                except Exception:
                    pass
            if save_dc is not None:
                try:
                    save_dc.DeleteDC()
                except Exception:
                    pass
            if mfc_dc is not None:
                try:
                    mfc_dc.DeleteDC()
                except Exception:
                    pass
            if hwnd_dc is not None and hwnd:
                try:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
                except Exception:
                    pass

    def close(self) -> None:
        pass
