from collections import deque
import time
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from app.mining.mining_controller import MiningController
from app.mining.mining_perception import MiningPerceptionResult
from app.core.logger import setup_logger

logger = setup_logger("PerceptionWorkerThread")


class PerceptionWorkerThread(QThread):
    """Executes perception engine and mining controller on a dedicated background thread.

    Uses a 'latest frame wins' single-element buffer to prevent frame queue accumulation
    and GUI thread starvation.
    """

    perception_complete_signal = Signal(object, float, float, str)
    # Emits: (MiningPerceptionResult perception, float proc_time_ms, float perception_fps, str health_status)

    def __init__(self, mining_controller: MiningController, target_fps: int = 10):
        super().__init__()
        self.mining_controller = mining_controller
        self.target_fps = max(1, min(60, target_fps))
        self._running = False
        self._mutex = QMutex()

        # Single-element latest frame buffer & frame drop counters
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_available: bool = False
        self._total_enqueued_frames: int = 0
        self._dropped_frames: int = 0

        # Performance monitoring metrics & rolling average history
        self.current_fps: float = 0.0
        self.last_proc_time_ms: float = 0.0
        self._proc_time_history = deque(maxlen=10)
        self.health_status: str = "HEALTHY"  # "HEALTHY", "SLOW", "ERROR"

    def set_target_fps(self, fps: int) -> None:
        with QMutexLocker(self._mutex):
            self.target_fps = max(1, min(60, fps))

    def enqueue_frame(self, frame: np.ndarray) -> None:
        """Pushes the newest captured frame into the single-element buffer.

        If the worker is busy processing a frame, the pending frame is replaced
        (latest frame wins), incrementing the dropped frame count.
        """
        if frame is None or frame.size == 0:
            return
        with QMutexLocker(self._mutex):
            self._total_enqueued_frames += 1
            if self._new_frame_available:
                self._dropped_frames += 1
            self._latest_frame = frame
            self._new_frame_available = True

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)

    def get_metrics(self) -> dict:
        with QMutexLocker(self._mutex):
            total = self._total_enqueued_frames
            dropped = self._dropped_frames
            pct = (dropped / total * 100.0) if total > 0 else 0.0
            avg_proc = float(np.mean(self._proc_time_history)) if self._proc_time_history else self.last_proc_time_ms
            formatted_text = f"{dropped:,} ({pct:.1f}%) [Rate Sync]" if self.health_status == "HEALTHY" else f"{dropped:,} ({pct:.1f}%)"
            return {
                "perception_fps": self.current_fps,
                "proc_time_ms": self.last_proc_time_ms,
                "avg_proc_time_ms": avg_proc,
                "total_enqueued_frames": total,
                "dropped_frames": dropped,
                "skipped_frames": dropped,
                "drop_rate_pct": pct,
                "dropped_formatted": formatted_text,
                "health": self.health_status,
            }

    def run(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = True

        logger.info(f"Perception worker thread started (Target FPS: {self.target_fps}).")
        frame_count = 0
        last_fps_calc = time.time()

        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
                target_fps = self.target_fps

            loop_start = time.time()
            target_frame_time = 1.0 / max(1, target_fps)

            # Retrieve latest frame from single-element buffer under lock
            current_frame = None
            with QMutexLocker(self._mutex):
                if self._new_frame_available:
                    current_frame = self._latest_frame
                    self._latest_frame = None
                    self._new_frame_available = False

            if current_frame is not None:
                t0 = time.perf_counter()
                try:
                    perception = self.mining_controller.process_tick(current_frame)
                    t1 = time.perf_counter()
                    proc_time_ms = (t1 - t0) * 1000.0

                    self._proc_time_history.append(proc_time_ms)
                    avg_proc_time = float(np.mean(self._proc_time_history))

                    # Health status determination based on rolling window average
                    if self.mining_controller.current_state.value == "PERCEPTION_ERROR":
                        health = "ERROR"
                    elif avg_proc_time > (target_frame_time * 1000.0 * 2.0):
                        health = "OVERLOADED"
                    elif avg_proc_time > (target_frame_time * 1000.0 * 1.25):
                        health = "SLOW"
                    else:
                        health = "HEALTHY"

                    frame_count += 1

                except Exception as e:
                    logger.error(f"Unhandled exception in perception worker run loop: {e}", exc_info=True)
                    proc_time_ms = 0.0
                    health = "ERROR"
                    perception = MiningPerceptionResult()

                with QMutexLocker(self._mutex):
                    self.last_proc_time_ms = proc_time_ms
                    self.health_status = health

                # Calculate perception FPS every 1 second
                now = time.time()
                elapsed = now - last_fps_calc
                if elapsed >= 1.0:
                    with QMutexLocker(self._mutex):
                        self.current_fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_calc = now

                # Emit lightweight result signal to GUI main thread
                self.perception_complete_signal.emit(
                    perception,
                    proc_time_ms,
                    self.current_fps,
                    health
                )

            # Regulate Perception FPS
            process_time = time.time() - loop_start
            sleep_time = target_frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Perception worker thread stopped.")
