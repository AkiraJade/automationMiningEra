"""Unified Mining Perception Aggregator with Modular Detectors & Debounced Logging."""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from app.vision.player_detector import PlayerDetector, PlayerDetection
from app.vision.wall_detector import WallDetector, WallDetection
from app.vision.target_detector import TargetDetector, TargetDetection
from app.vision.yellow_detector import YellowGlowDetector, YellowGlowDetectionResult
from app.vision.spider_detector import SpiderDetector, SpiderDetection
from app.vision.message_detector import MessageDetector, MessageDetection
from app.vision.status_detectors import StatusDetector, StatusDetectionResult, DrillState, BatteryState, MineLocationState
from app.vision.yolo import YoloDetector
from app.vision.reference import ReferenceManager, ReferenceMatchResult
from app.mining.mining_target import TargetMemoryBank
from app.core.logger import setup_logger

logger = setup_logger("MiningPerception")


@dataclass
class MiningPerceptionResult:
    timestamp: float = field(default_factory=time.time)

    player: PlayerDetection = field(default_factory=PlayerDetection)
    wall: WallDetection = field(default_factory=WallDetection)
    target: TargetDetection = field(default_factory=TargetDetection)
    yellow_glow: YellowGlowDetectionResult = field(default_factory=YellowGlowDetectionResult)
    spider: SpiderDetection = field(default_factory=SpiderDetection)
    message: MessageDetection = field(default_factory=MessageDetection)
    status: StatusDetectionResult = field(default_factory=StatusDetectionResult)
    reference_matches: List[ReferenceMatchResult] = field(default_factory=list)

    overall_confidence: float = 0.0
    result_age_sec: float = 0.0
    detector_ages: dict = field(default_factory=dict)

    def summary_text(self) -> str:
        p_str = self.player.summary_text()
        w_str = self.wall.summary_text()
        t_str = self.target.summary_text()
        s_str = self.spider.summary_text()
        y_str = self.yellow_glow.summary_text()
        m_str = self.message.summary_text()
        return f"{p_str} | {w_str} | {t_str} | {y_str} | {s_str} | {m_str}"


class MiningPerceptionEngine:
    """Aggregates all modular vision detectors, reference library matcher, and logs debounced perception events."""

    def __init__(self, yolo_path: str = "models/graal_mining.pt", reference_dir: str = "reference"):
        self.player_detector = PlayerDetector()
        self.wall_detector = WallDetector()
        self.target_memory = TargetMemoryBank()
        self.target_detector = TargetDetector(memory_bank=self.target_memory)
        self.yellow_detector = YellowGlowDetector()
        self.spider_detector = SpiderDetector()
        self.message_detector = MessageDetector()
        self.status_detector = StatusDetector()
        self.yolo_detector = YoloDetector(model_path=yolo_path)
        self.reference_manager = ReferenceManager(base_dir=reference_dir)

        # Debounce log state tracking
        self._last_logged_player = False
        self._last_logged_iteration = -1
        self._last_logged_yellow = False
        self._last_logged_spider = False
        self._last_logged_message = False

        # Frequency scheduling state
        self._tick_count: int = 0
        self._last_status_result: Optional[StatusDetectionResult] = None
        self._last_status_timestamp: float = 0.0
        self._last_message_result: Optional[MessageDetection] = None
        self._last_message_timestamp: float = 0.0

        self.status_schedule_interval: int = 3   # Every 3 ticks
        self.message_schedule_interval: int = 3  # Every 3 ticks

    def process_frame(self, frame: np.ndarray) -> MiningPerceptionResult:
        now = time.time()
        result = MiningPerceptionResult(timestamp=now)

        if frame is None or frame.size == 0:
            return result

        import cv2
        self._tick_count += 1

        # Single-pass grayscale conversion reuse for all template matching in this cycle
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Per-cycle reference match cache to eliminate duplicate cv2.matchTemplate calls per tick
        match_cache: dict = {}

        # 1. YOLO Detections (if model loaded)
        yolo_dets = self.yolo_detector.detect(frame) if self.yolo_detector.is_loaded else None

        # 2. Player Detection (High frequency: every tick)
        result.player = self.player_detector.detect(
            frame,
            yolo_detections=yolo_dets,
            reference_manager=self.reference_manager,
            gray_image=gray_frame,
            match_cache=match_cache,
        )

        # 3. Wall Detection (High frequency: every tick)
        result.wall = self.wall_detector.detect(frame, player_center=result.player.center)

        # 4. Yellow Glow Detection (High frequency: every tick)
        result.yellow_glow = self.yellow_detector.detect(
            frame,
            reference_manager=self.reference_manager,
            gray_image=gray_frame,
            match_cache=match_cache,
        )

        # 5. Target Detection (High frequency: every tick)
        target_center = result.yellow_glow.center
        target_bbox = result.yellow_glow.bbox
        target_conf = result.yellow_glow.confidence

        if not target_center and result.wall.detected and result.wall.bbox:
            wx, wy, ww, wh = result.wall.bbox
            target_center = (wx + ww // 2, wy + wh // 2)
            target_bbox = result.wall.bbox
            target_conf = result.wall.confidence

        result.target = self.target_detector.update_target(
            center=target_center,
            bbox=target_bbox,
            confidence=target_conf,
            is_yellow_completed=result.yellow_glow.is_confirmed
        )

        # 6. Spider Detection (High frequency: every tick)
        result.spider = self.spider_detector.detect(
            frame,
            player_center=result.player.center,
            yolo_detections=yolo_dets,
            reference_manager=self.reference_manager,
            gray_image=gray_frame,
            match_cache=match_cache,
        )

        # 7. Message Detection (Scheduled medium frequency)
        if self._tick_count % self.message_schedule_interval == 1 or self._last_message_result is None:
            result.message = self.message_detector.detect(
                frame,
                reference_manager=self.reference_manager,
                gray_image=gray_frame,
                match_cache=match_cache,
            )
            self._last_message_result = result.message
            self._last_message_timestamp = now
        else:
            result.message = self._last_message_result

        # 8. Status Infrastructure (Scheduled low frequency)
        if self._tick_count % self.status_schedule_interval == 1 or self._last_status_result is None:
            result.status = self.status_detector.detect(
                frame,
                player_detected=result.player.detected,
                reference_manager=self.reference_manager,
                gray_image=gray_frame,
                match_cache=match_cache,
            )
            self._last_status_result = result.status
            self._last_status_timestamp = now
        else:
            result.status = self._last_status_result

        # 9. Aggregate All Active Reference Matches (Reuses per-cycle match_cache)
        result.reference_matches = self.reference_manager.find_all_matches(
            frame, gray_image=gray_frame, match_cache=match_cache
        )

        # 10. Record Detector Result Age Timestamps
        result.detector_ages = {
            "player": 0.0,
            "spider": 0.0,
            "target": 0.0,
            "status": round(now - self._last_status_timestamp, 3),
            "message": round(now - self._last_message_timestamp, 3),
        }
        result.result_age_sec = max(result.detector_ages.values()) if result.detector_ages else 0.0

        # 10. Aggregate Overall Confidence
        conf_scores = []
        if result.player.detected:
            conf_scores.append(result.player.confidence)
        if result.target.detected:
            conf_scores.append(result.target.confidence)
        if result.yellow_glow.is_confirmed:
            conf_scores.append(result.yellow_glow.confidence)

        result.overall_confidence = float(np.mean(conf_scores)) if conf_scores else 0.75

        # 11. Debounced Logging
        self._log_debounced_events(result)

        return result

    def _log_debounced_events(self, p: MiningPerceptionResult) -> None:
        """Emits event log messages only when meaningful perception state changes occur."""
        # Log player detection state change
        if p.player.detected != self._last_logged_player:
            if p.player.detected:
                logger.info(f"[PERCEPTION] Player detected confidence={p.player.confidence:.2f}")
            else:
                logger.warning("[PERCEPTION] Player lost / UNKNOWN")
            self._last_logged_player = p.player.detected

        # Log target iteration change
        if p.target.detected and p.target.iteration != self._last_logged_iteration:
            logger.info(f"[MINING] Iteration changed -> {p.target.iteration}/3")
            self._last_logged_iteration = p.target.iteration

        # Log yellow glow completion confirmation
        if p.yellow_glow.is_confirmed and not self._last_logged_yellow:
            logger.info(f"[MINING] Yellow glow confirmed confidence={p.yellow_glow.confidence:.2f} - Target completed!")
            self._last_logged_yellow = True
        elif not p.yellow_glow.is_confirmed:
            self._last_logged_yellow = False

        # Log spider threat
        if p.spider.detected and not self._last_logged_spider:
            logger.warning(f"[SPIDER] Spider detected confidence={p.spider.confidence:.2f} distance={p.spider.distance_from_player:.0f}px")
            self._last_logged_spider = True
        elif not p.spider.detected:
            self._last_logged_spider = False

        # Log Nothing to Mine Here message
        if p.message.nothing_to_mine_detected and not self._last_logged_message:
            logger.warning(f"[MINING] 'Nothing to Mine Here' detected - Cooldown active ({p.message.cooldown_remaining:.1f}s)")
            self._last_logged_message = True
        elif not p.message.nothing_to_mine_detected:
            self._last_logged_message = False
