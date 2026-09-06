"""Unified Mining Scene Model for Graal Mining Macro."""

import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict

from app.vision.player_detector import PlayerDetection
from app.vision.wall_detector import WallDetection
from app.vision.target_detector import TargetDetection
from app.vision.yellow_detector import YellowGlowDetector, YellowGlowDetectionResult
from app.vision.spider_detector import SpiderDetection
from app.vision.message_detector import MessageDetection
from app.vision.status_detectors import StatusDetectionResult, DrillState
from app.vision.reference import ReferenceMatchResult


@dataclass
class MiniRockDetection:
    """Detection result for small rock/pebble drops behind the player."""
    detected: bool = False
    is_candidate: bool = False
    consecutive_frames: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    state: str = "MINI_ROCK_NONE"  # "MINI_ROCK_NONE", "MINI_ROCK_CANDIDATE", "MINI_ROCK_CONFIRMED"

    def summary_text(self) -> str:
        if not self.detected and not self.is_candidate:
            return "MINI ROCK: NONE"
        return f"MINI ROCK: {self.state} ({self.confidence * 100:.0f}%)"


@dataclass
class MiningSceneState:
    """Structured immutable snapshot of the entire mining scene geometry and entity states."""
    timestamp: float = field(default_factory=time.time)

    # Core Spatial Entities
    player: PlayerDetection = field(default_factory=PlayerDetection)
    facing_direction: str = "UNKNOWN"
    wall: WallDetection = field(default_factory=WallDetection)
    target: TargetDetection = field(default_factory=TargetDetection)
    target_contact_point: Optional[Tuple[int, int]] = None
    target_roi: Optional[Tuple[int, int, int, int]] = None
    target_source: str = "NONE"  # "WALL_CONTACT", "REFERENCE", "NONE"

    # Contextual Entities
    yellow_glow: YellowGlowDetectionResult = field(default_factory=YellowGlowDetectionResult)
    mini_rock: MiniRockDetection = field(default_factory=MiniRockDetection)
    spider: SpiderDetection = field(default_factory=SpiderDetection)
    message: MessageDetection = field(default_factory=MessageDetection)
    status: StatusDetectionResult = field(default_factory=StatusDetectionResult)

    # Reference Matches & Timings
    reference_matches: List[ReferenceMatchResult] = field(default_factory=list)
    overall_confidence: float = 0.0
    result_age_sec: float = 0.0
    detector_timings: Dict[str, float] = field(default_factory=dict)

    def summary_text(self) -> str:
        p_str = self.player.summary_text()
        f_str = f"FACING:{self.facing_direction}"
        w_str = self.wall.summary_text()
        t_str = self.target.summary_text()
        y_str = self.yellow_glow.summary_text()
        r_str = self.mini_rock.summary_text()
        s_str = self.spider.summary_text()
        m_str = self.message.summary_text()
        return f"{p_str} | {f_str} | {w_str} | {t_str} | {y_str} | {r_str} | {s_str} | {m_str}"
