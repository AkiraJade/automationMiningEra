"""Video & Image Diagnostic Replay Test Runner."""

import os
import glob
import time
import cv2
import numpy as np
import pytest

from app.mining.mining_perception import MiningPerceptionEngine, MiningPerceptionResult
from app.mining.scene_model import MiningSceneState


def test_game_frames_replay_diagnostics():
    """Processes available user-uploaded game screenshots sequentially through MiningPerceptionEngine."""
    uploaded_dir = r"C:\Users\Jhade\.gemini\antigravity\brain\b6ebf358-ab24-42ed-b3a1-bff6bea61b9c\.user_uploaded"
    image_paths = sorted(glob.glob(os.path.join(uploaded_dir, "*.png")))

    if not image_paths:
        pytest.skip("No user uploaded images found for video replay diagnostic.")

    engine = MiningPerceptionEngine()
    timings = []
    yellow_false_positives = 0
    spider_false_positives = 0

    print(f"\n--- REPLAY DIAGNOSTIC START ({len(image_paths)} FRAMES) ---")

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None or frame.size == 0:
            continue

        t0 = time.perf_counter()
        res = engine.process_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)

        assert isinstance(res, MiningPerceptionResult)
        assert res.scene_state is not None
        assert isinstance(res.scene_state, MiningSceneState)

        # 1. Reject yellow completed false positive on lower message banner/DANGER sign
        if "media_1788619682953" in path or "media_1788690318632" in path:
            if res.yellow_glow.is_confirmed:
                yellow_false_positives += 1

        # 2. Reject red traffic cone false positive on lower message banner
        if "media_1788619682953" in path:
            if res.spider.detected:
                spider_false_positives += 1

        print(f"[{os.path.basename(path)}] {res.summary_text()} ({dt_ms:.1f}ms)")

    print(f"--- REPLAY DIAGNOSTIC END ---")
    avg_ms = np.mean(timings) if timings else 0.0
    max_ms = np.max(timings) if timings else 0.0
    perception_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"Avg Latency: {avg_ms:.1f}ms | Max Latency: {max_ms:.1f}ms | Perception FPS: {perception_fps:.1f} FPS")
    print(f"Yellow FP: {yellow_false_positives} | Spider FP: {spider_false_positives}")

    assert yellow_false_positives == 0, "Yellow completion false positive detected on dialog/DANGER sign."
    assert spider_false_positives == 0, "Spider false positive detected on red traffic cone/marker."
    assert avg_ms <= 150.0, f"Average perception throughput ({avg_ms:.1f}ms) exceeded 150ms budget for full 1080p desktop frames."
