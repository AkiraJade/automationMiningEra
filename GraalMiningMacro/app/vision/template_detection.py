"""Template Matching Computer Vision Detector for Graal Mining Macro."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class TemplateMatchResult:
    found: bool
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    confidence: float


class TemplateDetector:
    """Matches UI elements, red button, and indicators using OpenCV template matching."""

    @staticmethod
    def match_template(
        image: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.7,
        method: int = cv2.TM_CCOEFF_NORMED
    ) -> List[TemplateMatchResult]:
        if image is None or template is None or image.size == 0 or template.size == 0:
            return []

        img_h, img_w = image.shape[:2]
        tpl_h, tpl_w = template.shape[:2]

        if img_h < tpl_h or img_w < tpl_w:
            return []

        res = cv2.matchTemplate(image, template, method)
        loc = np.where(res >= threshold)
        results: List[TemplateMatchResult] = []

        for pt in zip(*loc[::-1]):
            score = float(res[pt[1], pt[0]])
            center_x = pt[0] + tpl_w // 2
            center_y = pt[1] + tpl_h // 2
            results.append(
                TemplateMatchResult(
                    found=True,
                    bbox=(pt[0], pt[1], tpl_w, tpl_h),
                    center=(center_x, center_y),
                    confidence=score,
                )
            )

        return results
