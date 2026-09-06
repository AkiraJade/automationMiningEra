"""Template Matcher Engine with Image Caching and Robust Error Boundaries."""

import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from app.vision.reference.reference_model import ReferenceImage, ReferenceMatchResult
from app.vision.reference.reference_registry import ReferenceRegistry
from app.core.logger import setup_logger

logger = setup_logger("ReferenceMatcher")


class ReferenceMatcher:
    """Optimized OpenCV template matcher with template caching and ROI support."""

    def __init__(self):
        # Template cache: file_path -> (bgr_array, gray_array, mtime)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        # Scaled template cache: (file_path, scale, mtime) -> scaled_gray_array
        self._scaled_cache: Dict[Tuple[str, float, float], np.ndarray] = {}

    def _get_template(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieves BGR and Grayscale template arrays from memory cache, reading disk if modified."""
        if not file_path or not os.path.exists(file_path):
            return None, None

        try:
            mtime = os.path.getmtime(file_path)
            if file_path in self._cache:
                bgr, gray, cached_mtime = self._cache[file_path]
                if cached_mtime == mtime:
                    return bgr, gray

            # Read from disk
            bgr = cv2.imread(file_path)
            if bgr is None or bgr.size == 0:
                logger.warning(f"Could not load reference image from disk: '{file_path}'")
                return None, None

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            self._cache[file_path] = (bgr, gray, mtime)
            return bgr, gray
        except Exception as e:
            logger.error(f"Error loading template image '{file_path}': {e}")
            return None, None

    def _get_scaled_template(self, file_path: str, scale: float) -> Optional[np.ndarray]:
        """Retrieves a cached scaled grayscale template array."""
        _, gray = self._get_template(file_path)
        if gray is None:
            return None

        if abs(scale - 1.0) < 1e-3:
            return gray

        try:
            mtime = os.path.getmtime(file_path)
            scaled_key = (file_path, round(scale, 3), mtime)
            if scaled_key in self._scaled_cache:
                return self._scaled_cache[scaled_key]

            h, w = gray.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self._scaled_cache[scaled_key] = scaled_gray
            return scaled_gray
        except Exception as e:
            logger.error(f"Error scaling template image '{file_path}' at scale {scale}: {e}")
            return None

    def match_single(
        self,
        image: np.ndarray,
        ref: ReferenceImage,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[Dict[Tuple[str, Optional[Tuple[int, int, int, int]]], ReferenceMatchResult]] = None,
        multi_scale: bool = True
    ) -> ReferenceMatchResult:
        """Evaluates a single ReferenceImage against an image frame (or ROI)."""
        if not ref.enabled or image is None or image.size == 0:
            return ReferenceMatchResult(
                found=False,
                reference_id=ref.id,
                reference_name=ref.name,
                category=ref.category,
                subcategory=ref.subcategory,
                method="TEMPLATE",
                rejection_reason="DISABLED_OR_INVALID_INPUT"
            )

        # Check per-cycle match cache first to avoid duplicate cv2.matchTemplate calls
        cache_key = (ref.id, roi)
        if match_cache is not None and cache_key in match_cache:
            return match_cache[cache_key]

        _, tpl_gray = self._get_template(ref.file_path)
        if tpl_gray is None:
            res_err = ReferenceMatchResult(
                found=False,
                reference_id=ref.id,
                reference_name=ref.name,
                category=ref.category,
                subcategory=ref.subcategory,
                method="TEMPLATE",
                rejection_reason="FILE_MISSING",
                error_message="Reference file missing or unreadable"
            )
            if match_cache is not None:
                match_cache[cache_key] = res_err
            return res_err

        try:
            from app.core import config
            offset_x, offset_y = 0, 0

            # Reuse single-pass converted grayscale image if available
            if gray_image is not None:
                search_gray = gray_image
            elif len(image.shape) == 3:
                search_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                search_gray = image

            # Apply ROI cropping if specified
            if roi is not None:
                rx, ry, rw, rh = roi
                img_h, img_w = search_gray.shape[:2]
                rx = max(0, min(rx, img_w - 1))
                ry = max(0, min(ry, img_h - 1))
                rw = max(1, min(rw, img_w - rx))
                rh = max(1, min(rh, img_h - ry))

                search_gray = search_gray[ry:ry+rh, rx:rx+rw]
                offset_x, offset_y = rx, ry

            img_h, img_w = search_gray.shape[:2]

            # Multi-scale setup
            scales_to_test = [1.0]
            if multi_scale:
                min_s = getattr(config, "MIN_SCALE", 0.85)
                max_s = getattr(config, "MAX_SCALE", 1.15)
                step_s = getattr(config, "SCALE_STEP", 0.05)
                scales_to_test = list(np.arange(min_s, max_s + 1e-5, step_s))
                # Ensure 1.0 is checked first for performance short-circuiting
                scales_to_test = sorted(scales_to_test, key=lambda s: abs(s - 1.0))

            best_max_val = -1.0
            best_max_loc = (0, 0)
            best_scale = 1.0
            best_tpl_w, best_tpl_h = tpl_gray.shape[1], tpl_gray.shape[0]

            for s in scales_to_test:
                scaled_tpl = self._get_scaled_template(ref.file_path, s)
                if scaled_tpl is None:
                    continue

                th, tw = scaled_tpl.shape[:2]
                if img_h < th or img_w < tw:
                    continue

                res = cv2.matchTemplate(search_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_max_val:
                    best_max_val = float(max_val)
                    best_max_loc = max_loc
                    best_scale = float(s)
                    best_tpl_w, best_tpl_h = tw, th

                # Short-circuit if scale=1.0 has excellent match
                if abs(s - 1.0) < 1e-3 and best_max_val >= 0.85:
                    break

            if best_max_val < 0.0:
                res_small = ReferenceMatchResult(
                    found=False,
                    reference_id=ref.id,
                    reference_name=ref.name,
                    category=ref.category,
                    subcategory=ref.subcategory,
                    raw_score=0.0,
                    confidence=0.0,
                    method="TEMPLATE",
                    scale=1.0,
                    rejection_reason="SEARCH_REGION_TOO_SMALL",
                    error_message="Search region too small for template scales"
                )
                if match_cache is not None:
                    match_cache[cache_key] = res_small
                return res_small

            match_x = best_max_loc[0] + offset_x
            match_y = best_max_loc[1] + offset_y
            center_x = match_x + best_tpl_w // 2
            center_y = match_y + best_tpl_h // 2

            min_conf = getattr(config, "MINIMUM_MATCH_CONFIDENCE", 0.65)
            thresh = float(ref.threshold)
            found = (best_max_val >= thresh) and (best_max_val >= min_conf)

            rejection_reason = ""
            if not found:
                rejection_reason = f"MATCH_SCORE_BELOW_THRESHOLD ({best_max_val:.2f} < {thresh:.2f})"

            res_match = ReferenceMatchResult(
                found=found,
                accepted=found,
                reference_id=ref.id,
                reference_name=ref.name,
                matched_reference=ref.name,
                category=ref.category,
                subcategory=ref.subcategory,
                raw_score=best_max_val,
                confidence=best_max_val if found else 0.0,
                bbox=(match_x, match_y, best_tpl_w, best_tpl_h),
                center=(center_x, center_y),
                method="TEMPLATE",
                scale=best_scale,
                rejection_reason=rejection_reason,
            )
            if match_cache is not None:
                match_cache[cache_key] = res_match
            return res_match
        except Exception as e:
            logger.error(f"OpenCV template matching exception for reference '{ref.name}': {e}")
            res_exc = ReferenceMatchResult(
                found=False,
                reference_id=ref.id,
                reference_name=ref.name,
                category=ref.category,
                subcategory=ref.subcategory,
                raw_score=0.0,
                confidence=0.0,
                method="TEMPLATE",
                rejection_reason=f"EXCEPTION ({e})",
                error_message=str(e)
            )
            if match_cache is not None:
                match_cache[cache_key] = res_exc
            return res_exc

    def match_all_in_category(
        self,
        image: np.ndarray,
        category: str,
        registry: ReferenceRegistry,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> List[ReferenceMatchResult]:
        """Matches all enabled references in a category against image, returning sorted matches."""
        results: List[ReferenceMatchResult] = []
        references = registry.get_enabled_by_category(category)

        for ref in references:
            res = self.match_single(image, ref, roi=roi, gray_image=gray_image, match_cache=match_cache)
            if res.found:
                results.append(res)

        # Sort matches by raw_score descending
        results.sort(key=lambda r: r.raw_score, reverse=True)
        return results

    def match_all(
        self,
        image: np.ndarray,
        registry: ReferenceRegistry,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> List[ReferenceMatchResult]:
        """Matches all enabled references across all categories against image."""
        results: List[ReferenceMatchResult] = []
        all_refs = registry.get_all()

        for ref in all_refs:
            if ref.enabled:
                res = self.match_single(image, ref, roi=roi, gray_image=gray_image, match_cache=match_cache)
                if res.found:
                    results.append(res)

        results.sort(key=lambda r: r.raw_score, reverse=True)
        return results
