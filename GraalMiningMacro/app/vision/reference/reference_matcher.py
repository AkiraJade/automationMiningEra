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

    def match_single(
        self,
        image: np.ndarray,
        ref: ReferenceImage,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[Dict[Tuple[str, Optional[Tuple[int, int, int, int]]], ReferenceMatchResult]] = None
    ) -> ReferenceMatchResult:
        """Evaluates a single ReferenceImage against an image frame (or ROI)."""
        if not ref.enabled or image is None or image.size == 0:
            return ReferenceMatchResult(found=False, reference_id=ref.id, reference_name=ref.name, category=ref.category, subcategory=ref.subcategory)

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
                error_message="Reference file missing or unreadable"
            )
            if match_cache is not None:
                match_cache[cache_key] = res_err
            return res_err

        try:
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
            tpl_h, tpl_w = tpl_gray.shape[:2]

            # Dimension safety: resize template if larger than search region
            if img_h < tpl_h or img_w < tpl_w:
                scale_h = (img_h - 2) / float(tpl_h) if img_h > 2 else 1.0
                scale_w = (img_w - 2) / float(tpl_w) if img_w > 2 else 1.0
                scale = min(scale_h, scale_w)

                if scale <= 0.1:
                    res_small = ReferenceMatchResult(
                        found=False,
                        reference_id=ref.id,
                        reference_name=ref.name,
                        category=ref.category,
                        subcategory=ref.subcategory,
                        error_message="Search region too small for template"
                    )
                    if match_cache is not None:
                        match_cache[cache_key] = res_small
                    return res_small

                new_w = max(1, int(tpl_w * scale))
                new_h = max(1, int(tpl_h * scale))
                tpl_gray = cv2.resize(tpl_gray, (new_w, new_h))
                tpl_h, tpl_w = new_h, new_w

            # Run OpenCV template matching
            res = cv2.matchTemplate(search_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            match_x = max_loc[0] + offset_x
            match_y = max_loc[1] + offset_y
            center_x = match_x + tpl_w // 2
            center_y = match_y + tpl_h // 2

            found = float(max_val) >= float(ref.threshold)

            res_match = ReferenceMatchResult(
                found=found,
                reference_id=ref.id,
                reference_name=ref.name,
                category=ref.category,
                subcategory=ref.subcategory,
                confidence=float(max_val),
                bbox=(match_x, match_y, tpl_w, tpl_h),
                center=(center_x, center_y),
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

        # Sort matches by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def match_all(
        self,
        image: np.ndarray,
        registry: ReferenceRegistry,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None
    ) -> List[ReferenceMatchResult]:
        """Matches all enabled references across all categories against image."""
        results: List[ReferenceMatchResult] = []
        all_refs = registry.get_all()

        for ref in all_refs:
            if ref.enabled:
                res = self.match_single(image, ref, gray_image=gray_image, match_cache=match_cache)
                if res.found:
                    results.append(res)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
