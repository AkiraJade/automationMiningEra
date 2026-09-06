"""Reference Perception Manager Coordinating Registry and Matcher Execution."""

import numpy as np
from typing import Optional, List, Tuple
from app.vision.reference.reference_model import ReferenceImage, ReferenceMatchResult
from app.vision.reference.reference_registry import ReferenceRegistry
from app.vision.reference.reference_matcher import ReferenceMatcher
from app.core.logger import setup_logger

logger = setup_logger("ReferenceManager")


class ReferenceManager:
    """High-level vision coordinator interface for reference library and detector querying."""

    def __init__(self, base_dir: str = "reference"):
        self.registry = ReferenceRegistry(base_dir=base_dir)
        self.matcher = ReferenceMatcher()

    def find_best_match(
        self,
        image: np.ndarray,
        category: str,
        subcategory: Optional[str] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        multi_scale: bool = False
    ) -> Optional[ReferenceMatchResult]:
        """Finds the highest confidence match in a category (and optional subcategory)."""
        if image is None or image.size == 0:
            return None

        refs = self.registry.get_enabled_by_category(category)
        if subcategory:
            refs = [r for r in refs if r.subcategory == subcategory]

        best_match: Optional[ReferenceMatchResult] = None

        for ref in refs:
            match_res = self.matcher.match_single(
                image, ref, roi=roi, gray_image=gray_image, match_cache=match_cache, multi_scale=multi_scale
            )
            if match_res.found:
                if best_match is None or match_res.confidence > best_match.confidence:
                    best_match = match_res

        return best_match

    def find_all_matches(
        self,
        image: np.ndarray,
        category: Optional[str] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        gray_image: Optional[np.ndarray] = None,
        match_cache: Optional[dict] = None,
        multi_scale: bool = False
    ) -> List[ReferenceMatchResult]:
        """Returns all reference matches across categories or within a specific category."""
        if image is None or image.size == 0:
            return []

        if category:
            return self.matcher.match_all_in_category(
                image, category, self.registry, roi=roi, gray_image=gray_image, match_cache=match_cache, multi_scale=multi_scale
            )
        else:
            return self.matcher.match_all(
                image, self.registry, roi=roi, gray_image=gray_image, match_cache=match_cache, multi_scale=multi_scale
            )
