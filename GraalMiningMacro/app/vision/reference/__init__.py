"""Reference-Based Mining Vision Package for Graal Mining Macro."""

from app.vision.reference.reference_model import ReferenceImage, ReferenceMatchResult, CATEGORIES, SUBCATEGORIES, DEFAULT_THRESHOLDS
from app.vision.reference.reference_registry import ReferenceRegistry
from app.vision.reference.reference_matcher import ReferenceMatcher
from app.vision.reference.reference_manager import ReferenceManager

__all__ = [
    "ReferenceImage",
    "ReferenceMatchResult",
    "CATEGORIES",
    "SUBCATEGORIES",
    "DEFAULT_THRESHOLDS",
    "ReferenceRegistry",
    "ReferenceMatcher",
    "ReferenceManager",
]
