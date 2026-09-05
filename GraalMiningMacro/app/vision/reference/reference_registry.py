"""Persistent Registry for Managing Reference Image Metadata and Library Folders."""

import os
import json
import uuid
import time
import cv2
import numpy as np
from typing import List, Dict, Optional
from app.vision.reference.reference_model import ReferenceImage, CATEGORIES, DEFAULT_THRESHOLDS
from app.core.logger import setup_logger

logger = setup_logger("ReferenceRegistry")


class ReferenceRegistry:
    """Manages persistent reference library metadata and disk storage."""

    def __init__(self, base_dir: str = "reference", json_filename: str = "references.json"):
        self.base_dir = os.path.abspath(base_dir)
        self.json_path = os.path.join(self.base_dir, json_filename)
        self.references: Dict[str, ReferenceImage] = {}

        self.ensure_directories()
        self.load()

    def ensure_directories(self) -> None:
        """Creates the root reference directory and all category subdirectories."""
        os.makedirs(self.base_dir, exist_ok=True)
        for cat in CATEGORIES:
            cat_dir = os.path.join(self.base_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

    def load(self) -> None:
        """Loads reference registry metadata from disk cleanly."""
        self.references.clear()
        if not os.path.exists(self.json_path):
            logger.info(f"No existing reference registry found at '{self.json_path}'. Starting with empty registry.")
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    try:
                        ref = ReferenceImage.from_dict(item)
                        self.references[ref.id] = ref
                    except Exception as e:
                        logger.warning(f"Skipping malformed reference entry: {e}")
            logger.info(f"Loaded {len(self.references)} reference items from registry.")
        except Exception as e:
            logger.error(f"Failed to load reference registry JSON: {e}")

    def save(self) -> None:
        """Persists reference metadata to references.json on disk."""
        try:
            data = [ref.to_dict() for ref in self.references.values()]
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.debug(f"Saved {len(data)} reference entries to '{self.json_path}'.")
        except Exception as e:
            logger.error(f"Failed to save reference registry: {e}")

    def add_reference(
        self,
        name: str,
        category: str,
        subcategory: str = "custom",
        source_file_or_image: Optional[object] = None,
        threshold: Optional[float] = None,
        notes: str = ""
    ) -> Optional[ReferenceImage]:
        """Creates a managed reference item, copies/saves the image file to the reference directory, and updates registry."""
        if category not in CATEGORIES:
            category = "player"

        cat_dir = os.path.join(self.base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        ref_id = str(uuid.uuid4())[:8]
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.strip()).lower() or f"ref_{ref_id}"
        target_filename = f"{safe_name}_{ref_id}.png"
        target_path = os.path.abspath(os.path.join(cat_dir, target_filename))

        try:
            if isinstance(source_file_or_image, np.ndarray):
                if source_file_or_image.size == 0:
                    logger.error("Cannot add reference: Source numpy image array is empty.")
                    return None
                cv2.imwrite(target_path, source_file_or_image)
            elif isinstance(source_file_or_image, str) and os.path.exists(source_file_or_image):
                img = cv2.imread(source_file_or_image)
                if img is None or img.size == 0:
                    logger.error(f"Cannot add reference: Source image file '{source_file_or_image}' is unreadable or corrupt.")
                    return None
                cv2.imwrite(target_path, img)
            else:
                logger.error("Cannot add reference: Invalid source image or file path provided.")
                return None
        except Exception as e:
            logger.error(f"Failed to write reference image file to '{target_path}': {e}")
            return None

        thresh = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(category, 0.80)
        ref = ReferenceImage(
            id=ref_id,
            name=name,
            category=category,
            subcategory=subcategory,
            file_path=target_path,
            enabled=True,
            threshold=thresh,
            created_at=time.time(),
            notes=notes,
        )

        self.references[ref_id] = ref
        self.save()
        logger.info(f"Successfully added reference '{ref.name}' ({ref.category}/{ref.subcategory}) to '{target_path}'.")
        return ref

    def delete_reference(self, ref_id: str) -> bool:
        """Deletes reference metadata and removes image file from disk if present."""
        if ref_id not in self.references:
            return False

        ref = self.references.pop(ref_id)
        if ref.file_path and os.path.exists(ref.file_path):
            try:
                os.remove(ref.file_path)
            except Exception as e:
                logger.warning(f"Could not remove reference file '{ref.file_path}': {e}")

        self.save()
        logger.info(f"Deleted reference '{ref.name}' (ID: {ref_id}).")
        return True

    def toggle_reference(self, ref_id: str, enabled: bool) -> bool:
        """Enables or disables a reference item."""
        if ref_id in self.references:
            self.references[ref_id].enabled = enabled
            self.save()
            return True
        return False

    def update_threshold(self, ref_id: str, threshold: float) -> bool:
        """Updates confidence threshold for a reference item."""
        if ref_id in self.references:
            self.references[ref_id].threshold = max(0.1, min(0.99, float(threshold)))
            self.save()
            return True
        return False

    def get_by_category(self, category: str) -> List[ReferenceImage]:
        """Returns all reference items belonging to a category."""
        return [ref for ref in self.references.values() if ref.category == category]

    def get_enabled_by_category(self, category: str) -> List[ReferenceImage]:
        """Returns all enabled reference items belonging to a category whose files exist."""
        return [
            ref for ref in self.references.values()
            if ref.category == category and ref.enabled and os.path.exists(ref.file_path)
        ]

    def get_all(self) -> List[ReferenceImage]:
        """Returns all registered reference items."""
        return list(self.references.values())
