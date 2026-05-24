"""Stage 1: derive bbox from PNG alpha channel.

For sprites with transparent backgrounds the bbox is simply the
bounding rect of all non-transparent pixels. Confidence = 1.0.

Expected input directory layout (produced by PokeAPISpriteScraper):
    raw_images/pokeapi_sprites/{pokemon_id:04d}/{sprite_type}.png

pokemon_id is parsed from the 4-digit directory name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from annotation.types import AnnotationStore, BBoxAnnotation, ImageAnnotation

logger = logging.getLogger(__name__)


class AlphaBBoxStage:
    def __init__(self, *, min_bbox_area: int = 100) -> None:
        self.min_bbox_area = min_bbox_area

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, sprite_dir: Path, store: AnnotationStore) -> int:
        """Annotate all PNGs under sprite_dir and append to store.

        Returns the number of successfully annotated images.
        """
        images = sorted(sprite_dir.rglob("*.png"))
        ok = 0
        for img_path in images:
            ann = self._annotate(img_path)
            if ann is not None:
                store.append(ann)
                ok += 1
            else:
                logger.debug("skip %s (no valid bbox)", img_path)
        logger.info("alpha_bbox: annotated %d / %d images", ok, len(images))
        return ok

    def annotate_single(self, img_path: Path) -> ImageAnnotation | None:
        return self._annotate(img_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotate(self, img_path: Path) -> ImageAnnotation | None:
        pokemon_id = self._parse_pokemon_id(img_path)
        if pokemon_id is None:
            logger.warning("Cannot parse pokemon_id from path: %s", img_path)
            return None

        bbox = self._extract_bbox(img_path)
        if bbox is None:
            return None

        bbox.pokemon_id = pokemon_id
        w, h = Image.open(img_path).size
        return ImageAnnotation(
            image_path=str(img_path),
            width=w,
            height=h,
            bboxes=[bbox],
            stage="alpha",
        )

    def _extract_bbox(self, img_path: Path) -> BBoxAnnotation | None:
        try:
            img = np.array(Image.open(img_path).convert("RGBA"))
        except Exception as e:
            logger.warning("Cannot open %s: %s", img_path, e)
            return None

        alpha = img[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any():
            return None

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        y1, y2 = int(row_indices[0]), int(row_indices[-1])
        x1, x2 = int(col_indices[0]), int(col_indices[-1])

        # x2/y2 are inclusive; make them exclusive (+1) so width = x2-x1
        x2 += 1
        y2 += 1

        if (x2 - x1) * (y2 - y1) < self.min_bbox_area:
            return None

        return BBoxAnnotation(
            pokemon_id=-1,  # filled in by caller
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=1.0,
            source="alpha",
        )

    @staticmethod
    def _parse_pokemon_id(img_path: Path) -> int | None:
        """Extract pokemon_id from 4-digit parent directory name."""
        for part in img_path.parts:
            if len(part) == 4 and part.isdigit():
                return int(part)
        return None
