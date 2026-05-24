"""Perceptual hash-based duplicate filter.

Computes pHash for each image and removes near-duplicates (Hamming distance ≤ threshold).
Hash DB is persisted as JSON so repeated runs skip already-seen files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

_DB_FILENAME = ".dedup_hashes.json"


class DedupFilter:
    def __init__(self, output_dir: Path, *, phash_threshold: int = 5) -> None:
        self.output_dir = output_dir
        self.threshold = phash_threshold
        self._db_path = output_dir / _DB_FILENAME
        self._hash_db: dict[str, str] = self._load_db()  # hex_hash -> relative_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image_paths: list[Path]) -> tuple[list[Path], list[Path]]:
        """Return (kept, removed) lists after deduplication."""
        kept: list[Path] = []
        removed: list[Path] = []

        for path in image_paths:
            if self._is_duplicate(path):
                logger.debug("dup  %s", path)
                path.unlink(missing_ok=True)
                removed.append(path)
            else:
                kept.append(path)

        self._save_db()
        logger.info("dedup: kept=%d removed=%d", len(kept), len(removed))
        return kept, removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, path: Path) -> bool:
        try:
            h = imagehash.phash(Image.open(path))
        except Exception as e:
            logger.warning("Cannot hash %s: %s", path, e)
            return False

        h_hex = str(h)
        for existing_hex in self._hash_db:
            if h - imagehash.hex_to_hash(existing_hex) <= self.threshold:
                return True

        self._hash_db[h_hex] = str(path.relative_to(self.output_dir))
        return False

    def _load_db(self) -> dict[str, str]:
        if self._db_path.exists():
            return json.loads(self._db_path.read_text())
        return {}

    def _save_db(self) -> None:
        self._db_path.write_text(json.dumps(self._hash_db, ensure_ascii=False, indent=2))
