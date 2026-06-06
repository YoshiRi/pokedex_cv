"""Perceptual hash-based duplicate filter.

Computes pHash for each image and removes near-duplicates (Hamming distance ≤ threshold).

Key design: sprites from the same Pokemon directory are NEVER compared against each
other — pHash cannot distinguish shiny vs. non-shiny since it operates on grayscale
structure. Only cross-Pokemon comparisons are performed.

Hash DB is persisted as JSON so repeated runs skip already-seen files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

_DB_FILENAME = ".dedup_hashes.json"


@dataclass
class _HashEntry:
    ph: str          # hex pHash string
    path: str        # relative path to the image
    pokemon_id: int | None  # parsed from 4-digit directory name


class DedupFilter:
    def __init__(self, output_dir: Path, *, phash_threshold: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.ph_threshold = phash_threshold
        self._db_path = self.output_dir / _DB_FILENAME
        self._db: dict[str, _HashEntry] = self._load_db()

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
            ph = imagehash.phash(Image.open(path))
        except Exception as e:
            logger.warning("Cannot hash %s: %s", path, e)
            return False

        ph_hex = str(ph)
        pokemon_id = self._parse_pokemon_id(path)

        for entry in self._db.values():
            # Never dedup sprites from the same Pokemon: pHash can't distinguish
            # shiny vs. non-shiny (same structure, different color)
            if pokemon_id is not None and entry.pokemon_id == pokemon_id:
                continue
            if ph - imagehash.hex_to_hash(entry.ph) <= self.ph_threshold:
                return True

        try:
            rel = str(path.relative_to(self.output_dir))
        except ValueError:
            rel = str(path)

        self._db[ph_hex] = _HashEntry(ph=ph_hex, path=rel, pokemon_id=pokemon_id)
        return False

    @staticmethod
    def _parse_pokemon_id(path: Path) -> int | None:
        """Extract pokemon_id from 4-digit parent directory name."""
        for part in path.parts:
            if len(part) == 4 and part.isdigit():
                return int(part)
        return None

    def _load_db(self) -> dict[str, _HashEntry]:
        if not self._db_path.exists():
            return {}
        raw = json.loads(self._db_path.read_text(encoding="utf-8"))
        db: dict[str, _HashEntry] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                db[k] = _HashEntry(**v)
            else:
                # Old format (str value): discard and rebuild
                logger.info("Resetting dedup DB (old format detected)")
                return {}
        return db

    def _save_db(self) -> None:
        self._db_path.write_text(
            json.dumps({k: asdict(v) for k, v in self._db.items()}, indent=2),
            encoding="utf-8",
        )
