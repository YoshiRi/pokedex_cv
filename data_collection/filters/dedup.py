"""Perceptual hash-based duplicate filter.

Design decisions:
  - Same-Pokemon sprites (same {id} directory) are NEVER deduplicated.
    Shiny sprites differ only in color; phash (structure-based, grayscale) cannot
    distinguish them and would incorrectly flag them as duplicates.
  - Cross-Pokemon dedup uses a stricter phash threshold (default 3, vs naive 5)
    to reduce false positives between structurally similar sprites.
  - colorhash was evaluated but is ineffective for transparent-background sprites:
    the transparent area (alpha=0 → black) overwhelms the hue distribution,
    making colorhash nearly identical for sprites with different colors.

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
    ph: str          # phash hex
    path: str
    pokemon_id: str  # 4-digit dir name (e.g. "0025"); "" if not parseable


class DedupFilter:
    def __init__(
        self,
        output_dir: Path,
        *,
        phash_threshold: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.ph_threshold = phash_threshold
        self._db_path = self.output_dir / _DB_FILENAME
        self._db: dict[str, _HashEntry] = self._load_db()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image_paths: list[Path]) -> tuple[list[Path], list[Path]]:
        """Return (kept, removed) after deduplication."""
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

    @staticmethod
    def _parse_pokemon_id(path: Path) -> str:
        for part in path.parts:
            if len(part) == 4 and part.isdigit():
                return part
        return ""

    def _is_duplicate(self, path: Path) -> bool:
        try:
            ph = imagehash.phash(Image.open(path))
        except Exception as e:
            logger.warning("Cannot hash %s: %s", path, e)
            return False

        pokemon_id = self._parse_pokemon_id(path)

        for entry in self._db.values():
            # Never compare sprites from the same Pokemon
            if pokemon_id and entry.pokemon_id == pokemon_id:
                continue

            if ph - imagehash.hex_to_hash(entry.ph) <= self.ph_threshold:
                logger.debug("dup %s <-> %s", path.name, entry.path)
                return True

        self._db[str(ph)] = _HashEntry(
            ph=str(ph),
            path=str(path.relative_to(self.output_dir)),
            pokemon_id=pokemon_id,
        )
        return False

    def _load_db(self) -> dict[str, _HashEntry]:
        if not self._db_path.exists():
            return {}
        try:
            raw = json.loads(self._db_path.read_text())
            # Detect and discard old format (str values instead of dict)
            if raw and isinstance(next(iter(raw.values())), str):
                logger.info("dedup: old DB format detected, resetting")
                return {}
            return {k: _HashEntry(**v) for k, v in raw.items()}
        except Exception as e:
            logger.warning("Cannot load dedup DB (%s) — starting fresh", e)
            return {}

    def _save_db(self) -> None:
        self._db_path.write_text(
            json.dumps({k: asdict(v) for k, v in self._db.items()}, ensure_ascii=False, indent=2)
        )
