"""Quality filter: resolution, transparency, and blur checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    min_width: int = 32
    min_height: int = 32
    min_nontransparent_pixels: int = 10
    min_laplacian_variance: float = 50.0
    # Images smaller than this (in px on longest side) skip the blur check.
    # Game sprites are intentionally pixelated so blur detection is unreliable.
    small_sprite_threshold: int = 128


@dataclass
class QualityResult:
    path: Path
    passed: bool
    reason: str = ""


class QualityFilter:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def run(self, image_paths: list[Path]) -> tuple[list[Path], list[Path]]:
        """Return (kept, removed) lists."""
        kept: list[Path] = []
        removed: list[Path] = []

        for path in image_paths:
            result = self.check(path)
            if result.passed:
                kept.append(path)
            else:
                logger.debug("fail %s (%s)", path, result.reason)
                path.unlink(missing_ok=True)
                removed.append(path)

        logger.info("quality: kept=%d removed=%d", len(kept), len(removed))
        return kept, removed

    def check(self, path: Path) -> QualityResult:
        try:
            img = Image.open(path)
        except Exception as e:
            return QualityResult(path, False, f"cannot open: {e}")

        w, h = img.size
        cfg = self.config

        if w < cfg.min_width or h < cfg.min_height:
            return QualityResult(path, False, f"too small: {w}×{h}")

        if img.mode == "RGBA":
            alpha = np.array(img)[:, :, 3]
            if int(alpha.sum()) < cfg.min_nontransparent_pixels:
                return QualityResult(path, False, "nearly fully transparent")

        # Skip blur check for small game sprites (intentionally pixelated)
        if max(w, h) >= cfg.small_sprite_threshold:
            gray = np.array(img.convert("L"))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if laplacian_var < cfg.min_laplacian_variance:
                return QualityResult(path, False, f"blurry: laplacian={laplacian_var:.1f}")

        return QualityResult(path, True)
