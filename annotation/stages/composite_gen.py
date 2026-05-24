"""Stage 2: generate synthetic images by compositing sprites onto backgrounds.

Sprites are pasted at random positions and scales.
Bboxes are computed from the exact paste coordinates → confidence=1.0.
Supports 1-N Pokemon per composite image.

Background priority:
  1. Images from backgrounds_dir (if provided and non-empty)
  2. Procedurally generated backgrounds (solid, gradient, noise)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from annotation.schema import AnnotationStore, BBoxAnnotation, ImageAnnotation

logger = logging.getLogger(__name__)


@dataclass
class CompositeConfig:
    output_dir: Path
    num_composites: int = 5          # composites per sprite
    output_size: tuple[int, int] = (640, 640)
    min_scale: float = 0.05          # sprite relative to output_size
    max_scale: float = 0.50
    min_pokemon: int = 1             # Pokemon per composite
    max_pokemon: int = 3
    backgrounds_dir: Path | None = None
    seed: int | None = None


class CompositeGenStage:
    def __init__(self, config: CompositeConfig) -> None:
        self.cfg = config
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        self._bg_paths = self._load_bg_paths()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        sprite_annotations: list[ImageAnnotation],
        store: AnnotationStore,
    ) -> int:
        """Generate composites from annotated sprites and append to store.

        Returns the number of composite images generated.
        """
        sprites = self._load_sprites(sprite_annotations)
        if not sprites:
            logger.warning("No valid sprites found for composite generation")
            return 0

        count = 0
        for sprite_img, pokemon_id in sprites:
            for i in range(self.cfg.num_composites):
                ann = self._make_composite(sprite_img, pokemon_id, count)
                if ann is not None:
                    store.append(ann)
                    count += 1

        logger.info("composite_gen: generated %d composites from %d sprites", count, len(sprites))
        return count

    # ------------------------------------------------------------------
    # Composite creation
    # ------------------------------------------------------------------

    def _make_composite(
        self,
        primary_sprite: Image.Image,
        primary_id: int,
        index: int,
    ) -> ImageAnnotation | None:
        W, H = self.cfg.output_size
        bg = self._get_background(W, H)
        canvas = bg.copy().convert("RGBA")

        bboxes: list[BBoxAnnotation] = []

        n_pokemon = random.randint(self.cfg.min_pokemon, self.cfg.max_pokemon)
        # First slot is always the primary sprite to guarantee it appears
        sprites_to_paste = [(primary_sprite, primary_id)]
        # Additional random sprites share the same source list for now;
        # callers can extend this by passing a sprite pool separately.
        for _ in range(n_pokemon - 1):
            sprites_to_paste.append((primary_sprite, primary_id))

        for sprite, pid in sprites_to_paste:
            result = self._paste_sprite(canvas, sprite, pid)
            if result is not None:
                bboxes.append(result)

        if not bboxes:
            return None

        out_img = canvas.convert("RGB")
        out_path = self.cfg.output_dir / f"composite_{index:06d}.jpg"
        out_img.save(out_path, quality=95)

        return ImageAnnotation(
            image_path=str(out_path),
            width=W,
            height=H,
            bboxes=bboxes,
            stage="composite",
        )

    def _paste_sprite(
        self,
        canvas: Image.Image,
        sprite: Image.Image,
        pokemon_id: int,
    ) -> BBoxAnnotation | None:
        W, H = canvas.size
        scale = random.uniform(self.cfg.min_scale, self.cfg.max_scale)
        new_w = max(1, int(W * scale))
        new_h = max(1, int(new_w * sprite.height / sprite.width))

        resized = sprite.resize((new_w, new_h), Image.LANCZOS)

        max_x = W - new_w
        max_y = H - new_h
        if max_x <= 0 or max_y <= 0:
            return None

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        canvas.paste(resized, (x, y), mask=resized.split()[3] if resized.mode == "RGBA" else None)

        return BBoxAnnotation(
            pokemon_id=pokemon_id,
            x1=x, y1=y,
            x2=x + new_w, y2=y + new_h,
            confidence=1.0,
            source="composite",
        )

    # ------------------------------------------------------------------
    # Background helpers
    # ------------------------------------------------------------------

    def _load_bg_paths(self) -> list[Path]:
        if self.cfg.backgrounds_dir and self.cfg.backgrounds_dir.is_dir():
            paths = list(self.cfg.backgrounds_dir.glob("*.[jp][pn]g"))
            paths += list(self.cfg.backgrounds_dir.glob("*.jpeg"))
            if paths:
                logger.info("Loaded %d background images from %s", len(paths), self.cfg.backgrounds_dir)
                return paths
        logger.info("No background images found; using procedural backgrounds")
        return []

    def _get_background(self, W: int, H: int) -> Image.Image:
        if self._bg_paths:
            path = random.choice(self._bg_paths)
            try:
                return Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
            except Exception:
                pass
        return self._generate_background(W, H)

    @staticmethod
    def _generate_background(W: int, H: int) -> Image.Image:
        choice = random.choice([
            "solid", "gradient_h", "gradient_v",
            "noise_smooth", "sky", "grass", "checkerboard", "bokeh",
        ])
        arr = np.zeros((H, W, 3), dtype=np.uint8)

        if choice == "solid":
            arr[:] = [random.randint(0, 255) for _ in range(3)]

        elif choice == "gradient_h":
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            t = np.linspace(0, 1, W)[np.newaxis, :, np.newaxis]
            arr[:] = (c1 * (1 - t) + c2 * t).astype(np.uint8)

        elif choice == "gradient_v":
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            t = np.linspace(0, 1, H)[:, np.newaxis, np.newaxis]
            arr[:] = (c1 * (1 - t) + c2 * t).astype(np.uint8)

        elif choice == "noise_smooth":
            # Low-frequency noise: scale down then up to blur
            scale = random.randint(4, 16)
            small = np.random.randint(0, 255, (H // scale + 1, W // scale + 1, 3), dtype=np.uint8)
            from PIL import Image as _Image
            arr = np.array(
                _Image.fromarray(small).resize((W, H), _Image.BILINEAR)
            )

        elif choice == "sky":
            # Blue sky gradient with optional white clouds (blobs)
            sky_top = np.array([random.randint(80, 135), random.randint(150, 210), random.randint(220, 255)])
            sky_bot = np.array([random.randint(160, 220), random.randint(210, 240), 255])
            t = np.linspace(0, 1, H)[:, np.newaxis, np.newaxis]
            arr[:] = (sky_top * (1 - t) + sky_bot * t).astype(np.uint8)
            # Add a few cloud blobs
            for _ in range(random.randint(2, 6)):
                cx, cy = random.randint(0, W), random.randint(0, H // 2)
                rx, ry = random.randint(W // 8, W // 3), random.randint(H // 16, H // 8)
                yy, xx = np.ogrid[:H, :W]
                mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
                alpha = np.clip(1 - (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1)
                alpha[~mask] = 0
                for c in range(3):
                    arr[:, :, c] = np.clip(
                        arr[:, :, c] * (1 - alpha * 0.7) + 255 * alpha * 0.7, 0, 255
                    ).astype(np.uint8)

        elif choice == "grass":
            # Green ground with slight variation
            base = np.array([random.randint(30, 80), random.randint(100, 160), random.randint(30, 70)])
            arr[:] = base
            variation = np.random.randint(-25, 25, (H, W, 3))
            arr = np.clip(arr.astype(int) + variation, 0, 255).astype(np.uint8)
            # Darker at the bottom (shadow)
            shadow = np.linspace(1.0, 0.75, H)[:, np.newaxis, np.newaxis]
            arr = (arr * shadow).astype(np.uint8)

        elif choice == "checkerboard":
            tile = random.randint(W // 16, W // 4)
            c1 = np.array([random.randint(0, 200) for _ in range(3)])
            c2 = np.array([random.randint(0, 200) for _ in range(3)])
            yy, xx = np.mgrid[:H, :W]
            mask = ((yy // tile) + (xx // tile)) % 2 == 0
            arr[mask] = c1
            arr[~mask] = c2

        elif choice == "bokeh":
            # Blurred random colored circles simulating bokeh background
            bg = np.array([random.randint(0, 100) for _ in range(3)], dtype=np.float32)
            arr[:] = bg.astype(np.uint8)
            for _ in range(random.randint(8, 20)):
                cx, cy = random.randint(0, W), random.randint(0, H)
                r = random.randint(W // 12, W // 4)
                color = np.array([random.randint(50, 255) for _ in range(3)], dtype=np.float32)
                yy, xx = np.ogrid[:H, :W]
                dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                alpha = np.clip(1 - dist / r, 0, 1) ** 2
                for c in range(3):
                    arr[:, :, c] = np.clip(
                        arr[:, :, c] * (1 - alpha * 0.6) + color[c] * alpha * 0.6, 0, 255
                    ).astype(np.uint8)

        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # Sprite loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sprites(
        annotations: list[ImageAnnotation],
    ) -> list[tuple[Image.Image, int]]:
        sprites: list[tuple[Image.Image, int]] = []
        for ann in annotations:
            if ann.stage != "alpha" or not ann.bboxes:
                continue
            try:
                img = Image.open(ann.image_path).convert("RGBA")
                sprites.append((img, ann.bboxes[0].pokemon_id))
            except Exception as e:
                logger.debug("Cannot load %s: %s", ann.image_path, e)
        return sprites
