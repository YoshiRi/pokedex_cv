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


def _iou(
    x1a: int, y1a: int, x2a: int, y2a: int,
    x1b: int, y1b: int, x2b: int, y2b: int,
) -> float:
    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    return inter / (area_a + area_b - inter)


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
    max_iou: float = 0.3             # reject placements overlapping existing bboxes above this IoU
    truncation_prob: float = 0.15    # probability of placing sprite partially outside canvas
    augment_sprites: bool = True     # apply random flip/rotation/color jitter to sprites


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
                ann = self._make_composite(sprite_img, pokemon_id, count, sprite_pool=sprites)
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
        *,
        sprite_pool: list[tuple[Image.Image, int]] | None = None,
    ) -> ImageAnnotation | None:
        W, H = self.cfg.output_size
        bg = self._get_background(W, H)
        canvas = bg.copy().convert("RGBA")

        bboxes: list[BBoxAnnotation] = []

        n_pokemon = random.randint(self.cfg.min_pokemon, self.cfg.max_pokemon)
        # First slot is always the primary sprite to guarantee it appears
        sprites_to_paste = [(primary_sprite, primary_id)]
        # Additional slots draw from the full sprite pool (different species)
        # to teach the model multi-class scenes, falling back to the primary
        # sprite when no pool is available.
        for _ in range(n_pokemon - 1):
            if sprite_pool and len(sprite_pool) > 1:
                sprites_to_paste.append(random.choice(sprite_pool))
            else:
                sprites_to_paste.append((primary_sprite, primary_id))

        for sprite, pid in sprites_to_paste:
            result = self._paste_sprite(canvas, sprite, pid, existing_bboxes=bboxes)
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
        existing_bboxes: list[BBoxAnnotation] | None = None,
    ) -> BBoxAnnotation | None:
        W, H = canvas.size
        scale = random.uniform(self.cfg.min_scale, self.cfg.max_scale)
        new_w = max(1, int(W * scale))
        new_h = max(1, int(new_w * sprite.height / sprite.width))

        resized = sprite.resize((new_w, new_h), Image.LANCZOS)

        if self.cfg.augment_sprites:
            resized = self._augment_sprite(resized)
            new_w, new_h = resized.size

        allow_truncation = random.random() < self.cfg.truncation_prob
        if allow_truncation:
            margin = max(new_w, new_h) // 2
            x = random.randint(-margin, W - new_w + margin)
            y = random.randint(-margin, H - new_h + margin)
        else:
            max_x = W - new_w
            max_y = H - new_h
            if max_x <= 0 or max_y <= 0:
                return None
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

        vis_x1 = max(0, x)
        vis_y1 = max(0, y)
        vis_x2 = min(W, x + new_w)
        vis_y2 = min(H, y + new_h)
        vis_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
        full_area = new_w * new_h
        if vis_area < full_area * 0.25 or vis_x2 <= vis_x1 or vis_y2 <= vis_y1:
            return None

        if existing_bboxes and self.cfg.max_iou < 1.0:
            for eb in existing_bboxes:
                if _iou(vis_x1, vis_y1, vis_x2, vis_y2, eb.x1, eb.y1, eb.x2, eb.y2) > self.cfg.max_iou:
                    return None

        crop_x = max(0, -x)
        crop_y = max(0, -y)
        crop_r = min(new_w, W - x)
        crop_b = min(new_h, H - y)
        cropped = resized.crop((crop_x, crop_y, crop_r, crop_b))
        mask = cropped.split()[3] if cropped.mode == "RGBA" else None
        canvas.paste(cropped, (vis_x1, vis_y1), mask=mask)

        return BBoxAnnotation(
            pokemon_id=pokemon_id,
            x1=vis_x1, y1=vis_y1,
            x2=vis_x2, y2=vis_y2,
            confidence=1.0,
            source="composite",
        )

    @staticmethod
    def _augment_sprite(sprite: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)

        angle = random.uniform(-15, 15)
        if abs(angle) > 1:
            sprite = sprite.rotate(angle, resample=Image.BILINEAR, expand=True)

        if random.random() < 0.5:
            from PIL import ImageEnhance
            brightness = random.uniform(0.7, 1.3)
            sprite = ImageEnhance.Brightness(sprite).enhance(brightness)
            contrast = random.uniform(0.8, 1.2)
            sprite = ImageEnhance.Contrast(sprite).enhance(contrast)

        return sprite

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

    _BG_TYPES = [
        "solid", "gradient_h", "gradient_v", "noise",
        "sky", "grass", "checkerboard", "bokeh",
    ]

    @staticmethod
    def _generate_background(W: int, H: int) -> Image.Image:
        choice = random.choice(CompositeGenStage._BG_TYPES)
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

        elif choice == "noise":
            arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        elif choice == "sky":
            top = np.array([random.randint(80, 180), random.randint(140, 220), random.randint(200, 255)])
            bot = np.array([random.randint(180, 255), random.randint(200, 255), random.randint(220, 255)])
            t = np.linspace(0, 1, H)[:, np.newaxis, np.newaxis]
            arr[:] = (top * (1 - t) + bot * t).astype(np.uint8)

        elif choice == "grass":
            base = np.array([random.randint(30, 80), random.randint(100, 180), random.randint(20, 60)])
            noise = np.random.randint(-30, 30, (H, W, 3), dtype=np.int16)
            arr[:] = np.clip(base + noise, 0, 255).astype(np.uint8)

        elif choice == "checkerboard":
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            cell = random.choice([16, 32, 48, 64])
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            mask = ((yy // cell) + (xx // cell)) % 2 == 0
            arr[mask] = c1
            arr[~mask] = c2

        elif choice == "bokeh":
            base = np.array([random.randint(0, 60) for _ in range(3)])
            arr[:] = base
            yy, xx = np.ogrid[:H, :W]
            for _ in range(random.randint(8, 25)):
                cx, cy = random.randint(0, W), random.randint(0, H)
                r = random.randint(20, min(W, H) // 4)
                color = np.array([random.randint(100, 255) for _ in range(3)])
                alpha = random.uniform(0.15, 0.45)
                dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                mask = dist2 < r * r
                falloff = np.clip(1.0 - np.sqrt(dist2[mask].astype(np.float32)) / r, 0, 1)
                for c in range(3):
                    channel = arr[:, :, c]
                    channel[mask] = np.clip(
                        channel[mask] + (color[c] * alpha * falloff), 0, 255
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
