"""Visualize annotation bboxes on images for manual review.

Usage:
    python annotation/review/visualize.py \\
        --annotations datasets/raw_annotated/annotations.jsonl \\
        --output review_output/ \\
        --n 20 \\
        --stage alpha          # filter by stage
        --min-confidence 0.7
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from annotation.types import AnnotationStore, ImageAnnotation

logger = logging.getLogger(__name__)

# Color per source stage
_STAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "alpha": (0, 255, 0),
    "composite": (0, 200, 255),
    "sam": (255, 165, 0),
    "feature_match": (255, 0, 128),
}
_DEFAULT_COLOR = (255, 255, 0)


def draw_annotations(ann: ImageAnnotation, class_names: list[str] | None = None) -> Image.Image:
    img = Image.open(ann.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for bbox in ann.bboxes:
        color = _STAGE_COLORS.get(bbox.source, _DEFAULT_COLOR)
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color, width=2)

        label = str(bbox.pokemon_id)
        if class_names and 0 <= bbox.pokemon_id < len(class_names):
            label = class_names[bbox.pokemon_id]
        text = f"{label} {bbox.confidence:.2f}"

        # Text background
        bbox_text = draw.textbbox((bbox.x1, bbox.y1 - 16), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((bbox.x1, bbox.y1 - 16), text, fill=(0, 0, 0), font=font)

    return img


def visualize(
    annotations: list[ImageAnnotation],
    output_dir: Path,
    *,
    n: int | None = None,
    stage_filter: str | None = None,
    min_confidence: float = 0.0,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    filtered = [
        a for a in annotations
        if Path(a.image_path).exists()
        and (stage_filter is None or a.stage == stage_filter)
        and any(b.confidence >= min_confidence for b in a.bboxes)
    ]

    if n is not None:
        filtered = random.sample(filtered, min(n, len(filtered)))

    try:
        from pokedex_cv.pokemon import get_class_names
        class_names = get_class_names("en")
    except Exception:
        class_names = None

    for ann in filtered:
        img = draw_annotations(ann, class_names)
        src_name = Path(ann.image_path).stem
        out_path = output_dir / f"{src_name}_review.jpg"
        img.save(out_path, quality=90)

    logger.info("Saved %d review images → %s", len(filtered), output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("review_output"))
    p.add_argument("--n", type=int, default=None, help="Max images to visualize")
    p.add_argument("--stage", default=None, help="Filter by stage (alpha/composite/sam)")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    store = AnnotationStore(args.annotations)
    annotations = store.load_all()
    logger.info("Loaded %d records", len(annotations))

    visualize(
        annotations,
        args.output,
        n=args.n,
        stage_filter=args.stage,
        min_confidence=args.min_confidence,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
