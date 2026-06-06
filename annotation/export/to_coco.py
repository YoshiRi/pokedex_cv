"""Export annotations to COCO format.

Output: datasets/{name}/annotations.json

COCO format:
    {
      "images": [{"id":1, "file_name":"...", "width":640, "height":640}],
      "annotations": [{"id":1, "image_id":1, "category_id":25,
                        "bbox":[x,y,w,h], "area":..., "iscrowd":0}],
      "categories": [{"id":1, "name":"Bulbasaur"}, ...]
    }

Usage:
    python annotation/export/to_coco.py \\
        --annotations datasets/raw_annotated/annotations.jsonl \\
        --output datasets/pokemon_detection/annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from annotation.schema import AnnotationStore, ImageAnnotation

logger = logging.getLogger(__name__)


def export_coco(
    annotations: list[ImageAnnotation],
    output_path: Path,
    *,
    min_confidence: float = 0.0,
    class_names: list[str] | None = None,
) -> None:
    if class_names is None:
        try:
            from pokedex_cv.pokemon import get_class_names
            class_names = get_class_names("en")
        except Exception:
            logger.warning("Cannot load class names; using numeric labels")
            max_id = max(
                (b.pokemon_id for a in annotations for b in a.bboxes),
                default=0,
            )
            class_names = [str(i) for i in range(max_id + 1)]

    categories = [
        {"id": i, "name": name, "supercategory": "pokemon"}
        for i, name in enumerate(class_names)
        if i > 0  # skip background (id=0)
    ]

    images = []
    coco_annotations = []
    ann_id = 1

    for img_id, ann in enumerate(annotations, start=1):
        images.append({
            "id": img_id,
            "file_name": Path(ann.image_path).name,
            "width": ann.width,
            "height": ann.height,
        })
        for bbox in ann.bboxes:
            if bbox.confidence < min_confidence:
                continue
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bbox.pokemon_id,
                "bbox": [bbox.x1, bbox.y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "confidence": bbox.confidence,
                "source": bbox.source,
            })
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"images": images, "annotations": coco_annotations, "categories": categories},
            f, ensure_ascii=False, indent=2,
        )
    logger.info(
        "COCO export done: %d images, %d annotations → %s",
        len(images), len(coco_annotations), output_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path (e.g. datasets/pokemon_detection/annotations.json)")
    p.add_argument("--min-confidence", type=float, default=0.0)
    args = p.parse_args()

    store = AnnotationStore(args.annotations)
    annotations = store.load_all()
    logger.info("Loaded %d annotation records", len(annotations))

    export_coco(annotations, args.output, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
