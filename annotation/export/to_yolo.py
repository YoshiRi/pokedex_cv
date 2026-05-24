"""Export annotations to Ultralytics YOLO format.

Output structure:
    datasets/{name}/
    ├── images/train/  images/val/  images/test/
    ├── labels/train/  labels/val/  labels/test/
    └── data.yaml

Label file format (one bbox per line):
    {class_id} {cx} {cy} {w} {h}   (all values normalized 0-1)

Usage:
    python annotation/export/to_yolo.py \\
        --annotations datasets/raw_annotated/annotations.jsonl \\
        --output datasets/pokemon_detection \\
        --split 0.8 0.1 0.1
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from annotation.schema import AnnotationStore, ImageAnnotation

logger = logging.getLogger(__name__)


def export_yolo(
    annotations: list[ImageAnnotation],
    output_dir: Path,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    min_confidence: float = 0.0,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> None:
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    random.seed(seed)
    anns = [a for a in annotations if a.bboxes]
    random.shuffle(anns)

    n = len(anns)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    splits = {
        "train": anns[:n_train],
        "val": anns[n_train: n_train + n_val],
        "test": anns[n_train + n_val:],
    }

    for split_name, split_anns in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for ann in split_anns:
            src = Path(ann.image_path)
            if not src.exists():
                logger.warning("Image not found: %s", src)
                continue

            dst_img = img_dir / src.name
            shutil.copy2(src, dst_img)

            lines = []
            for bbox in ann.bboxes:
                if bbox.confidence < min_confidence:
                    continue
                cx, cy, w, h = bbox.to_yolo(ann.width, ann.height)
                lines.append(f"{bbox.pokemon_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if lines:
                lbl_path = lbl_dir / (src.stem + ".txt")
                lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger.info("  %s: %d images", split_name, len(split_anns))

    _write_data_yaml(output_dir, class_names)
    logger.info("YOLO export done → %s", output_dir)


def _write_data_yaml(output_dir: Path, class_names: list[str] | None) -> None:
    if class_names is None:
        # Lazy load from pokemon_classes.yaml
        try:
            sys.path.insert(0, str(output_dir.parent.parent))
            from pokedex_cv.pokemon import get_class_names
            class_names = get_class_names("en")
        except Exception:
            logger.warning("Cannot load class names; writing placeholder in data.yaml")
            class_names = ["background"]

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info("Wrote %s (nc=%d)", yaml_path, len(class_names))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    store = AnnotationStore(args.annotations)
    annotations = store.load_all()
    logger.info("Loaded %d annotation records", len(annotations))

    export_yolo(
        annotations,
        args.output,
        tuple(args.split),
        min_confidence=args.min_confidence,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
