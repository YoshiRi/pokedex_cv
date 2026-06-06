"""Export annotations to Ultralytics YOLO format.

Output structure:
    datasets/{name}/
    ├── images/train/  images/val/  images/test/
    ├── labels/train/  labels/val/  labels/test/
    └── data.yaml

Label file format (one bbox per line):
    {class_id} {cx} {cy} {w} {h}   (all values normalized 0-1)

class_map (optional): ordered list of national Pokedex IDs.
    Index = YOLO class_id, value = Pokedex number stored in BBoxAnnotation.
    If omitted, bbox.pokemon_id is written as-is.

Usage:
    # With experiment config (recommended for PoC)
    python annotation/export/to_yolo.py \\
        --config configs/poc_20species.yaml \\
        --annotations datasets/poc_20species/raw_annotated/annotations.jsonl \\
        --output datasets/poc_20species/yolo

    # Manual
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


def _unique_stem(src: Path) -> str:
    """Build a globally unique output stem for an image file.

    Sprites live in 4-digit Pokemon ID directories (e.g. 0001/front_default.png).
    Without prefixing, all front_default.png files from different Pokemon overwrite
    each other in the flat YOLO output dir.

    Rule:
      - 4-digit parent dir (sprite) → "{pokemon_id}_{stem}"  e.g. "0001_front_default"
      - anything else (composite)  → original stem            e.g. "composite_000000"
    """
    parent = src.parent.name
    if len(parent) == 4 and parent.isdigit():
        return f"{parent}_{src.stem}"
    return src.stem


def export_yolo(
    annotations: list[ImageAnnotation],
    output_dir: Path,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    min_confidence: float = 0.0,
    class_map: list[int] | None = None,
    seed: int = 42,
    clean: bool = False,
) -> None:
    """Export annotations to YOLO format.

    Args:
        annotations: list of ImageAnnotation records.
        output_dir: destination root; images/ labels/ data.yaml written here.
        split: (train, val, test) fractions summing to 1.0.
        min_confidence: bboxes below this threshold are skipped.
        class_map: ordered list of national Pokedex IDs → index = YOLO class_id.
            Bboxes whose pokemon_id is not in the map are skipped.
            If None, pokemon_id is used directly as class_id.
        seed: random seed for split shuffling.
        clean: if True, remove output_dir before writing (guarantees fresh dataset).
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Removed existing output dir: %s (clean=True)", output_dir)

    pokedex_to_class: dict[int, int] | None = None
    if class_map is not None:
        pokedex_to_class = {pid: idx for idx, pid in enumerate(class_map)}

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

            stem = _unique_stem(src)
            dst_img = img_dir / (stem + src.suffix)
            shutil.copy2(src, dst_img)

            lines = []
            for bbox in ann.bboxes:
                if bbox.confidence < min_confidence:
                    continue
                if pokedex_to_class is not None:
                    class_id = pokedex_to_class.get(bbox.pokemon_id)
                    if class_id is None:
                        logger.debug("pokemon_id %d not in class_map; skipping bbox", bbox.pokemon_id)
                        continue
                else:
                    class_id = bbox.pokemon_id
                cx, cy, w, h = bbox.to_yolo(ann.width, ann.height)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if lines:
                lbl_path = lbl_dir / (stem + ".txt")
                lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger.info("  %s: %d images", split_name, len(split_anns))

    _write_data_yaml(output_dir, class_map)
    logger.info("YOLO export done → %s", output_dir)


def _write_data_yaml(output_dir: Path, class_map: list[int] | None) -> None:
    class_names: list[str] | None = None

    if class_map is not None:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from pokedex_cv.pokemon import get_name
            class_names = [get_name(pid, "en") for pid in class_map]
        except Exception:
            logger.warning("Cannot load Pokemon names; using pokedex_id strings")
            class_names = [str(pid) for pid in class_map]
    else:
        try:
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


def _resolve_class_map(cfg: dict) -> list[int] | None:
    """Determine class_map from experiment config.

    Priority:
      1. Explicit ``class_map`` list in config → use as-is.
      2. ``class_map`` absent/null + ``pokemon_ids`` list → sorted pokemon_ids.
      3. ``class_map`` absent/null + ``pokemon_ids`` null → all IDs from
         pokemon_classes.yaml (full 1025-species run).
      4. Nothing resolvable → return None (bbox.pokemon_id used directly).
    """
    explicit = cfg.get("class_map")
    if explicit is not None:
        return explicit

    pokemon_ids = cfg.get("pokemon_ids")
    if pokemon_ids:
        return sorted(pokemon_ids)

    # Try to load all IDs from pokemon_classes.yaml
    classes_path = Path(__file__).parent.parent.parent / "pokemon_classes.yaml"
    if classes_path.exists():
        try:
            data = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
            all_ids = sorted(data["pokemon"].keys())
            logger.info("class_map: auto-generated %d classes from pokemon_classes.yaml", len(all_ids))
            return all_ids
        except Exception as e:
            logger.warning("Cannot auto-generate class_map: %s", e)

    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment config YAML (class_map, export settings read from here)")
    p.add_argument("--split", type=float, nargs=3, default=None,
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--clean", action="store_true",
                   help="Remove output dir before writing (guarantees fresh dataset)")
    args = p.parse_args()

    cfg: dict = {}
    if args.config:
        with args.config.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    export_cfg = cfg.get("export", {})
    class_map: list[int] | None = _resolve_class_map(cfg)
    split = tuple(args.split or export_cfg.get("split", [0.8, 0.1, 0.1]))
    min_confidence = args.min_confidence if args.min_confidence is not None else export_cfg.get("min_confidence", 0.0)
    seed = args.seed if args.seed is not None else export_cfg.get("seed", 42)

    # Resolve output: CLI --output overrides config export.output_dir
    output_dir = args.output or Path(export_cfg["output_dir"])

    store = AnnotationStore(args.annotations)
    annotations = store.load_all()
    logger.info("Loaded %d annotation records", len(annotations))

    export_yolo(
        annotations,
        output_dir,
        split,
        min_confidence=min_confidence,
        class_map=class_map,
        seed=seed,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
