"""Validate an exported YOLO dataset for common integrity issues.

Checks:
  - Every image has a corresponding label file (and vice versa)
  - All bbox coordinates are in [0, 1]
  - All class_ids are in [0, nc-1]
  - No bbox with near-zero area (w < 1e-4 or h < 1e-4)
  - No duplicate filenames across splits

Usage:
    python dataset/validate.py --dataset datasets/poc_20species/yolo
    python dataset/validate.py --dataset datasets/poc_20species/yolo --split train val
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class ValidationResult:
    split: str
    n_images: int = 0
    n_labels: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _load_data_yaml(dataset_dir: Path) -> dict:
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")
    with yaml_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_split(
    dataset_dir: Path,
    split: str,
    nc: int,
    *,
    min_dim: float = 1e-4,
) -> ValidationResult:
    result = ValidationResult(split=split)

    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split

    if not img_dir.exists():
        result.warnings.append(f"images/{split}/ directory not found — skipping")
        return result

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for p in img_dir.glob(ext)
    )
    result.n_images = len(image_paths)

    seen_stems: set[str] = set()
    for img_path in image_paths:
        stem = img_path.stem

        if stem in seen_stems:
            result.errors.append(f"Duplicate filename: {img_path.name}")
        seen_stems.add(stem)

        lbl_path = lbl_dir / (stem + ".txt")
        if not lbl_path.exists():
            result.warnings.append(f"Missing label: {lbl_path}")
            continue

        result.n_labels += 1
        _validate_label_file(lbl_path, nc, min_dim, result)

    # Labels without matching image
    if lbl_dir.exists():
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            if lbl_path.stem not in seen_stems:
                result.warnings.append(f"Label without image: {lbl_path.name}")

    return result


def _validate_label_file(
    lbl_path: Path,
    nc: int,
    min_dim: float,
    result: ValidationResult,
) -> None:
    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        result.warnings.append(f"Empty label file: {lbl_path.name}")
        return

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            result.errors.append(
                f"{lbl_path.name}:{i+1} — expected 5 fields, got {len(parts)}: {line!r}"
            )
            continue

        try:
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            result.errors.append(f"{lbl_path.name}:{i+1} — non-numeric value: {line!r}")
            continue

        if not (0 <= class_id < nc):
            result.errors.append(
                f"{lbl_path.name}:{i+1} — class_id {class_id} out of range [0, {nc-1}]"
            )

        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if not (0.0 <= val <= 1.0):
                result.errors.append(
                    f"{lbl_path.name}:{i+1} — {name}={val:.6f} not in [0, 1]"
                )

        if w < min_dim:
            result.warnings.append(f"{lbl_path.name}:{i+1} — bbox width too small: {w:.6f}")
        if h < min_dim:
            result.warnings.append(f"{lbl_path.name}:{i+1} — bbox height too small: {h:.6f}")


def _collect_stems(dataset_dir: Path, split: str) -> set[str]:
    img_dir = dataset_dir / "images" / split
    if not img_dir.exists():
        return set()
    return {
        p.stem
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for p in img_dir.glob(ext)
    }


def validate_dataset(
    dataset_dir: Path,
    splits: tuple[str, ...] = SPLITS,
) -> bool:
    data = _load_data_yaml(dataset_dir)
    nc: int = data["nc"]
    names: list[str] = data.get("names", [])
    logger.info("Dataset: %s  nc=%d  names=%s...", dataset_dir, nc, names[:5])

    all_ok = True
    split_stems: dict[str, set[str]] = {}

    for split in splits:
        result = validate_split(dataset_dir, split, nc)
        split_stems[split] = _collect_stems(dataset_dir, split)

        status = "OK" if result.ok else "FAIL"
        logger.info(
            "[%s] %s — images=%d  labels=%d  errors=%d  warnings=%d",
            status, split, result.n_images, result.n_labels,
            len(result.errors), len(result.warnings),
        )

        for msg in result.errors:
            logger.error("  ERROR   %s", msg)
        for msg in result.warnings:
            logger.warning("  WARNING %s", msg)

        if not result.ok:
            all_ok = False

    # Cross-split duplicate detection
    split_list = [s for s in splits if s in split_stems]
    for i, s1 in enumerate(split_list):
        for s2 in split_list[i + 1:]:
            overlap = split_stems[s1] & split_stems[s2]
            if overlap:
                examples = sorted(overlap)[:5]
                logger.error(
                    "  ERROR   %d filename(s) appear in both '%s' and '%s': %s%s",
                    len(overlap), s1, s2, examples,
                    " ..." if len(overlap) > 5 else "",
                )
                all_ok = False

    if all_ok:
        logger.info("All checks passed.")

    return all_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLO dataset integrity")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Root directory of YOLO dataset (contains data.yaml)")
    p.add_argument("--split", nargs="+", default=list(SPLITS),
                   choices=list(SPLITS), metavar="SPLIT",
                   help="Splits to validate (default: train val test)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    ok = validate_dataset(args.dataset, tuple(args.split))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
