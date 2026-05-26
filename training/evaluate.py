"""Evaluate trained YOLO model.

Two modes:
  val   — run YOLO validation on synthetic val split (requires ground truth)
  real  — run inference on a directory of real images (no ground truth needed)

Usage:
    # Synthetic validation (default)
    python training/evaluate.py --config configs/poc_20species.yaml

    # Explicit weights
    python training/evaluate.py \\
        --weights runs/train/poc_20species_yolov8n/weights/best.pt \\
        --data datasets/poc_20species/yolo/data.yaml

    # Real-image test
    python training/evaluate.py \\
        --config configs/poc_20species.yaml \\
        --mode real \\
        --images /path/to/real_pokemon_photos/ \\
        --output runs/eval/real_test/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_weights(cfg: dict, args: argparse.Namespace) -> Path:
    if args.weights:
        return Path(args.weights)
    train_cfg = cfg.get("training", {})
    project = Path(args.project or train_cfg.get("project", "runs/train"))
    name = args.name or train_cfg.get("name", cfg.get("name", "yolo_train"))
    best = project / name / "weights" / "best.pt"
    if not best.exists():
        logger.error("Weights not found at %s. Run training first or pass --weights.", best)
        sys.exit(1)
    return best


def _resolve_data_yaml(cfg: dict, args: argparse.Namespace) -> str | None:
    if args.data:
        return args.data
    export_dir = cfg.get("export", {}).get("output_dir")
    if export_dir:
        return str(Path(export_dir) / "data.yaml")
    return None


def evaluate_val(weights: Path, data_yaml: str, imgsz: int, batch: int, device: str | None) -> None:
    """Run YOLO validation on the synthetic val split."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    kwargs: dict = {
        "data": data_yaml,
        "imgsz": imgsz,
        "batch": batch,
        "verbose": True,
    }
    if device is not None:
        kwargs["device"] = device

    logger.info("Running validation on: %s", data_yaml)
    metrics = model.val(**kwargs)

    # Print summary
    print("\n=== Validation Results ===")
    print(f"mAP@50       : {metrics.box.map50:.4f}")
    print(f"mAP@50:95    : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print("\nPer-class AP@50:")
        names = model.names
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"  {names.get(idx, idx):20s}: {ap:.4f}")


def evaluate_real(
    weights: Path,
    images_dir: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> None:
    """Run inference on real images and save annotated results."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for p in images_dir.glob(ext)
    )
    if not image_paths:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Running inference on %d images → %s", len(image_paths), output_dir)

    predict_kwargs: dict = {
        "source": str(images_dir),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "save": True,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
    }
    if device is not None:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)

    # Summary: count detections per class
    from collections import Counter
    class_counts: Counter = Counter()
    for r in results:
        for cls_id in r.boxes.cls.int().tolist():
            class_names = model.names
            class_counts[class_names.get(cls_id, str(cls_id))] += 1

    print("\n=== Real-Image Detection Summary ===")
    print(f"Images processed : {len(results)}")
    print(f"Total detections : {sum(class_counts.values())}")
    print("\nDetections per class:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {count}")
    print(f"\nAnnotated images saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Pokemon YOLO model")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment config YAML")
    p.add_argument("--mode", choices=["val", "real"], default="val",
                   help="val: synthetic validation; real: inference on real images")

    # Model
    p.add_argument("--weights", type=str, default=None,
                   help="Path to model weights (.pt). Defaults to best.pt in training run.")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)

    # Data
    p.add_argument("--data", type=str, default=None,
                   help="YOLO data.yaml (val mode)")
    p.add_argument("--images", type=Path, default=None,
                   help="Directory of real images (real mode)")
    p.add_argument("--output", type=Path, default=Path("runs/eval/real_test"),
                   help="Output directory for real-mode annotated images")

    # Inference params
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    weights = _resolve_weights(cfg, args)

    if args.mode == "val":
        data_yaml = _resolve_data_yaml(cfg, args)
        if not data_yaml:
            logger.error("No data.yaml found. Use --data or set export.output_dir in config.")
            sys.exit(1)
        evaluate_val(weights, data_yaml, args.imgsz, args.batch, args.device)

    else:  # real
        if args.images is None:
            logger.error("--images required for --mode real")
            sys.exit(1)
        evaluate_real(
            weights,
            args.images,
            args.output,
            args.imgsz,
            args.conf,
            args.iou,
            args.device,
        )


if __name__ == "__main__":
    main()
