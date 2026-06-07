"""Export trained YOLO model to deployment formats.

Supported formats: onnx, torchscript, openvino, tflite, coreml
Default: onnx (broadest compatibility)

Usage:
    # ONNX export (default)
    python training/export.py --config configs/poc_20species.yaml

    # Specific weights and format
    python training/export.py \\
        --weights runs/train/poc_20species_yolov8n/weights/best.pt \\
        --format onnx \\
        --imgsz 640

    # Multiple formats
    python training/export.py --config configs/poc_20species.yaml --format onnx torchscript
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ["onnx", "torchscript", "openvino", "tflite", "coreml", "engine"]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_weights(cfg: dict, args: argparse.Namespace) -> Path:
    if args.weights:
        weights = Path(args.weights)
    else:
        train_cfg = cfg.get("training", {})
        project = Path(args.project or train_cfg.get("project", "runs/train"))
        name = args.name or train_cfg.get("name", cfg.get("name", "yolo_train"))
        weights = project / name / "weights" / "best.pt"

    if not weights.exists():
        # Fail fast with a clear message — otherwise Ultralytics treats a missing
        # .pt path as a model name and tries to fetch it from GitHub releases.
        logger.error("Weights not found at %s. Run training first or pass --weights.", weights)
        sys.exit(1)
    return weights


def export_model(weights: Path, fmt: str, imgsz: int, device: str | None, half: bool) -> Path:
    """Export model to the given format. Returns path to exported file."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    kwargs: dict = {
        "format": fmt,
        "imgsz": imgsz,
        "half": half,
    }
    if device is not None:
        kwargs["device"] = device

    logger.info("Exporting to %s format (imgsz=%d, half=%s)...", fmt, imgsz, half)
    export_path = model.export(**kwargs)
    logger.info("Exported → %s", export_path)
    return Path(export_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO Pokemon detector")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment config YAML")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to .pt weights file")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument(
        "--format",
        nargs="+",
        default=["onnx"],
        choices=SUPPORTED_FORMATS,
        metavar="FORMAT",
        help=f"Export format(s): {SUPPORTED_FORMATS}. Default: onnx",
    )
    p.add_argument("--imgsz", type=int, default=None,
                   help="Inference size (default: training.imgsz from --config, else 640)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--half", action="store_true",
                   help="Export FP16 (requires CUDA or CoreML)")
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
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("training", {}).get("imgsz", 640)
    logger.info("Resolved params: weights=%s imgsz=%d formats=%s", weights, imgsz, args.format)

    for fmt in args.format:
        export_model(weights, fmt, imgsz, args.device, args.half)


if __name__ == "__main__":
    main()
