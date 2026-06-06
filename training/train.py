"""Train a YOLO model on the Pokemon detection dataset.

Usage:
    # From experiment config (recommended)
    python training/train.py --config configs/poc_20species.yaml

    # Explicit overrides
    python training/train.py \\
        --config configs/poc_20species.yaml \\
        --data datasets/poc_20species/yolo/data.yaml \\
        --model yolov8n.pt \\
        --epochs 50 \\
        --batch 32

    # Resume interrupted run
    python training/train.py --config configs/poc_20species.yaml --resume
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


def build_train_kwargs(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge experiment config training section with CLI overrides."""
    train_cfg = cfg.get("training", {})
    export_cfg = cfg.get("export", {})

    # Resolve data.yaml path from export config if not overridden on CLI
    data_yaml: str | None = args.data
    if data_yaml is None:
        export_dir = export_cfg.get("output_dir")
        if export_dir:
            data_yaml = str(Path(export_dir) / "data.yaml")

    kwargs: dict = {
        "model": args.model or train_cfg.get("model", "yolov8n.pt"),
        "data": data_yaml,
        "epochs": args.epochs or train_cfg.get("epochs", 100),
        "imgsz": args.imgsz or train_cfg.get("imgsz", 640),
        "batch": args.batch or train_cfg.get("batch", 16),
        "patience": train_cfg.get("patience", 20),
        "optimizer": train_cfg.get("optimizer", "auto"),
        "project": args.project or train_cfg.get("project", "runs/train"),
        "name": args.name or train_cfg.get("name", cfg.get("name", "yolo_train")),
        "exist_ok": train_cfg.get("exist_ok", True),
    }

    # Hyperparameter override config YAML
    if args.hyp:
        kwargs["cfg"] = str(args.hyp)

    # Device
    if args.device is not None:
        kwargs["device"] = args.device

    return kwargs


def train(kwargs: dict, resume: bool = False) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_arg = kwargs.pop("model")

    if resume:
        # Resume: pass last.pt as model; Ultralytics handles the rest
        project = Path(kwargs.get("project", "runs/train"))
        name = kwargs.get("name", "")
        last_pt = project / name / "weights" / "last.pt"
        if not last_pt.exists():
            logger.error("Cannot resume: %s not found", last_pt)
            sys.exit(1)
        logger.info("Resuming from %s", last_pt)
        model = YOLO(str(last_pt))
        kwargs["resume"] = True
    else:
        model = YOLO(model_arg)

    logger.info("Starting training with kwargs: %s", kwargs)
    results = model.train(**kwargs)
    logger.info("Training complete. Results saved to %s", results.save_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO Pokemon detector")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment config YAML (configs/poc_20species.yaml)")
    p.add_argument("--data", type=str, default=None,
                   help="Path to YOLO data.yaml (overrides config)")
    p.add_argument("--model", type=str, default=None,
                   help="Base model weights (e.g. yolov8n.pt) or checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="Device: 0, 0,1, cpu (default: auto)")
    p.add_argument("--project", type=str, default=None,
                   help="Output project directory (default: runs/train)")
    p.add_argument("--name", type=str, default=None,
                   help="Run name subdirectory")
    p.add_argument("--hyp", type=Path, default=None,
                   help="Hyperparameter override YAML (e.g. training/configs/yolov8n_poc.yaml)")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last.pt in the run directory")
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

    kwargs = build_train_kwargs(cfg, args)

    if kwargs.get("data") is None:
        logger.error("No data.yaml specified. Use --data or set export.output_dir in config.")
        sys.exit(1)

    train(kwargs, resume=args.resume)


if __name__ == "__main__":
    main()
