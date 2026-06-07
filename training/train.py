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
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dataset_fingerprint(data_yaml_path: Path) -> str:
    """Short content hash of data.yaml — changes whenever the dataset is regenerated.

    Cheap stand-in for proper dataset versioning: lets a run manifest answer
    "was this the same dataset export as that other run?" without re-reading
    every image/label file.
    """
    return hashlib.sha256(data_yaml_path.read_bytes()).hexdigest()[:12]


def _write_run_manifest(
    run_dir: Path,
    *,
    config_path: Path | None,
    data_path: Path,
    data_cfg: dict,
    kwargs: dict,
) -> None:
    """Record what produced this run's weights: config, dataset, hyperparameters.

    Written into the same directory Ultralytics saves weights/ to, so
    `runs/train/<name>/` is self-describing — answers "which experiment
    config and which dataset export produced best.pt?" without needing to
    cross-reference shell history.
    """
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_config": str(config_path) if config_path else None,
        "model": kwargs.get("model"),
        "hyp": kwargs.get("cfg"),
        "epochs": kwargs.get("epochs"),
        "imgsz": kwargs.get("imgsz"),
        "batch": kwargs.get("batch"),
        "dataset": {
            "data_yaml": str(data_path),
            "fingerprint": _dataset_fingerprint(data_path),
            "nc": data_cfg.get("nc"),
            "names": data_cfg.get("names"),
        },
    }
    manifest_path = run_dir / "experiment_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, sort_keys=False)
    logger.info("Wrote run manifest → %s", manifest_path)


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

    # Hyperparameter override config YAML: CLI flag takes priority over
    # the experiment config's `training.hyp` (e.g. training/configs/yolov8n_poc.yaml)
    hyp_path = args.hyp or train_cfg.get("hyp")
    if hyp_path:
        kwargs["cfg"] = str(hyp_path)

    # Device
    if args.device is not None:
        kwargs["device"] = args.device

    return kwargs


def train(kwargs: dict, resume: bool = False, config_path: Path | None = None) -> None:
    data_path = Path(kwargs["data"])
    if not data_path.exists():
        logger.error(
            "data.yaml not found: %s — run the export step first "
            "(python scripts/run_pipeline.py --config <config> --steps export).",
            data_path,
        )
        sys.exit(1)
    data_cfg = load_config(data_path)
    logger.info("Dataset: %s (nc=%d)", data_path, data_cfg.get("nc", 0))

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

    _write_run_manifest(
        Path(results.save_dir),
        config_path=config_path,
        data_path=data_path,
        data_cfg=data_cfg,
        kwargs={**kwargs, "model": model_arg},
    )


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
                   help="Device: 0, 0,1, cpu, mps (Apple Silicon GPU) — default: Ultralytics auto-selects")
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

    logger.info(
        "Resolved params: model=%s data=%s epochs=%d imgsz=%d batch=%d hyp=%s",
        kwargs["model"], kwargs["data"], kwargs["epochs"], kwargs["imgsz"], kwargs["batch"],
        kwargs.get("cfg", "none"),
    )
    train(kwargs, resume=args.resume, config_path=args.config)


if __name__ == "__main__":
    main()
