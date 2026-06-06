"""Annotation pipeline CLI.

Orchestrates alpha_bbox → composite_gen stages and saves results to
annotations.jsonl for downstream export.

Usage:
    # From experiment config (recommended)
    python annotation/pipeline.py --config configs/poc_20species.yaml

    # With overrides
    python annotation/pipeline.py --config configs/poc_20species.yaml --num-composites 5

    # Manual (no config)
    python annotation/pipeline.py \\
        --raw-dir data/sprites/pokeapi_sprites \\
        --output datasets/raw_annotated \\
        --composite --num-composites 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation.schema import AnnotationStore
from annotation.stages.alpha_bbox import AlphaBBoxStage
from annotation.stages.composite_gen import CompositeConfig, CompositeGenStage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_args(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    """Fill in args from experiment config where CLI arg was not supplied."""
    col_cfg = cfg.get("collection", {})
    ann_cfg = cfg.get("annotation", {})

    # raw_dir: {collection.output_dir}/pokeapi_sprites
    if args.raw_dir is None:
        base = col_cfg.get("output_dir")
        if base:
            args.raw_dir = str(Path(base) / "pokeapi_sprites")

    if args.output is None:
        args.output = ann_cfg.get("output_dir", "datasets/raw_annotated")

    if not args.composite and ann_cfg.get("num_composites", 0) > 0:
        args.composite = True

    args.num_composites = args.num_composites or ann_cfg.get("num_composites", 5)
    args.output_size = args.output_size or ann_cfg.get("output_size", 640)
    args.min_scale = args.min_scale or ann_cfg.get("min_scale", 0.05)
    args.max_scale = args.max_scale or ann_cfg.get("max_scale", 0.50)
    args.min_pokemon = args.min_pokemon or ann_cfg.get("min_pokemon", 1)
    args.max_pokemon = args.max_pokemon or ann_cfg.get("max_pokemon", 3)
    args.seed = args.seed if args.seed is not None else ann_cfg.get("seed", 42)

    return args


def run_pipeline(args: argparse.Namespace) -> None:
    if args.raw_dir is None:
        logger.error("--raw-dir is required (or set collection.output_dir in --config)")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "annotations.jsonl"

    if args.overwrite and store_path.exists():
        store_path.unlink()
        logger.info("Removed existing %s (--overwrite)", store_path)

    store = AnnotationStore(store_path)

    # ------------------------------------------------------------------
    # Stage 1: alpha_bbox
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: alpha_bbox ===")
    alpha_stage = AlphaBBoxStage(min_bbox_area=args.min_bbox_area)
    n_alpha = alpha_stage.run(Path(args.raw_dir), store)
    logger.info("alpha_bbox: %d images annotated", n_alpha)

    # ------------------------------------------------------------------
    # Stage 2: composite_gen (optional)
    # ------------------------------------------------------------------
    if args.composite:
        logger.info("=== Stage 2: composite_gen ===")
        composites_dir = Path(args.output) / "composites"
        cfg = CompositeConfig(
            output_dir=composites_dir,
            num_composites=args.num_composites,
            output_size=(args.output_size, args.output_size),
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            min_pokemon=args.min_pokemon,
            max_pokemon=args.max_pokemon,
            backgrounds_dir=Path(args.backgrounds) if args.backgrounds else None,
            seed=args.seed,
        )
        alpha_anns = [a for a in store.load_all() if a.stage == "alpha"]
        logger.info("Using %d alpha-annotated sprites as composite source", len(alpha_anns))

        composite_stage = CompositeGenStage(cfg)
        n_composite = composite_stage.run(alpha_anns, store)
        logger.info("composite_gen: %d images generated", n_composite)

    total = len(store.load_all())
    logger.info("Pipeline done. Total annotations in store: %d → %s", total, store_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotation pipeline")
    p.add_argument("--config", type=Path, default=None,
                   help="Experiment config YAML (e.g. configs/poc_20species.yaml)")
    p.add_argument("--raw-dir", default=None,
                   help="Root directory of collected sprites")
    p.add_argument("--output", default=None,
                   help="Output directory for annotations.jsonl and composites")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing annotations.jsonl before running")

    p.add_argument("--min-bbox-area", type=int, default=100)

    p.add_argument("--composite", action="store_true")
    p.add_argument("--num-composites", type=int, default=None)
    p.add_argument("--output-size", type=int, default=None)
    p.add_argument("--min-scale", type=float, default=None)
    p.add_argument("--max-scale", type=float, default=None)
    p.add_argument("--min-pokemon", type=int, default=None)
    p.add_argument("--max-pokemon", type=int, default=None)
    p.add_argument("--backgrounds", default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config)
    args = _resolve_args(args, cfg)
    run_pipeline(args)


if __name__ == "__main__":
    main()
