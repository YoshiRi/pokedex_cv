"""Annotation pipeline CLI.

Accepts either explicit CLI flags or a cross-cutting experiment config.

Usage:
    # Via experiment config (recommended)
    python annotation/pipeline.py --config configs/poc_20species.yaml --composite

    # Via explicit flags
    python annotation/pipeline.py \\
        --raw-dir raw_images/pokeapi_sprites \\
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

from annotation.stages.alpha_bbox import AlphaBBoxStage
from annotation.stages.composite_gen import CompositeConfig, CompositeGenStage
from annotation.schema import AnnotationStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_experiment_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_args_from_config(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset args from experiment config if --config is given."""
    if args.config is None:
        return args
    cfg = load_experiment_config(Path(args.config))
    ann = cfg.get("annotation", {})
    col = cfg.get("collection", {})

    if args.raw_dir is None:
        raw_base = Path(col.get("output_dir", "raw_images"))
        args.raw_dir = str(raw_base / "pokeapi_sprites")
    if args.output is None:
        args.output = ann.get("output_dir", "datasets/raw_annotated")
    if not args.composite:
        args.composite = True  # default on when using experiment config
    args.num_composites = args.num_composites or ann.get("num_composites", 5)
    args.output_size = args.output_size or ann.get("output_size", 640)
    args.min_scale = args.min_scale or ann.get("min_scale", 0.05)
    args.max_scale = args.max_scale or ann.get("max_scale", 0.50)
    args.min_pokemon = args.min_pokemon or ann.get("min_pokemon", 1)
    args.max_pokemon = args.max_pokemon or ann.get("max_pokemon", 3)
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    args = resolve_args_from_config(args)

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
    # Stage 2: composite_gen
    # ------------------------------------------------------------------
    if args.composite:
        logger.info("=== Stage 2: composite_gen ===")
        composites_dir = output_dir / "composites"
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

    all_recs = store.load_all()
    by_stage = {}
    for r in all_recs:
        by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
    logger.info("Pipeline done. Total=%d %s → %s", len(all_recs), by_stage, store_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotation pipeline")
    p.add_argument("--config", default=None,
                   help="Experiment config (e.g. configs/poc_20species.yaml)")
    p.add_argument("--raw-dir", default=None,
                   help="Root dir of collected sprites")
    p.add_argument("--output", default=None,
                   help="Output dir for annotations.jsonl and composites")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-bbox-area", type=int, default=100)
    p.add_argument("--composite", action="store_true")
    p.add_argument("--num-composites", type=int, default=None)
    p.add_argument("--output-size", type=int, default=None)
    p.add_argument("--min-scale", type=float, default=None)
    p.add_argument("--max-scale", type=float, default=None)
    p.add_argument("--min-pokemon", type=int, default=None)
    p.add_argument("--max-pokemon", type=int, default=None)
    p.add_argument("--backgrounds", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
