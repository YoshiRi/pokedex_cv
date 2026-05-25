"""Annotation pipeline CLI.

Orchestrates alpha_bbox → composite_gen stages and saves results to
annotations.jsonl for downstream export.

Usage:
    # Annotate collected sprites + generate composites
    python annotation/pipeline.py \\
        --raw-dir raw_images/pokeapi_sprites \\
        --output datasets/raw_annotated \\
        --composite \\
        --num-composites 10 \\
        --backgrounds backgrounds/

    # Alpha annotation only (no composite)
    python annotation/pipeline.py \\
        --raw-dir raw_images/pokeapi_sprites \\
        --output datasets/raw_annotated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "annotations.jsonl"

    # Overwrite mode: remove existing store if requested
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
        # Load alpha annotations as sprite source
        alpha_anns = [a for a in store.load_all() if a.stage == "alpha"]
        logger.info("Using %d alpha-annotated sprites as composite source", len(alpha_anns))

        composite_stage = CompositeGenStage(cfg)
        n_composite = composite_stage.run(alpha_anns, store)
        logger.info("composite_gen: %d images generated", n_composite)

    total = len(store.load_all())
    logger.info("Pipeline done. Total annotations in store: %d → %s", total, store_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotation pipeline")
    p.add_argument("--raw-dir", required=True,
                   help="Root directory of collected sprites (e.g. raw_images/pokeapi_sprites)")
    p.add_argument("--output", default="datasets/raw_annotated",
                   help="Output directory for annotations.jsonl and composites")
    p.add_argument("--overwrite", action="store_true",
                   help="Remove existing annotations.jsonl before running")

    # Alpha stage
    p.add_argument("--min-bbox-area", type=int, default=100,
                   help="Minimum bbox area in pixels (alpha stage)")

    # Composite stage
    p.add_argument("--composite", action="store_true",
                   help="Run composite_gen stage after alpha_bbox")
    p.add_argument("--num-composites", type=int, default=5,
                   help="Composites to generate per sprite")
    p.add_argument("--output-size", type=int, default=640,
                   help="Composite image size (square, pixels)")
    p.add_argument("--min-scale", type=float, default=0.05)
    p.add_argument("--max-scale", type=float, default=0.50)
    p.add_argument("--min-pokemon", type=int, default=1)
    p.add_argument("--max-pokemon", type=int, default=3)
    p.add_argument("--backgrounds", default=None,
                   help="Directory of background images")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
