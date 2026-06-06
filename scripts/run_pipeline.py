"""End-to-end pipeline runner.

Reads an experiment config and runs: collect → annotate → export → validate.
Training is separate (requires GPU/ultralytics).

Usage:
    # Full run (incremental: keeps existing annotations)
    python scripts/run_pipeline.py --config configs/poc_20species.yaml

    # Clean run (wipes annotations + export dir before running)
    python scripts/run_pipeline.py --config configs/poc_20species.yaml --clean

    # Partial run
    python scripts/run_pipeline.py --config configs/poc_20species.yaml --steps annotate export validate --clean
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_STEPS = ("collect", "annotate", "export", "validate")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Step 1: collect
# ------------------------------------------------------------------

async def step_collect(cfg: dict) -> None:
    from data_collection.filters.dedup import DedupFilter
    from data_collection.filters.quality_check import QualityConfig, QualityFilter
    from data_collection.scrapers.pokeapi_sprites import PokeAPISpriteScraper

    col_cfg = cfg.get("collection", {})
    output_dir = Path(col_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pokemon_ids: list[int] = cfg.get("pokemon_ids", [])
    if not pokemon_ids:
        classes_path = Path(__file__).parent.parent / "pokemon_classes.yaml"
        data = yaml.safe_load(classes_path.read_text())
        pokemon_ids = sorted(data["pokemon"].keys())

    sprite_types: list[str] | None = col_cfg.get("sprite_types")
    logger.info("=== Step 1: collect  (%d Pokemon) ===", len(pokemon_ids))

    scraper = PokeAPISpriteScraper(
        output_dir,
        sprite_types=sprite_types,
        requests_per_second=col_cfg.get("rate_limit", {}).get("requests_per_second", 5.0),
        concurrency=col_cfg.get("concurrency", 8),
    )
    summary = await scraper.collect(pokemon_ids)
    logger.info("collect done: %s", summary)

    collected = list(output_dir.rglob("*.png"))
    logger.info("Total images on disk: %d", len(collected))

    filter_cfg = cfg.get("filters", {})

    if filter_cfg.get("dedup", {}).get("enabled", True):
        dedup = DedupFilter(
            output_dir,
            phash_threshold=filter_cfg["dedup"].get("phash_threshold", 3),
        )
        collected, _ = dedup.run(collected)

    if filter_cfg.get("quality", {}).get("enabled", True):
        qcfg = filter_cfg.get("quality", {})
        quality = QualityFilter(
            QualityConfig(
                min_width=qcfg.get("min_width", 32),
                min_height=qcfg.get("min_height", 32),
                min_nontransparent_pixels=qcfg.get("min_nontransparent_pixels", 10),
                min_laplacian_variance=qcfg.get("min_laplacian_variance", 50.0),
                small_sprite_threshold=qcfg.get("small_sprite_threshold", 128),
            )
        )
        collected, _ = quality.run(collected)

    logger.info("After filters: %d images", len(collected))


# ------------------------------------------------------------------
# Step 2: annotate
# ------------------------------------------------------------------

def step_annotate(cfg: dict, *, overwrite: bool = False) -> None:
    from annotation.pipeline import _resolve_args, run_pipeline
    import argparse as _ap

    logger.info("=== Step 2: annotate ===")
    dummy = _ap.Namespace(
        config=None,
        raw_dir=None,
        output=None,
        overwrite=overwrite,
        min_bbox_area=100,
        composite=False,
        num_composites=None,
        output_size=None,
        min_scale=None,
        max_scale=None,
        min_pokemon=None,
        max_pokemon=None,
        backgrounds=None,
        seed=None,
    )
    args = _resolve_args(dummy, cfg)
    run_pipeline(args)


# ------------------------------------------------------------------
# Step 3: export
# ------------------------------------------------------------------

def step_export(cfg: dict, *, clean: bool = False) -> Path:
    from annotation.export.to_yolo import export_yolo
    from annotation.schema import AnnotationStore

    logger.info("=== Step 3: export ===")
    ann_cfg = cfg.get("annotation", {})
    export_cfg = cfg.get("export", {})

    annotations_path = Path(ann_cfg.get("output_dir", "datasets/raw_annotated")) / "annotations.jsonl"
    output_dir = Path(export_cfg["output_dir"])
    class_map: list[int] | None = cfg.get("class_map")
    split = tuple(export_cfg.get("split", [0.8, 0.1, 0.1]))
    seed = export_cfg.get("seed", 42)
    min_confidence = export_cfg.get("min_confidence", 0.0)

    store = AnnotationStore(annotations_path)
    annotations = store.load_all()
    logger.info("Loaded %d annotation records from %s", len(annotations), annotations_path)

    export_yolo(
        annotations,
        output_dir,
        split,
        min_confidence=min_confidence,
        class_map=class_map,
        seed=seed,
        clean=clean,
    )
    return output_dir


# ------------------------------------------------------------------
# Step 4: validate
# ------------------------------------------------------------------

def step_validate(dataset_dir: Path) -> bool:
    from dataset.validate import validate_dataset

    logger.info("=== Step 4: validate ===")
    ok = validate_dataset(dataset_dir)
    if ok:
        logger.info("Validation passed.")
    else:
        logger.error("Validation FAILED — fix errors before training.")
    return ok


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end pipeline runner")
    p.add_argument("--config", type=Path, required=True,
                   help="Experiment config YAML (e.g. configs/poc_20species.yaml)")
    p.add_argument(
        "--steps",
        nargs="+",
        default=list(ALL_STEPS),
        choices=list(ALL_STEPS),
        metavar="STEP",
        help=f"Steps to run (default: all). Options: {ALL_STEPS}",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Clean run: overwrite annotations.jsonl (annotate step) "
            "and remove export output dir before writing (export step). "
            "Use this to guarantee a fresh dataset with no leftover state."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    steps = args.steps

    logger.info("Running pipeline: %s  (config=%s, clean=%s)", steps, args.config, args.clean)

    dataset_dir: Path | None = None

    if "collect" in steps:
        asyncio.run(step_collect(cfg))

    if "annotate" in steps:
        step_annotate(cfg, overwrite=args.clean)

    if "export" in steps:
        dataset_dir = step_export(cfg, clean=args.clean)

    if "validate" in steps:
        if dataset_dir is None:
            dataset_dir = Path(cfg["export"]["output_dir"])
        ok = step_validate(dataset_dir)
        if not ok:
            sys.exit(1)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
