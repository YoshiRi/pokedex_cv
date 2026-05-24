"""CLI entry point for data collection.

Accepts either the module-local config (data_collection/config.yaml)
or a cross-cutting experiment config (e.g. configs/poc_20species.yaml).

Usage:
    python data_collection/collect.py                             # default config
    python data_collection/collect.py --config configs/poc_20species.yaml
    python data_collection/collect.py --ids 1 4 7 25             # override IDs
    python data_collection/collect.py --sprite-types front_default official_artwork
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.filters.dedup import DedupFilter
from data_collection.filters.quality_check import QualityConfig, QualityFilter
from data_collection.scrapers.pokeapi_sprites import PokeAPISpriteScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_config(cfg: dict) -> dict:
    """Convert any config format to a unified internal dict.

    Supports:
      - data_collection/config.yaml  (has 'sources.pokeapi_sprites' key)
      - configs/poc_*.yaml           (has 'collection' key)
    """
    if "collection" in cfg:
        # Experiment config format (configs/poc_20species.yaml)
        col = cfg["collection"]
        return {
            "output_dir": col.get("output_dir", cfg.get("output_dir", "raw_images")),
            "pokemon_ids": cfg.get("pokemon_ids"),
            "sprite_types": col.get("sprite_types"),
            "requests_per_second": col.get("rate_limit", {}).get("requests_per_second", 5.0),
            "concurrency": col.get("concurrency", 8),
            "filters": cfg.get("filters", {}),
        }
    else:
        # Module config format (data_collection/config.yaml)
        spr = cfg.get("sources", {}).get("pokeapi_sprites", {})
        return {
            "output_dir": cfg.get("output_dir", "raw_images"),
            "pokemon_ids": cfg.get("pokemon_ids"),
            "sprite_types": spr.get("sprite_types"),
            "requests_per_second": spr.get("rate_limit", {}).get("requests_per_second", 5.0),
            "concurrency": spr.get("concurrency", 8),
            "filters": cfg.get("filters", {}),
        }


def resolve_pokemon_ids(cfg_ids: list[int] | None) -> list[int]:
    if cfg_ids:
        return sorted(cfg_ids)
    classes_path = Path(__file__).parent.parent / "pokemon_classes.yaml"
    if classes_path.exists():
        data = yaml.safe_load(classes_path.read_text())
        return sorted(data["pokemon"].keys())
    logger.warning("pokemon_classes.yaml not found. Using id range 1-1025.")
    return list(range(1, 1026))


async def run_collection(args: argparse.Namespace, cfg: dict) -> list[Path]:
    ncfg = normalize_config(cfg)
    output_dir = Path(ncfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pokemon_ids = resolve_pokemon_ids(args.ids or ncfg["pokemon_ids"])
    sprite_types = args.sprite_types or ncfg["sprite_types"]
    logger.info("Collecting %d Pokemon, sprite_types=%s", len(pokemon_ids), sprite_types)

    scraper = PokeAPISpriteScraper(
        output_dir,
        sprite_types=sprite_types,
        requests_per_second=ncfg["requests_per_second"],
        concurrency=ncfg["concurrency"],
    )
    summary = await scraper.collect(pokemon_ids)
    logger.info("Scraper done: %s", summary)

    collected = list(output_dir.rglob("*.png"))
    logger.info("Total PNG files: %d", len(collected))

    filter_cfg = ncfg.get("filters", {})

    if filter_cfg.get("dedup", {}).get("enabled", True):
        dedup = DedupFilter(
            output_dir,
            phash_threshold=filter_cfg.get("dedup", {}).get("phash_threshold", 5),
        )
        collected, removed = dedup.run(collected)
        logger.info("After dedup: kept=%d removed=%d", len(collected), len(removed))

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
        collected, removed = quality.run(collected)
        logger.info("After quality: kept=%d removed=%d", len(collected), len(removed))

    logger.info("Final image count: %d", len(collected))
    return collected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Pokemon images for training.")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--ids", type=int, nargs="+", metavar="ID")
    p.add_argument("--sprite-types", nargs="+", metavar="TYPE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    asyncio.run(run_collection(args, cfg))


if __name__ == "__main__":
    main()
