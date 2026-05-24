"""PokeAPI Sprites scraper.

Images are fetched from the PokeAPI/sprites GitHub repository via raw.githubusercontent.com.
All sprites have transparent backgrounds → bbox can be derived from the alpha channel.

Sprite type URL patterns (replace {id} with national Pokedex number):
  front_default     sprites/pokemon/{id}.png
  back_default      sprites/pokemon/back/{id}.png
  front_shiny       sprites/pokemon/shiny/{id}.png
  back_shiny        sprites/pokemon/back/shiny/{id}.png
  official_artwork  sprites/pokemon/other/official-artwork/{id}.png
  home              sprites/pokemon/other/home/{id}.png
"""

from __future__ import annotations

from pathlib import Path

from data_collection.scrapers.base_scraper import BaseScraper

_BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

SPRITE_URL_PATTERNS: dict[str, str] = {
    "front_default": f"{_BASE}/{{id}}.png",
    "back_default": f"{_BASE}/back/{{id}}.png",
    "front_shiny": f"{_BASE}/shiny/{{id}}.png",
    "back_shiny": f"{_BASE}/back/shiny/{{id}}.png",
    "official_artwork": f"{_BASE}/other/official-artwork/{{id}}.png",
    "home": f"{_BASE}/other/home/{{id}}.png",
}


class PokeAPISpriteScraper(BaseScraper):
    def __init__(
        self,
        output_dir: Path,
        sprite_types: list[str] | None = None,
        *,
        requests_per_second: float = 5.0,
        concurrency: int = 8,
    ) -> None:
        super().__init__(output_dir, requests_per_second=requests_per_second, concurrency=concurrency)
        unknown = set(sprite_types or []) - set(SPRITE_URL_PATTERNS)
        if unknown:
            raise ValueError(f"Unknown sprite types: {unknown}. Valid: {list(SPRITE_URL_PATTERNS)}")
        self._sprite_types = sprite_types or list(SPRITE_URL_PATTERNS)

    def get_targets(self, pokemon_id: int) -> list[tuple[str, str]]:
        return [
            (stype, SPRITE_URL_PATTERNS[stype].format(id=pokemon_id))
            for stype in self._sprite_types
        ]

    def output_path(self, pokemon_id: int, sprite_type: str) -> Path:
        return self.output_dir / "pokeapi" / f"{pokemon_id:04d}" / f"{sprite_type}.png"
