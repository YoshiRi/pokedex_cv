"""
Pokemon ID / name lookup utilities.

Class IDs equal national Pokedex numbers (1-indexed).
Class 0 is reserved as background.

    PokemonID.PIKACHU          => 25
    get_name(25, lang="ja")    => "ピカチュウ"
    get_class_names(lang="en") => ["background", "Bulbasaur", ...]
"""

from __future__ import annotations

import re
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

_YAML_PATH: Final[Path] = Path(__file__).parent.parent / "pokemon_classes.yaml"
_BACKGROUND_LABEL: Final[str] = "background"


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    if not _YAML_PATH.exists():
        raise FileNotFoundError(
            f"{_YAML_PATH} not found. "
            "Run `python scripts/fetch_pokemon_names.py` to generate it."
        )
    with _YAML_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data() -> dict[int, dict[str, str]]:
    return _load_yaml()["pokemon"]


def get_name(pokemon_id: int, lang: str = "en") -> str:
    """Return the Pokemon name for the given class ID and language."""
    entry = _data().get(pokemon_id)
    if entry is None:
        raise KeyError(f"Unknown pokemon_id: {pokemon_id}")
    name = entry.get(lang)
    if name is None:
        available = list(entry.keys())
        raise KeyError(f"Language '{lang}' not found for id={pokemon_id}. Available: {available}")
    return name


def get_class_names(lang: str = "en") -> list[str]:
    """Return list of class names indexed by class_id.

    Index 0 is 'background'. Index n == national Pokedex number n.
    Suitable for passing directly to model configs (nc = len(result)).
    """
    data = _data()
    max_id = max(data.keys())
    names = [_BACKGROUND_LABEL] * (max_id + 1)
    for pid, entry in data.items():
        names[pid] = entry.get(lang, entry.get("en", str(pid)))
    return names


def _to_enum_key(english_name: str) -> str:
    key = english_name.upper()
    key = re.sub(r"[^A-Z0-9]", "_", key)
    if key[0].isdigit():
        key = "P" + key
    return key


def _build_enum() -> type[IntEnum]:
    data = _data()
    members = {_to_enum_key(entry["en"]): pid for pid, entry in data.items() if "en" in entry}
    return IntEnum("PokemonID", members)  # type: ignore[return-value]


# Build PokemonID lazily on first attribute access to avoid import-time YAML load.
class _LazyEnum:
    _enum: type[IntEnum] | None = None

    def _get(self) -> type[IntEnum]:
        if self._enum is None:
            self._enum = _build_enum()
        return self._enum

    def __getattr__(self, name: str) -> int:
        return getattr(self._get(), name)

    def __call__(self, value: int) -> IntEnum:
        return self._get()(value)

    def __iter__(self):
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())


PokemonID: _LazyEnum = _LazyEnum()
