"""
Fetch all Pokemon names and generate pokemon_classes.yaml.

Sources (tried in order):
  1. PokeAPI GraphQL  https://beta.pokeapi.co/graphql/v1beta  (single request, fast)
  2. PokeAPI api-data https://raw.githubusercontent.com/PokeAPI/api-data/...  (per-species JSON)

Usage:
    pip install httpx pyyaml
    python scripts/fetch_pokemon_names.py
    python scripts/fetch_pokemon_names.py --out pokemon_classes.yaml --max-id 1025
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

import httpx
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPHQL_URL = "https://beta.pokeapi.co/graphql/v1beta"
API_DATA_BASE = (
    "https://raw.githubusercontent.com/PokeAPI/api-data/master"
    "/data/api/v2/pokemon-species/{id}/index.json"
)
MAX_POKEMON_ID = 1025
CONCURRENCY = 20

# PokeAPI language name -> our YAML key
LANGUAGE_MAP: dict[str, str] = {
    # api-data uses lowercase keys
    "en": "en",
    "ja": "ja",
    "ja-hrkt": "ja_hrkt",
    "ja-roma": "ja_ro",
    "fr": "fr",
    "de": "de",
    "ko": "ko",
    "zh-hans": "zh_hans",
    "zh-hant": "zh_hant",
    # GraphQL uses different casing for the same languages
    "ja-Hrkt": "ja_hrkt",
    "roomaji": "ja_ro",
    "zh-Hans": "zh_hans",
    "zh-Hant": "zh_hant",
}

GRAPHQL_QUERY = """
query AllSpeciesNames {
  pokemon_v2_pokemonspecies(order_by: {id: asc}) {
    id
    pokemon_v2_pokemonspeciesnames {
      name
      pokemon_v2_language { name }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Source 1: GraphQL (single request)
# ---------------------------------------------------------------------------

def _fetch_graphql() -> dict[int, dict[str, str]] | None:
    print("Trying PokeAPI GraphQL...", flush=True)
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(GRAPHQL_URL, json={"query": GRAPHQL_QUERY})
            resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            print(f"  GraphQL errors: {data['errors']}", file=sys.stderr)
            return None
        species_list = data["data"]["pokemon_v2_pokemonspecies"]
        result: dict[int, dict[str, str]] = {}
        for sp in species_list:
            names = {}
            for entry in sp["pokemon_v2_pokemonspeciesnames"]:
                key = LANGUAGE_MAP.get(entry["pokemon_v2_language"]["name"])
                if key:
                    names[key] = entry["name"]
            result[sp["id"]] = names
        print(f"  -> {len(result)} species via GraphQL")
        return result
    except Exception as e:
        print(f"  GraphQL failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Source 2: api-data static JSON (per-species async fetch)
# ---------------------------------------------------------------------------

async def _fetch_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    pid: int,
) -> tuple[int, dict[str, str]] | None:
    url = API_DATA_BASE.format(id=pid)
    async with sem:
        try:
            resp = await client.get(url, timeout=15.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            names: dict[str, str] = {}
            for entry in data.get("names", []):
                lang_raw = entry["language"]["name"]
                key = LANGUAGE_MAP.get(lang_raw)
                if key:
                    names[key] = entry["name"]
            return pid, names
        except Exception as e:
            print(f"  warn: id={pid} {e}", file=sys.stderr)
            return None


async def _fetch_api_data(max_id: int) -> dict[int, dict[str, str]]:
    print(f"Fetching {max_id} species from PokeAPI/api-data (concurrency={CONCURRENCY})...", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        headers={"User-Agent": "pokedex-cv/0.1"},
        follow_redirects=True,
    ) as client:
        tasks = [_fetch_one(client, sem, pid) for pid in range(1, max_id + 1)]
        results = await asyncio.gather(*tasks)

    pokemon: dict[int, dict[str, str]] = {}
    for r in results:
        if r is not None:
            pid, names = r
            pokemon[pid] = names
    print(f"  -> {len(pokemon)} species fetched")
    return pokemon


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

def _build_yaml(pokemon: dict[int, dict[str, str]], source: str) -> dict:
    ids = sorted(pokemon.keys())
    return {
        "metadata": {
            "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "total": len(ids),
            "id_range": [ids[0], ids[-1]],
            "languages": list(dict.fromkeys(LANGUAGE_MAP.values())),
        },
        "pokemon": {pid: pokemon[pid] for pid in ids},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "pokemon_classes.yaml"),
    )
    parser.add_argument("--max-id", type=int, default=MAX_POKEMON_ID)
    parser.add_argument(
        "--source",
        choices=["auto", "graphql", "api-data"],
        default="auto",
        help="Data source (auto tries GraphQL first, falls back to api-data)",
    )
    args = parser.parse_args()

    pokemon: dict[int, dict[str, str]] | None = None
    source_used = ""

    if args.source in ("auto", "graphql"):
        pokemon = _fetch_graphql()
        if pokemon:
            source_used = GRAPHQL_URL

    if pokemon is None and args.source in ("auto", "api-data"):
        pokemon = asyncio.run(_fetch_api_data(args.max_id))
        source_used = API_DATA_BASE

    if not pokemon:
        print("All sources failed.", file=sys.stderr)
        sys.exit(1)

    data = _build_yaml(pokemon, source_used)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"Saved: {out_path}")
    print(f"  total={data['metadata']['total']}, id_range={data['metadata']['id_range']}")


if __name__ == "__main__":
    main()
