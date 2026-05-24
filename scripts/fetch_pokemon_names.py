"""
Fetch all Pokemon names from PokeAPI GraphQL and generate pokemon_classes.yaml.

Usage:
    pip install httpx pyyaml
    python scripts/fetch_pokemon_names.py
    python scripts/fetch_pokemon_names.py --out pokemon_classes.yaml
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import httpx
import yaml

GRAPHQL_URL = "https://beta.pokeapi.co/graphql/v1beta"

# PokeAPI language name -> our key
LANGUAGE_MAP = {
    "en": "en",
    "ja": "ja",
    "ja-Hrkt": "ja_hrkt",
    "roomaji": "ja_ro",
    "fr": "fr",
    "de": "de",
    "ko": "ko",
    "zh-Hans": "zh_hans",
    "zh-Hant": "zh_hant",
}

QUERY = """
query AllSpeciesNames {
  pokemon_v2_pokemonspecies(order_by: {id: asc}) {
    id
    pokemon_v2_pokemonspeciesnames {
      name
      pokemon_v2_language {
        name
      }
    }
  }
}
"""


def fetch_all_species() -> list[dict]:
    print("Fetching from PokeAPI GraphQL...", flush=True)
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(GRAPHQL_URL, json={"query": QUERY})
        resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        print("GraphQL errors:", json.dumps(data["errors"], indent=2), file=sys.stderr)
        sys.exit(1)
    return data["data"]["pokemon_v2_pokemonspecies"]


def build_yaml_data(species_list: list[dict]) -> dict:
    pokemon: dict[int, dict[str, str]] = {}
    for species in species_list:
        pid = species["id"]
        names: dict[str, str] = {}
        for entry in species["pokemon_v2_pokemonspeciesnames"]:
            lang_raw = entry["pokemon_v2_language"]["name"]
            key = LANGUAGE_MAP.get(lang_raw)
            if key:
                names[key] = entry["name"]
        pokemon[pid] = names

    ids = sorted(pokemon.keys())
    return {
        "metadata": {
            "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": GRAPHQL_URL,
            "total": len(ids),
            "id_range": [ids[0], ids[-1]],
            "languages": list(LANGUAGE_MAP.values()),
        },
        "pokemon": pokemon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "pokemon_classes.yaml"),
        help="Output YAML path",
    )
    args = parser.parse_args()

    species_list = fetch_all_species()
    print(f"  -> {len(species_list)} species fetched")

    data = build_yaml_data(species_list)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"Saved: {out_path}")
    print(f"  total={data['metadata']['total']}, id_range={data['metadata']['id_range']}")


if __name__ == "__main__":
    main()
