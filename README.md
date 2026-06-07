# pokedex_cv

Pokemon object detection pipeline — from online sprite collection to trained YOLO model.

Given an image, the model detects each visible Pokemon and outputs its national Pokedex number and bounding box.

---

## Architecture

```
data_collection/   Async scraping of PokeAPI sprites (httpx + rate limiting)
annotation/        bbox pipeline: alpha channel → composite synthesis → export
dataset/           YOLO dataset validation
training/          YOLOv8 train / evaluate / export
configs/           Experiment configs (one file = full reproducible run)
scripts/           End-to-end pipeline runner
```

---

## Quick start

### Prerequisites

```bash
pip install ultralytics httpx pyyaml pillow numpy imagehash opencv-python-headless
```

### PoC run (20 species, runs in ~2 minutes without GPU)

```bash
# Collect + annotate + export + validate
python scripts/run_pipeline.py --config configs/poc_20species.yaml --clean

# Train (requires GPU + ultralytics)
python training/train.py --config configs/poc_20species.yaml

# Evaluate
python training/evaluate.py --config configs/poc_20species.yaml
python training/evaluate.py --config configs/poc_20species.yaml --mode real --images /path/to/photos/

# Export to ONNX
python training/export.py --config configs/poc_20species.yaml
```

### Full run (1025 species)

```bash
python scripts/run_pipeline.py --config configs/full_1025species.yaml --clean
python training/train.py --config configs/full_1025species.yaml
```

---

## Pipeline steps

### 1. Data collection (`data_collection/`)

Downloads sprites from [PokeAPI/sprites](https://github.com/PokeAPI/sprites) via GitHub raw URLs.
No API key required. Rate-limited async downloads with per-Pokemon dedup.

Sprite types collected: `front_default`, `front_shiny`, `official_artwork`, `home`
(back sprites optional; Gen 9 Pokemon lack back sprites in the source)

### 2. Annotation (`annotation/`)

**Stage 1 — alpha_bbox**: sprites have transparent backgrounds.
The bounding box is derived directly from the non-transparent pixel region (confidence = 1.0).

**Stage 2 — composite_gen**: sprites are pasted at random positions and scales onto
procedural backgrounds (solid, gradient, noise, sky, grass, checkerboard, bokeh).
Bbox is exact from paste coordinates (confidence = 1.0).

Output: `annotations.jsonl` (one `ImageAnnotation` record per line)

### 3. Export (`annotation/export/to_yolo.py`)

Converts `annotations.jsonl` → YOLO format with train/val/test split.

`class_map` (in experiment config) maps national Pokedex IDs → contiguous YOLO class IDs.
Annotations always store the true Pokedex number; remapping happens at export time.

### 4. Validation (`dataset/validate.py`)

Checks YOLO dataset integrity:
- bbox coordinates in [0, 1]
- class_id in [0, nc-1]
- every image has a label file (and vice versa)
- no filename collisions within or across splits

### 5. Training (`training/`)

Wraps Ultralytics `YOLO.train()`. Hyperparameter overrides in `training/configs/yolov8n_poc.yaml`.

Two evaluation modes:
- `val` — mAP@50 / mAP@50:95 on synthetic val split
- `real` — inference on real Pokemon photos (sim-to-real gap check)

---

## Experiment configs (`configs/`)

| Config | Species | Sprites | Composites | Purpose |
|---|---|---|---|---|
| `poc_20species.yaml` | 20 | 116 (in git) | 1740 | Pipeline verification |
| `full_1025species.yaml` | 1025 | ~4000 (gitignored) | ~20000 | Full training |

### Class ID design

```
national Pokedex ID (1–1025)  ← stored in annotations (pokemon_id)
           ↓  class_map at export time
YOLO class_id (0-indexed)     ← written to .txt label files
```

For the full dataset, `class_map: null` auto-generates from `pokemon_classes.yaml`:
`class_id 0 = #1 Bulbasaur`, `class_id 1 = #2 Ivysaur`, ..., `class_id 1024 = #1025 Pecharunt`

---

## Repository layout

```
configs/                      Experiment configs
  poc_20species.yaml
  full_1025species.yaml

data/sprites/pokeapi_sprites/ PoC sprites (20 species, in git)
datasets/                     Generated datasets (.gitignore)
runs/                         Training outputs (.gitignore)

data_collection/
  scrapers/pokeapi_sprites.py
  filters/dedup.py
  filters/quality_check.py

annotation/
  schema.py                   BBoxAnnotation / ImageAnnotation / AnnotationStore
  pipeline.py                 alpha_bbox → composite_gen
  stages/
  export/to_yolo.py
  export/to_coco.py
  review/visualize.py

dataset/
  validate.py

training/
  train.py
  evaluate.py
  export.py
  configs/yolov8n_poc.yaml

scripts/
  run_pipeline.py             collect → annotate → export → validate
  fetch_pokemon_names.py      regenerate pokemon_classes.yaml

pokemon_classes.yaml          1025-species name data (en/ja/ko/zh/fr/de)
```

---

## Data files (git)

| Path | Contents | Size |
|---|---|---|
| `pokemon_classes.yaml` | 1025-species names in 9 languages | ~200 KB |
| `data/sprites/pokeapi_sprites/` | PoC sprites (116 PNG files) | ~5 MB |

Full-dataset sprites and generated datasets are excluded from git (`.gitignore`).
