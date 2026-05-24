# Annotation Pipeline Design

## 目的

収集した画像に対してbboxアノテーションを自動付与する。
手動作業をゼロに近づけることを最優先とする。

---

## 内部データ型

すべてのステージが共通の型を出力する。

```python
BBoxAnnotation:
    pokemon_id: int     # = national Pokedex number (= class_id)
    x1, y1, x2, y2: int  # 絶対ピクセル座標
    confidence: float   # 自動付与の信頼度 (alpha=1.0, SAM=0.7〜0.9 など)
    source: str         # どのステージが生成したか

ImageAnnotation:
    image_path: Path
    width, height: int
    bboxes: list[BBoxAnnotation]
    stage: str          # "alpha" | "composite" | "sam" | "feature_match"
```

内部保存形式は `annotations.jsonl`（1行1レコード）。
→ export モジュールが YOLO / COCO に変換する。

---

## ステージ設計

### Stage 1: alpha_bbox（最優先）

| 項目 | 内容 |
|---|---|
| 対象 | `raw_images/pokeapi_sprites/{id}/*.png`（透過PNG） |
| 手法 | αチャンネルの非ゼロ領域の外接矩形 |
| bbox精度 | 完璧（confidence=1.0） |
| pokemon_id | ディレクトリ名の4桁数字から取得 |
| 速度 | 非常に高速（numpy演算のみ） |

```
alpha > 0 な画素の行・列の min/max → x1,y1,x2,y2
```

### Stage 2: composite_gen（合成画像生成）

| 項目 | 内容 |
|---|---|
| 対象 | alpha_bbox 済みスプライト → 新規合成画像を生成 |
| 手法 | スプライトをランダムスケール・位置で背景に貼付 |
| bbox精度 | 完璧（貼付座標から直接生成、confidence=1.0） |
| 背景 | ユーザー提供画像 or 自動生成（ソリッド・グラデーション・ノイズ） |

**パラメータ:**
- `num_composites`: スプライト1枚あたりの合成枚数（デフォルト5）
- `min/max_scale`: スプライトの背景サイズ比（0.05〜0.5）
- `min/max_pokemon`: 1画像あたりのポケモン数（1〜4）
- `output_size`: 合成画像サイズ（デフォルト640×640）

**背景戦略（優先順）:**
1. `backgrounds/` ディレクトリ内の画像を使用
2. SolidColor / LinearGradient / Perlin風ノイズを自動生成

### Stage 3: sam_segment（オプション）

| 項目 | 内容 |
|---|---|
| 対象 | 透過でない実環境・ゲーム画面画像 |
| 手法 | SAM2 自動マスク生成 → bbox変換 |
| bbox精度 | 高（confidence≈0.7〜0.9） |
| 依存 | `pip install sam2`（オプション依存） |

マスク候補から対象ポケモンのサイズ・アスペクト比でフィルタ。

### Stage 4: feature_match（計画中）

| 項目 | 内容 |
|---|---|
| 対象 | スプライトが写り込む複雑な画像 |
| 手法 | ORB/SIFT/LoFTR でテンプレート照合 → ホモグラフィからbbox |
| bbox精度 | 中〜高（confidence≈0.5〜0.8） |

---

## パイプラインフロー

```
raw_images/pokeapi_sprites/
        │
        ▼
[alpha_bbox]  ─────────────────────────→ ImageAnnotation (confidence=1.0)
        │
        ▼（スプライトをソースとして）
[composite_gen] ──────────────────────→ ImageAnnotation (confidence=1.0)
        │                                     + 合成画像を datasets/images/ に保存
        ▼
[sam_segment] (オプション)  ──────────→ ImageAnnotation (confidence≈0.8)
        │
        ▼
[annotations.jsonl]  ← 全ステージの出力を統合
        │
        ├── [export/to_yolo.py]  → datasets/{name}/images/ + labels/ + data.yaml
        └── [export/to_coco.py]  → datasets/{name}/annotations.json
```

---

## 出力ディレクトリ構造

```
datasets/
└── pokemon_detection/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/          # YOLO形式
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── annotations.json # COCO形式（オプション）
    └── data.yaml        # Ultralytics用設定
```

---

## CLIインターフェース

```bash
# Step1: スプライトにalpha_bboxを付与し、合成画像も生成
python annotation/pipeline.py \
    --raw-dir raw_images/pokeapi_sprites \
    --output datasets/raw_annotated \
    --composite --num-composites 10

# Step2: YOLOデータセットにエクスポート
python annotation/export/to_yolo.py \
    --annotations datasets/raw_annotated/annotations.jsonl \
    --output datasets/pokemon_detection \
    --split 0.8 0.1 0.1

# 目視確認
python annotation/review/visualize.py \
    --annotations datasets/raw_annotated/annotations.jsonl \
    --n 20
```

---

## 各ステージのconfidence設計

| ステージ | confidence | 学習への影響 |
|---|---|---|
| alpha_bbox | 1.0 | すべて使用 |
| composite_gen | 1.0 | すべて使用 |
| sam_segment | 0.7〜0.9 | confidence < 0.5 は除外 |
| feature_match | 0.5〜0.8 | 手動レビュー推奨 |
| 手動アノテーション | 1.0 | すべて使用 |
