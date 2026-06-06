# Repository Design: Pokemon Detection with Bounding Box

## 概要

画像を入力としてポケモンをbboxで検知するモデルを構築する。
データ収集・アノテーション・学習セット作成・学習の各フェーズを独立したモジュールとして管理する。

## ゴール

- 入力画像からポケモンの種類とbboxを出力する物体検知モデルの学習
- データ収集から学習まで再現性のあるパイプラインを構築する
- 複数のモデルアーキテクチャを比較実験できる構造にする

---

## リポジトリ構造

```
pokedex_cv/
│
├── data_collection/              # フェーズ1: オンライン画像収集 ✅
│   ├── scrapers/
│   │   ├── base_scraper.py       # レート制限・リトライ共通処理
│   │   └── pokeapi_sprites.py    # PokeAPI/sprites GitHub raw
│   ├── filters/
│   │   ├── dedup.py              # perceptual hash で重複除去（同一種は比較スキップ）
│   │   └── quality_check.py     # 解像度・ブラー検出
│   └── config.yaml               # 収集対象種・ソース・保存先設定
│
├── annotation/                   # フェーズ2: 自動アノテーションパイプライン ✅
│   ├── pipeline.py               # ステージ群を束ねる全体フロー（--config 対応）
│   ├── schema.py                 # BBoxAnnotation / ImageAnnotation / AnnotationStore
│   ├── stages/
│   │   ├── alpha_bbox.py         # PNG透過→非透過領域からbbox直接生成
│   │   ├── composite_gen.py      # スプライト+多様な背景の合成画像生成
│   │   ├── sam_segment.py        # SAM2でセグメンテーション→bbox変換（オプション）
│   │   └── feature_match.py      # ORB/SIFT/LoFTRでスプライトをシーンに照合（計画中）
│   ├── review/
│   │   └── visualize.py          # 自動生成アノテーションの目視確認ツール
│   └── export/
│       ├── to_yolo.py            # YOLO形式（class_map によるID変換対応）
│       └── to_coco.py            # COCO形式
│
├── dataset/                      # フェーズ3: データセット検証 ✅
│   └── validate.py               # bbox越境・ラベル欠損・class_id整合性チェック
│   # 注: train/val分割・データ拡張は Ultralytics が内蔵しているため独自実装不要
│
├── training/                     # フェーズ4: 学習 ✅
│   ├── configs/
│   │   └── yolov8n_poc.yaml      # YOLOv8n ハイパーパラメータオーバーライド
│   ├── train.py                  # Ultralytics YOLO.train() ラッパー
│   ├── evaluate.py               # mAP・クラス別AP / 実画像推論モード
│   └── export.py                 # ONNX / TorchScript エクスポート
│
├── configs/                      # 実験設定（再現性の単位） ✅
│   └── poc_20species.yaml        # PoC: 20種、class_map、各フェーズのパラメータ
│
├── scripts/                      # ユーティリティスクリプト ✅
│   ├── fetch_pokemon_names.py    # pokemon_classes.yaml 生成
│   └── run_pipeline.py           # 実験 config から全工程を一括実行
│
├── docs/                         # 設計ドキュメント
│
├── data/sprites/                 # PoC スプライト（git 管理）
│   └── pokeapi_sprites/{id:04d}/{sprite_type}.png
├── datasets/                     # 変換済みデータセット (.gitignore)
├── runs/                         # 学習ログ・weights (.gitignore)
│
├── pokemon_classes.yaml          # 全ポケモン名・クラスID定義（1025種）
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 各フェーズの設計方針

### フェーズ1: データ収集 (`data_collection/`)

- スクレイパーは `BaseScraper` を継承する形で実装し、ソースごとに分離する
- レート制限・リトライは基底クラスで一元管理
- 収集後に重複除去（perceptual hash）と品質チェック（解像度・ブラー）を実施する
- **dedup の設計**: 同一ポケモンのスプライト（通常と色違い）は phash で比較しない。
  色違いは構造が同じで色だけ異なるため phash が一致してしまうため。

**実装済みソース:**
- PokeAPI/sprites (GitHub raw) — 公式スプライト・公式アートワーク・HOME スプライト

**計画中ソース:**
- Bulbapedia — 高解像度アートワーク・TCGカード画像

### フェーズ2: 自動アノテーション (`annotation/`)

アノテーションは画像種別に応じて以下のステージを段階的に適用する。

| ステージ | 対象画像 | 手法 | bbox精度 | 状態 |
|---|---|---|---|---|
| `alpha_bbox.py` | 公式スプライトPNG（透過背景） | αチャンネルのnon-zero領域から直接生成 | 完璧・即座 | ✅ |
| `composite_gen.py` | 合成画像（スプライト+背景貼付） | 貼付座標から直接生成 | 完璧 | ✅ |
| `sam_segment.py` | ゲーム画面・実写 | SAM2でセグメンテーション→bbox変換 | 高精度 | オプション |
| `feature_match.py` | スプライトが写り込む複雑な画像 | ORB/SIFT/LoFTRでテンプレート照合 | 中〜高精度 | 計画中 |

**アノテーションの内部形式**: `annotations.jsonl`（1行1レコード）
`BBoxAnnotation.pokemon_id` は常に国際図鑑番号を格納する。YOLO class_id への変換は
エクスポート時に `class_map` で行う（アノテーション自体は実験設定に依存しない）。

### フェーズ3: データセット検証 (`dataset/`)

**train/val分割と拡張は Ultralytics が担う**ため独自実装不要。

`validate.py` のみ実装する。チェック内容:
- bbox 座標が [0, 1] 範囲に収まっているか（YOLO 正規化）
- class_id が `data.yaml` の nc 範囲内か
- ラベルファイルが存在しない画像の検出
- bbox 面積が極小（幅/高さ < 1px 相当）でないか

### フェーズ4: 学習 (`training/`)

- Ultralytics API を共通インターフェースとして利用（YOLOv8 / RT-DETR を同じAPIで扱える）
- `train.py --config configs/{experiment}.yaml` で実験設定を切り替える
- `evaluate.py` には2モード:
  - `val`: 合成データの mAP@50 / mAP@50:95（自動評価）
  - `real`: 実画像への推論と検出数集計（sim-to-real ギャップ確認）
- `export.py` で ONNX / TorchScript エクスポート

### 実験管理 (`configs/`)

実験の再現性単位として `configs/{name}.yaml` を使う。

1ファイルに以下を統合:
- `class_map`: 図鑑番号→YOLO class_id の変換テーブル
- `collection`: 収集対象・ソース設定
- `annotation`: composite_gen のパラメータ
- `export`: データセット出力先・split 比率
- `training`: モデル・エポック数・batch サイズ等

### E2E ランナー (`scripts/run_pipeline.py`)

実験 config を渡すと collect → annotate → export → validate を一括実行する。
学習は GPU 環境（ローカル / Colab）で別途実行する想定。

---

## クラスID設計

```
国際図鑑番号 (1-1025) = pokemon_id    ← アノテーション内部表現
              ↓ export 時に class_map で変換
YOLO class_id (0-indexed contiguous)  ← ラベルファイル内
```

PoC (20種) の class_map:
- index 0 = Pokedex #1 (Bulbasaur)
- index 1 = Pokedex #4 (Charmander)
- ...
- index 19 = Pokedex #1025 (Pecharunt)

フル (1025種) では class_map = [1, 2, 3, ..., 1025]、class_id = pokedex_id - 1。

---

## gitignore 方針

| パス | 管理 | 理由 |
|---|---|---|
| `data/sprites/pokeapi_sprites/` | git 追跡 | 小サイズ（~5MB）、再現性のため |
| `datasets/` | .gitignore | 大容量・生成物 |
| `runs/` | .gitignore | 学習ログ・weights |
| `*.pt` / `*.onnx` | .gitignore | 大容量 |

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| `ultralytics` | YOLOv8/RT-DETR の学習・推論・エクスポート |
| `httpx` | 非同期スプライト収集 |
| `Pillow` | 画像読み書き・合成 |
| `numpy` | alpha bbox 抽出 |
| `imagehash` | perceptual hash による重複除去 |
| `opencv-python` | 特徴点マッチング・Laplacian blur 検出 |
| `albumentations` | （Ultralytics 組み込み拡張で代替、将来的な独自拡張用） |
| `pyyaml` | 設定ファイル読み書き |
| `segment-anything-2` | SAM2（オプション依存） |
