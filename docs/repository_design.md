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
├── data_collection/              # フェーズ1: オンライン画像収集
│   ├── scrapers/
│   │   ├── base_scraper.py       # レート制限・リトライ共通処理
│   │   ├── bulbapedia.py         # 公式スプライト・アートワーク
│   │   └── serebii.py
│   ├── filters/
│   │   ├── dedup.py              # perceptual hash で重複除去
│   │   └── quality_check.py     # 解像度・ブラー検出
│   └── config.yaml               # 収集対象種・ソース・保存先設定
│
├── annotation/                   # フェーズ2: 自動アノテーションパイプライン
│   ├── pipeline.py               # ステージ群を束ねる全体フロー
│   ├── stages/
│   │   ├── alpha_bbox.py         # PNG透過→非透過領域からbbox直接生成
│   │   ├── sam_segment.py        # SAM2でセグメンテーション→bbox変換
│   │   ├── feature_match.py      # ORB/SIFT/LoFTRでスプライトをシーンに照合
│   │   └── composite_gen.py      # スプライト+多様な背景の合成画像生成
│   ├── review/
│   │   └── visualize.py          # 自動生成アノテーションの目視確認ツール
│   └── export/
│       ├── to_yolo.py            # YOLO形式 (.txt + data.yaml)
│       └── to_coco.py            # COCO形式 (annotations.json)
│
├── dataset/                      # フェーズ3: データセット管理
│   ├── split.py                  # train/val/test 分割
│   ├── augment.py                # albumentations によるデータ拡張
│   └── validate.py               # ラベル欠損・bbox越境等の整合性チェック
│
├── training/                     # フェーズ4: 学習
│   ├── configs/
│   │   ├── yolov8n.yaml          # YOLOv8 nano
│   │   ├── yolov11n.yaml         # YOLOv11 nano
│   │   └── rtdetr_r50.yaml       # RT-DETR ResNet50
│   ├── train.py                  # --config で切り替えるエントリポイント
│   ├── evaluate.py               # mAP・混同行列等の評価
│   └── export.py                 # ONNX / TensorRT エクスポート
│
├── docs/                         # 設計ドキュメント
│
├── raw_images/                   # 収集した生画像 (.gitignore)
├── datasets/                     # 変換済みデータセット (.gitignore)
├── runs/                         # 学習ログ・weights (.gitignore)
│
├── pokemon_classes.yaml          # 全ポケモン名クラスID定義
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 各フェーズの設計方針

### フェーズ1: データ収集 (`data_collection/`)

- スクレイパーは `BaseScraper` を継承する形で実装し、ソースごとに分離する
- レート制限・リトライは基底クラスで一元管理
- `config.yaml` で収集対象ポケモン・ソース・保存先を設定する
- 収集後に重複除去（perceptual hash）と品質チェック（解像度・ブラー）を実施する

### フェーズ2: 自動アノテーション (`annotation/`)

アノテーションは画像種別に応じて以下のステージを段階的に適用する。

| ステージ | 対象画像 | 手法 | bbox精度 |
|---|---|---|---|
| `alpha_bbox.py` | 公式スプライトPNG（透過背景） | αチャンネルのnon-zero領域から直接生成 | 完璧・即座 |
| `composite_gen.py` | 合成画像（スプライト+背景貼付） | 貼付座標から直接生成 | 完璧 |
| `sam_segment.py` | ゲーム画面・実写 | SAM2でセグメンテーション→bbox変換 | 高精度 |
| `feature_match.py` | スプライトが写り込む複雑な画像 | ORB/SIFT/LoFTRでテンプレート照合 | 中〜高精度 |

**基本方針**: 学習データの大半はスプライト合成（`composite_gen.py`）で賄い、
SAM・特徴点マッチングは補完・精度向上用として活用する。

`pipeline.py` が各ステージを束ね、画像ソース（スプライト/実写等）に応じて
適切なステージを選択して実行する。

### フェーズ3: データセット管理 (`dataset/`)

- アノテーション済みデータを受け取り、train/val/test に分割する
- albumentations でデータ拡張を行う
- `validate.py` でbbox越境・ラベル欠損等の整合性チェックを実施してから学習に渡す

### フェーズ4: 学習 (`training/`)

- YOLOv8 / YOLOv11 / RT-DETR を `configs/` 以下の設定ファイルで切り替えられる設計にする
- `train.py --config configs/yolov8n.yaml` のように呼び出す
- Ultralytics API を共通インターフェースとして利用する（YOLOとRT-DETRを同じAPIで扱える）
- `evaluate.py` で mAP・クラス別AP・混同行列を出力する
- `export.py` で ONNX / TensorRT 形式にエクスポートする

---

## gitignore 方針

学習データ・生画像・weightsはリポジトリに含めない。

```
raw_images/
datasets/
runs/
*.pt
*.onnx
```

---

## 依存ライブラリ（予定）

| ライブラリ | 用途 |
|---|---|
| `ultralytics` | YOLOv8/v11/RT-DETR の学習・推論 |
| `segment-anything-2` | SAM2 による自動セグメンテーション |
| `opencv-python` | 特徴点マッチング・画像処理 |
| `albumentations` | データ拡張 |
| `imagehash` | perceptual hash による重複除去 |
| `httpx` / `aiohttp` | 非同期スクレイピング |
| `Pillow` | 画像読み書き・合成 |
| `pydantic` | 設定ファイルのバリデーション |
