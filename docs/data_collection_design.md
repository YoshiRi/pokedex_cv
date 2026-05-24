# Data Collection Design

## 目的

学習用ポケモン画像を自動収集する。
アノテーション（bbox付与）コストを最小化するため、**透過PNG（αチャンネルあり）を最優先**で収集する。

---

## 収集ソース一覧

| 優先度 | ソース | 画像種別 | 透過BG | bbox付与方法 | 実装状況 |
|---|---|---|---|---|---|
| ★★★ | PokeAPI Sprites (GitHub raw) | 公式スプライト・公式アートワーク・HOMEスプライト | ✅ | alpha_bbox（完璧） | 実装済み |
| ★★☆ | Bulbapedia | 高解像度アートワーク・ゲーム内画像 | 一部 | alpha_bbox / SAM | 計画中 |
| ★☆☆ | アニメ・ゲーム画面 | 実環境画像 | ❌ | SAM / feature_match | 計画中 |

---

## ソース別設計

### PokeAPI Sprites（主力ソース）

GitHubリポジトリ `PokeAPI/sprites` のRAWファイルを直接取得する。
API呼び出し不要・安定・レート制限が緩い。

**取得するスプライト種別:**

| sprite_type | URL パターン | 解像度目安 | 備考 |
|---|---|---|---|
| `front_default` | `sprites/pokemon/{id}.png` | 96×96 | 最も基本的なドット絵 |
| `back_default` | `sprites/pokemon/back/{id}.png` | 96×96 | 背面スプライト |
| `front_shiny` | `sprites/pokemon/shiny/{id}.png` | 96×96 | 色違い |
| `back_shiny` | `sprites/pokemon/back/shiny/{id}.png` | 96×96 | 色違い背面 |
| `official_artwork` | `sprites/pokemon/other/official-artwork/{id}.png` | 475×475 | 高解像度公式絵 |
| `home` | `sprites/pokemon/other/home/{id}.png` | 680×680 | Pokemon HOME準拠 |

全 1025 種 × 最大 6 種別 → 最大 **6150 枚**（存在しないスプライトは 404 になる）

**保存ディレクトリ構造:**
```
raw_images/
└── pokeapi_sprites/
    ├── 0001/            # Bulbasaur
    │   ├── front_default.png
    │   ├── official_artwork.png
    │   └── home.png
    └── 0025/            # Pikachu
        ├── front_default.png
        ├── back_default.png
        └── ...
```

### Bulbapedia（補助ソース、計画中）

- `https://bulbapedia.bulbagarden.net` の画像ページをスクレイピング
- 高解像度アートワーク・TCGカード画像・ゲーム内グラフィックを収集
- HTML解析が必要なため、実装は PokeAPI 完了後

---

## 収集フロー

```
[config.yaml] ──→ [collect.py]
                       │
              ┌────────┴────────┐
              ▼                 ▼
    PokeAPISpriteScraper   BulbapediaScraper (計画中)
              │
              ▼
    [raw_images/ に保存]
              │
              ▼
    [filters/dedup.py]        perceptual hash で重複除去
              │
              ▼
    [filters/quality_check.py] 解像度・ブラー・空画像チェック
              │
              ▼
    [収集レポート出力]
```

---

## レート制限方針

| ソース | 制限設定 | 理由 |
|---|---|---|
| GitHub raw | 5 req/s（デフォルト） | Unauthenticated: 60 req/min |
| Bulbapedia | 1 req/s | fansite、負荷をかけない |

`--concurrency` オプションで並列数を調整可能。

---

## フィルタ設計

### dedup.py

- **手法**: perceptual hash（pHash）+ ハミング距離
- **閾値**: distance ≤ 5 で重複とみなす
- **保存**: `.dedup_hashes.json` にハッシュDBをキャッシュ（再実行時スキップ）

### quality_check.py

| チェック項目 | 基準 | 対処 |
|---|---|---|
| 最小解像度 | 32×32 px 以上 | 除外 |
| 完全透過画像 | 非透過ピクセル ≥ 10 | 除外 |
| ブラー検出 | Laplacian 分散 ≥ 50 | 除外（ドット絵は例外判定あり） |

---

## 収集量の見込み

| ソース | 種別 | 枚数見込み |
|---|---|---|
| PokeAPI Sprites | ドット絵 (front/back/shiny×2) | ~4000 |
| PokeAPI Sprites | official_artwork | ~1025 |
| PokeAPI Sprites | home | ~1025 |
| **合計（フィルタ後）** | | **~5500〜6000** |

この量ではクラス数（1025）に対して1クラスあたり約5〜6枚と少ない。
不足分は `annotation/stages/composite_gen.py` による合成画像で補う（別設計書参照）。

---

## CLI 使用例

```bash
# 全ポケモンのスプライトを収集
python data_collection/collect.py

# 特定のポケモンのみ（デバッグ用）
python data_collection/collect.py --ids 1 4 7 25

# 特定のスプライト種別のみ
python data_collection/collect.py --sprite-types front_default official_artwork

# 設定ファイルを指定
python data_collection/collect.py --config data_collection/config.yaml
```
