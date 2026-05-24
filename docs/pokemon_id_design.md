# Pokemon ID / Name Mapping Design

## 基本方針

モデルのクラスIDには **ナショナルポケモン図鑑番号（1始まり）** をそのまま使う。

```
class_id == national_pokedex_number
```

### 選択理由

| 選択肢 | pros | cons |
|---|---|---|
| 図鑑番号そのまま (1-indexed) | 人間が直感的に理解できる、公式と一致 | 0番が未使用（背景クラス等で使える） |
| 0-indexed (図鑑番号 - 1) | ML慣習と一致 | 人間が確認するとき常に+1が必要 |

→ **図鑑番号そのまま**を採用。モデルの出力クラス数は `max_id + 1 = 1026`（0番はbackground）。

---

## 対象範囲

- 世代: 第1世代〜第9世代（DLC含む）
- 図鑑番号: **1〜1025**（ピカチュウ = 25、ニャオハ = 906 など）
- **フォーム（メガ進化・リージョンフォーム等）は基本クラスに含めない**
  - フォームは別途 `form_id` で管理する予定（後日設計）

---

## 言語サポート

| キー | 言語 | 例（ピカチュウ） |
|---|---|---|
| `en` | 英語 | Pikachu |
| `ja` | 日本語（漢字かな混じり） | ピカチュウ |
| `ja_hrkt` | 日本語（ひらがな/カタカナ読み） | ピカチュウ |
| `ja_ro` | 日本語ローマ字（ヘボン式） | Pikachu |
| `fr` | フランス語 | Pikachu |
| `de` | ドイツ語 | Pikachu |
| `ko` | 韓国語 | 피카츄 |
| `zh_hans` | 中国語（簡体字） | 皮卡丘 |
| `zh_hant` | 中国語（繁体字） | 皮卡丘 |

データ収集時の検索キーワードとして各言語名を使用する。

---

## マスタデータファイル: `pokemon_classes.yaml`

`scripts/fetch_pokemon_names.py` が PokeAPI GraphQL から取得・生成する。
手動編集はしない（再生成で上書きされる）。

### スキーマ

```yaml
metadata:
  generated_at: "2024-01-01T00:00:00Z"
  source: "https://beta.pokeapi.co/graphql/v1beta"
  total: 1025
  id_range: [1, 1025]
  languages: [en, ja, ja_hrkt, ja_ro, fr, de, ko, zh_hans, zh_hant]

pokemon:
  1:
    en: "Bulbasaur"
    ja: "フシギダネ"
    ja_hrkt: "フシギダネ"
    ja_ro: "Fushigidane"
    fr: "Bulbizarre"
    de: "Bisasam"
    ko: "이상해씨"
    zh_hans: "妙蛙种子"
    zh_hant: "妙蛙種子"
  25:
    en: "Pikachu"
    ja: "ピカチュウ"
    ...
```

---

## Python モジュール: `pokedex_cv/pokemon.py`

YAMLを読み込んで以下を提供する。

```python
from pokedex_cv.pokemon import PokemonID, get_name, get_class_names

# IntEnum として使える
PokemonID.PIKACHU          # => 25
int(PokemonID.PIKACHU)     # => 25

# 言語別名前取得
get_name(25, lang="ja")    # => "ピカチュウ"
get_name(25, lang="en")    # => "Pikachu"

# モデルのクラス名リスト（index=class_id、0はbackground）
get_class_names(lang="en") # => ["background", "Bulbasaur", "Ivysaur", ...]
```

---

## データフロー

```
PokeAPI GraphQL
      │
      ▼
scripts/fetch_pokemon_names.py
      │
      ▼
pokemon_classes.yaml  ←── gitで管理（生成物だが小さいのでcommitする）
      │
      ▼
pokedex_cv/pokemon.py  ←── 実行時にロード
      │
      ├── data_collection/  (検索キーワード生成)
      ├── annotation/       (クラスID付与)
      ├── dataset/          (data.yaml生成)
      └── training/         (モデルのnc=1026)
```
