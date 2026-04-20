# numeric_equation パターン

「Alice's Wonderland」問題における数値演算変換パターンの抽出・バリデーション実装。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `constants.py` | 演算モード・演算種別の定数定義 |
| `label.py` | フローラベルの日本語文字列生成 |
| `parser.py` | プロンプトから方程式ペアを解析 |
| `matching.py` | 演算子グループへのモードマッチング（最良解探索） |
| `predict.py` | 例示のみからの別解予測 |
| `validator.py` | PatternValidator サブクラスと出力生成 |
| `report.py` | PatternExtractor サブクラス（解析レポート生成） |

---

## 問題の構造

各問題は複数の記号演算を含む。例:

```
12+34 = 46
56+78 = 134
12*34 = 408
Now, determine the result for: 56*78
```

- 各行の左辺は「数字 + 演算子記号 + 数字」の形式
- 同じ演算子記号の行をひとつの「演算子グループ」として扱う
- **演算子記号ごとに独立してルールを決定する**（記号が変われば別のルールを持てる）

---

## ルールの構造（モード）

各演算子グループに対するルールは以下の3要素で構成される:

```
(rev_in: bool, swap: bool, out_mode: str)
```

| 要素 | 説明 |
|---|---|
| `rev_in` | 入力の桁を逆順にしてから演算する（例: 16 → 61） |
| `swap` | 左右オペランドを入れ替えてから演算する |
| `out_mode` | 出力変換方式（下表参照） |

### out_mode の種類

| 値 | 説明 | 例 |
|---|---|---|
| `none` | そのまま文字列化 | 45 → "45" |
| `num_rev` | 数値反転（桁のみ逆順・符号はそのまま） | 45 → "54", -45 → "-54" |
| `num_rev_sfx` | 数値反転＋末尾に演算子記号 | 45 → "54+" |
| `full_rev` | 全反転（符号含む文字列ごと逆順） | -45 → "54-" |

---

## マッチング探索順序（`matching.py` / `_best_mode_for_group`）

### フェーズ1: offset=0 の通常演算

全12モード × 全演算（add/sub/mul/abs_diff/neg_abs_diff/mod/concat/concat_strip）を試す。
**offset=0 のみ許可**（最もシンプルなルールを優先）。

モードの探索順: `(rev_in=False, swap=False)` → `(True, False)` → `(False, True)` → `(True, True)` の順に、`out_mode=none → num_rev → full_rev` を試す。

### フェーズ1.5: 特殊出力パターン

通常の算術では表現できない出力形式を試す:

| パターン名 | 出力形式 | 例 |
|---|---|---|
| `abs_diff_op_sign` | `op + str(\|a-b\|)` | 22(35=13( → "(" + "13" |
| `abs_num_rev_op` | `op + str(abs(result))[::-1]` | 16-61 → "-" + "54" |
| `abs_rev_op_suffix` | `str(abs(result))[::-1] + op` | 16-61 → "54" + "-" |

### フェーズ2: offset ±1 の通常演算

フェーズ1と同じ順で、`offset = -1, 0, +1` を許容して再試行。

> **オフセット制約**: offset は必ず -1, 0, +1 のいずれか。最初の例示から推定し、残りの例示で検証する。

---

## 別解の検出（`predict.py` / `_all_modes_for_group`）

`answer` を除いた例示のみで成立するルールが複数ある場合、それぞれのルールがターゲット入力に対して異なる予測を出すかを検証する。

1. 例示のみ（最終行を除く）でターゲット演算子グループの全マッチルールを収集
2. 各ルールでターゲット入力を予測（`_predict`）
3. 予測値が1種類に収束しない場合のみ「別解あり」として記録

結果は `alternatives.csv` に出力される（1行=1問題）。

### alternatives.csv の列

| 列 | 説明 |
|---|---|
| `id` | 問題ID |
| `target_operator` | ターゲット入力の演算子記号 |
| `answer_flow` | answer込みで確定したフロー |
| `answer` | 実際の答え |
| `alt_flows` | 例示のみから導かれる代案フローのリスト（JSON配列） |
| `alt_predictions` | 各代案の予測値リスト（JSON配列） |
| `matches_answer` | 各予測が実際の答えと一致するかのboolリスト（JSON配列） |

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `unmatched.csv` | マッチ失敗行（`failed_operators` 列にマッチできなかった記号） |
| `unmatched_summary.json` | マッチ率のサマリー |
| `matched.jsonl` | マッチ成功行の演算子フロー詳細 |
| `alternatives.csv` | 別解が存在する問題一覧 |
