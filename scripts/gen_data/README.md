# 何をするか
データの拡張

# 各ファイルの特徴

## gen_result.py
1.data/vocab.jsonlによってトークンを
デコーディング  
2.problems.jsonlから問題データを取得
3.generations.jsonl(モデルによる推論結果)をもとに問題とその出力結果をcsvにまとめる

## gen_problems.py
`train.csv` のプロンプトを解析してカテゴリ・例示・質問を抽出し、`data/problem/<id>.json` と `data/problems.jsonl` を生成する。

1. プロンプトの先頭行のキーワード（"bit manipulation" / "encryption" / "numeral system" 等）からカテゴリを判定する
2. カテゴリごとのパーサーで例示ペア（input_value, output_value）と質問を抽出する
3. `data/problem/<id>.json` にカテゴリ・プロンプト・回答・例示・質問をまとめて書き出す
4. 推論ジェネレーター（`scripts/cot_prompt/` 以下）を実行し、答えが一致すれば `status=rule_found`、調査ファイルがあれば `status=hypothesis_formed`、それ以外は `status=rule_unknown` とする
5. 全問題のメタデータ（id / category / status / submission）を `data/problems.jsonl` に書き出す

## gen_reasoning.py
`problems.jsonl` の各問題に対して決定的な推論テキストを生成し、`reasoning/<id>.txt` として保存する。

1. `problems.jsonl` から問題リストを読み込む
2. 各問題に対応するジェネレーター関数（`scripts/cot_prompt/` 以下）を実行して推論テキストを生成する
3. 推論テキスト末尾の `\boxed{}` から答えを抽出し、正解と照合して `status` を更新する（`rule_found` / `rule_unknown`）
4. 調査ファイル（`investigations/<id>.txt`）が存在する問題は `status=hypothesis_formed` に更新する
5. 更新した status を `problems.jsonl` に書き戻し、カテゴリ別の精度統計を表示する

## gen_corpus.py
`reasoning/*.txt` の推論テキストをもとに、ファインチューニング用の合成学習コーパスを生成する。

1. トークナイザー（vocab ファイルおよびチャットテンプレート用）を読み込む
2. `train.csv` から問題プロンプトと回答を、`problems.jsonl` からカテゴリを取得する
3. 推論ファイルが存在する問題ごとに:
   - `reasoning/<id>.txt` を読み込み、末尾の `\boxed{}` から答えを抽出して completion テキストを組み立てる
   - プロンプトと completion をそれぞれトークナイズし、トークン上限で切り捨てる
   - マスク（プロンプト部分=0、completion 部分=1）を付与し、マスク種別が切り替わる境界でセグメントに分割する
   - `corpus/<id>/synthetic.jsonl` にセグメントを書き出す
4. `augmentations/*.txt`（`[category]`/`[prompt]`/`[completion]` 形式の拡張データ）が存在すれば同様に処理する
5. 全エントリーのメタデータを `corpus.jsonl` にインデックスとして書き出し、カテゴリ別の統計を表示する

注意点:

`gen_corpus.py` は全 9,500 問題を無条件に学習データ化するのではなく、
`data/reasoning/<id>.txt` が存在する問題だけを対象にする。そのため、
`gen_reasoning.py` で推論テキストを生成できなかった `rule_unknown` 問題は
`corpus.jsonl` には入らず、件数が 9,500 より少なくなることがある。

例えば、`data/problems.jsonl` が 9,500 件でも `data/reasoning/*.txt` が
9,467 件なら、`gen_corpus.py` の出力も基本的に 9,467 entries になる。
これは token length による追加フィルタというより、reasoning file の有無による
入力データの絞り込みである。

一方、推論用には別に `data/corpus_infer/` と `data/corpus_infer.jsonl` を作る。
こちらは completion/reasoning を含まず、`train.csv` の prompt をチャットテンプレートで
tokenize した prompt token だけを保存する。`reasoning/*.txt` の有無に依存しないため、
通常は全 9,500 問題が入る。`scripts/infer/infer.py` は学習用 `corpus/` ではなく、
この `corpus_infer/` を読んで推論する。
