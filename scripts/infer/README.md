# Tinker inference and evaluation

## infer.py

`data/training/sft/<run>/checkpoints.jsonl` の `sampler_path`、または直接渡した
`tinker://.../sampler_weights/...` を使って Tinker SamplingClient で推論する。
推論後はデフォルトで `evaluator.py` を呼び、正誤判定まで行う。

プロンプトは推論時に再トークナイズせず、`gen_corpus.py` が作る
`data/corpus_infer/<problem_id>/prompt.jsonl` の tokenized prompt を読み込む。
`corpus_infer` は `reasoning/*.txt` の有無に依存せず、全 9,500 問題の prompt を
保持する。生成 token は detokenize して `output_text` と `predicted_answer` を保存する。

```bash
python3 scripts/infer/infer.py \
  --run 04-28-08-23 \
  --limit 20 \
  --max-tokens 4096 \
  --temperature 0 \
  --concurrency 8
```

カテゴリや status を絞る場合:

```bash
python3 scripts/infer/infer.py \
  --run 04-28-08-23 \
  --category equation_numeric_deduce \
  --status rule_unknown \
  --limit 50
```

出力先はデフォルトで `data/inference/<train-run>/<infer-time>/`。
例えば `--run 04-28-08-23` なら `data/inference/04-28-08-23/04-28-12-39-10/`。

- `config.json`
- `generations.jsonl`
- `eval.jsonl`
- `summary.json`

評価を走らせずに推論結果だけ保存したい場合は `--no-eval` を付ける。

## vllm/infer.py

Colab などで vLLM を使う推論コードは `scripts/infer/vllm/` に分けている。
詳細は `scripts/infer/vllm/README.md` を参照。

```bash
python3 scripts/infer/vllm/infer.py \
  --run 04-28-08-23 \
  --model-path /content/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --lora-path /content/nemotron-adapter-ready-to-submit \
  --limit 20 \
  --max-num-seqs 32 \
  --batch-size 128
```

## evaluator.py

保存済みの `generations.jsonl` を採点する。

```bash
python3 scripts/infer/evaluator.py data/inference/<run>/<infer-time>/generations.jsonl
```

採点は `extract_final_answer()` で生成文中の最後の `\boxed{...}` を優先抽出し、
無ければ final answer 表記、最後の数値、最後の非空行の順に fallback する。
抽出した `predicted_answer` と保存済み `answer` を比較し、`correct` を保存する。
