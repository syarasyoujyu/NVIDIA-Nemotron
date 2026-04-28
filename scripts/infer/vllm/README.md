# vLLM inference

`scripts/infer/vllm/infer.py` は Google Colab などの GPU 環境で vLLM を使って
推論するための entrypoint。Tinker 用の `scripts/infer/infer.py` とは分けている。

プロンプトは `data/corpus_infer/<problem_id>/prompt.jsonl` の tokenized prompt を
読み込み、vLLM へ `prompt_token_ids` として渡す。推論後は共通の
`scripts/infer/evaluator.py` で採点する。

```bash
python3 scripts/infer/vllm/infer.py \
  --run 04-28-08-23 \
  --model-path /content/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --lora-path /content/nemotron-adapter-ready-to-submit \
  --limit 20 \
  --max-tokens 4096 \
  --temperature 0 \
  --max-num-seqs 32 \
  --batch-size 128
```

出力先はデフォルトで `data/inference/vllm/<run>/<infer-time>/`。
