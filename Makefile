.PHONY: help patterns patterns-split patterns-reports infer-result infer-vllm-result

UV := uv run python3
UV_MODAL_RUN := uv run modal run

UV_MODAL_DEPLOY:=uv run modal deploy
PATTERN_DIR := data/patterns
TRAIN_ARGS ?= --num_epochs 1 --batch_size 64 --learning_rate 0.0002
INFER_RUN ?= 04-28-08-23
INFER_ARGS ?= --run $(INFER_RUN)
VLLM_INFER_MODEL_PATH ?= nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
VLLM_INFER_LORA_PATH ?=
VLLM_INFER_ARGS ?= --run $(INFER_RUN) --model-path $(VLLM_INFER_MODEL_PATH) $(if $(VLLM_INFER_LORA_PATH),--lora-path $(VLLM_INFER_LORA_PATH),)
VLLM_INFER_EXTRA_ARGS ?=

help:
	@echo "Available targets:"
	@echo "  make patterns         # split -> reports+unmatched を一気に実行"
	@echo "  make patterns-split   # data/train.csv を data/patterns に分割"
	@echo "  make patterns-reports # data/patterns から report と unmatched を生成"
	@echo "  make infer-result     # Tinker 推論を実行し、評価結果も保存"
	@echo "  make infer-vllm-result # vLLM 推論を実行し、評価結果も保存"

patterns: patterns-split patterns-reports

patterns-split:
	$(UV) scripts/extraction/split_train_by_pattern.py --output-dir $(PATTERN_DIR)

patterns-reports:
	$(UV) scripts/extraction/generate_pattern_rule_reports.py --input-dir $(PATTERN_DIR) --output-dir $(PATTERN_DIR) --mode all

extend-data-from-problems:
	$(UV) scripts/gen_data/gen_problems.py
	$(UV) scripts/gen_data/gen_reasoning.py
	$(UV) scripts/gen_data/gen_corpus.py
extend-data-from-results:
	$(UV) scripts/gen_data/gen_result.py

train-model:
	$(UV) scripts/train/sft.py $(TRAIN_ARGS)
train-model-modal:
	$(UV_MODAL_DEPLOY) scripts/train/sft.py $(TRAIN_ARGS)

upload-adapter:
	$(UV_MODAL_RUN) --detach scripts/submit/upload_adapter_to_kaggle.py

infer-result:
	$(UV) scripts/infer/infer.py $(INFER_ARGS)

infer-vllm-result:
	$(UV) scripts/infer/vllm/infer.py $(VLLM_INFER_ARGS) $(VLLM_INFER_EXTRA_ARGS)
