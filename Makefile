.PHONY: help raw-data patterns patterns-split patterns-reports extend-data-from-problems extend-data-from-generated train-model train-model-from-generated train-model-from-pretrained train-model-from-pretrained-generated train-model-modal train-model-modal-generated infer-result infer-vllm-result

UV := uv run python3
UV_MODAL_RUN := uv run modal run

UV_MODAL_DEPLOY:=uv run modal deploy
PATTERN_DIR := data/patterns
RAW_PATTERN_DIR := data/generated/patterns
GENERATED_TRAIN_CSV := data/generated/train.csv
GENERATED_TEST_CSV := data/generated/test.csv
RAW_DATA_MODE ?= generate
RAW_DATA_SEED ?= 0
RAW_FAIL_ON_COT_MISMATCH ?=
# category order:
# bit_manipulation,cipher,cryptarithm_deduce,cryptarithm_guess,equation_numeric_deduce,equation_numeric_guess,gravity,numeral,unit_conversion
RAW_TRAIN_COUNT_LIST ?= [1602,1576,329,494,596,136,1597,1576,1594]
RAW_TEST_COUNT_LIST ?= [600,700,800,100,800,0,400,300,700]
RAW_DATA_ARGS ?= --mode $(RAW_DATA_MODE) --seed $(RAW_DATA_SEED) --output-dir $(RAW_PATTERN_DIR) --train-csv-output $(GENERATED_TRAIN_CSV) --test-csv-output $(GENERATED_TEST_CSV) --train-counts "$(RAW_TRAIN_COUNT_LIST)" --test-counts "$(RAW_TEST_COUNT_LIST)" $(if $(RAW_FAIL_ON_COT_MISMATCH),--fail-on-cot-mismatch,)
# task type order:
# gravity,numeral,unit_conversion,cipher,bit_manipulation,equation_transformation
TASK_TYPE_LIMIT_COUNTS ?=
CATEGORY_LIMIT_COUNTS ?= [200,200,800,0,800,0,200,150,150]
TRAIN_COT_PROMPT_FILTER_MODE ?= correct
TRAIN_BATCH_STRATIFY_BY ?= task_type
TRAIN_TASK_TYPE_LIMIT_STRATEGY ?= random
TRAIN_TASK_TYPE_LIMIT_SEED ?= 0
TRAIN_ARGS ?= --num_epochs 4 --batch_size 32 --learning_rate 0.00002 --cot_prompt_filter_mode $(TRAIN_COT_PROMPT_FILTER_MODE) --batch_stratify_by $(TRAIN_BATCH_STRATIFY_BY) --task_type_limit_strategy $(TRAIN_TASK_TYPE_LIMIT_STRATEGY) --task_type_limit_seed $(TRAIN_TASK_TYPE_LIMIT_SEED) $(if $(TASK_TYPE_LIMIT_COUNTS),--task_type_limit_counts "$(TASK_TYPE_LIMIT_COUNTS)",) $(if $(CATEGORY_LIMIT_COUNTS),--category_limit_counts "$(CATEGORY_LIMIT_COUNTS)",)
TRAIN_GENERATED_ARGS ?= $(TRAIN_ARGS)
MODEL_CHECKPOINT ?=tinker://184d24d7-3808-56d6-b77a-c2c8562e29bd:train:0/weights/final
TRAIN_FROM_PRETRAINED_ARGS ?= $(TRAIN_ARGS) --from_pretrained --pretrained_path "$(MODEL_CHECKPOINT)"
TRAIN_FROM_PRETRAINED_GENERATED_ARGS ?= $(TRAIN_GENERATED_ARGS) --from_pretrained --pretrained_path "$(MODEL_CHECKPOINT)"
INFER_RUN ?= 04-28-08-23
INFER_ARGS ?= --run $(INFER_RUN)
VLLM_INFER_MODEL_PATH ?= nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
VLLM_INFER_LORA_PATH ?=
VLLM_INFER_ARGS ?= --run $(INFER_RUN) --model-path $(VLLM_INFER_MODEL_PATH) $(if $(VLLM_INFER_LORA_PATH),--lora-path $(VLLM_INFER_LORA_PATH),)
VLLM_INFER_EXTRA_ARGS ?=

help:
	@echo "Available targets:"
	@echo "  make raw-data         # cot_prompt 互換カテゴリの train/test raw pattern data を生成"
	@echo "                         # 例: make raw-data RAW_TRAIN_COUNT_LIST='[100,100,0,0,50,0,0,0,0]'"
	@echo "  make patterns         # split -> reports+unmatched を一気に実行"
	@echo "  make patterns-split   # data/train.csv を data/patterns に分割"
	@echo "  make patterns-reports # data/patterns から report と unmatched を生成"
	@echo "  make extend-data-from-problems  # data/train.csv から problem/reasoning/corpus を生成"
	@echo "  make extend-data-from-generated # data/generated/train.csv から problem/reasoning/corpus を生成"
	@echo "  make train-model      # Tinker SFT 学習を実行"
	@echo "  make train-model-from-generated # 生成データ由来の corpus で Tinker SFT 学習を実行"
	@echo "  make train-model-from-pretrained MODEL_CHECKPOINT='tinker://.../weights/final'"
	@echo "  make train-model-from-pretrained-generated MODEL_CHECKPOINT='tinker://.../weights/final'"
	@echo "  make infer-result     # Tinker 推論を実行し、評価結果も保存"
	@echo "  make infer-vllm-result # vLLM 推論を実行し、評価結果も保存"

raw-data:
	python3 scripts/gen_data/gen_raw_data.py $(RAW_DATA_ARGS)

gen-raw-data: raw-data

patterns: patterns-split patterns-reports

patterns-split:
	python3 scripts/extraction/split_train_by_pattern.py --output-dir $(PATTERN_DIR) --legacy-split

patterns-reports:
	python3 scripts/extraction/generate_pattern_rule_reports.py --input-dir $(PATTERN_DIR) --output-dir $(PATTERN_DIR) --mode all

extend-data-from-problems:
	$(UV) scripts/gen_data/gen_problems.py
	$(UV) scripts/gen_data/gen_reasoning.py
	$(UV) scripts/gen_data/gen_corpus.py
extend-data-from-generated: raw-data
	TRAIN_CSV=$(GENERATED_TRAIN_CSV) $(UV) scripts/gen_data/gen_problems.py
	TRAIN_CSV=$(GENERATED_TRAIN_CSV) $(UV) scripts/gen_data/gen_reasoning.py
	TRAIN_CSV=$(GENERATED_TRAIN_CSV) $(UV) scripts/gen_data/gen_corpus.py
extend-data-from-results:
	$(UV) scripts/gen_data/gen_result.py

train-model:
	$(UV) scripts/train/sft.py $(TRAIN_ARGS)
train-model-from-generated: extend-data-from-generated
	$(UV) scripts/train/sft.py $(TRAIN_GENERATED_ARGS)
train-model-from-pretrained:
	@test -n "$(MODEL_CHECKPOINT)" || (echo "MODEL_CHECKPOINT is required. Example: make train-model-from-pretrained MODEL_CHECKPOINT='tinker://.../weights/final'" >&2; exit 1)
	$(UV) scripts/train/sft.py $(TRAIN_FROM_PRETRAINED_ARGS)
train-model-from-pretrained-generated: extend-data-from-generated
	@test -n "$(MODEL_CHECKPOINT)" || (echo "MODEL_CHECKPOINT is required. Example: make train-model-from-pretrained-generated MODEL_CHECKPOINT='tinker://.../weights/final'" >&2; exit 1)
	$(UV) scripts/train/sft.py $(TRAIN_FROM_PRETRAINED_GENERATED_ARGS)
train-model-modal:
	$(UV_MODAL_DEPLOY) scripts/train/sft.py $(TRAIN_ARGS)
train-model-modal-generated: extend-data-from-generated
	$(UV_MODAL_DEPLOY) scripts/train/sft.py $(TRAIN_GENERATED_ARGS)

upload-adapter:
	$(UV_MODAL_RUN) --detach scripts/submit/upload_adapter_to_kaggle.py

infer-result:
	$(UV) scripts/infer/infer.py $(INFER_ARGS)

infer-vllm-result:
	$(UV) scripts/infer/vllm/infer.py $(VLLM_INFER_ARGS) $(VLLM_INFER_EXTRA_ARGS)
