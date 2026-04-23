.PHONY: help patterns patterns-split patterns-reports

UV := uv run python3
PATTERN_DIR := data/patterns

help:
	@echo "Available targets:"
	@echo "  make patterns         # split -> reports+unmatched を一気に実行"
	@echo "  make patterns-split   # data/train.csv を data/patterns に分割"
	@echo "  make patterns-reports # data/patterns から report と unmatched を生成"

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