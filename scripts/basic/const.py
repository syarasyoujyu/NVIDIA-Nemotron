import os
from pathlib import Path

TRAIN_CSV = Path(os.environ.get("TRAIN_CSV", "data/train.csv"))
TEST_CSV = Path(os.environ.get("TEST_CSV", "data/test.csv"))
AUGMENTATIONS_DIR = Path("data/augmentations")
PROBLEMS_INDEX = Path("data/problems.jsonl")
PROBLEM_DIR = Path("data/problem")
REASONING_DIR = Path("data/reasoning")
CORPUS_DIR = Path("data/corpus")
CORPUS_INDEX = Path("data/corpus.jsonl")
CORPUS_INFER_DIR = Path("data/corpus_infer")
CORPUS_INFER_INDEX = Path("data/corpus_infer.jsonl")
TOKENIZER_PATH = Path("data/tokenizer.json")
INVESTIGATIONS_DIR = Path("data/investigations")
SFT_DIR = Path("data/training/sft")
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
TOKEN_LIMIT = 8192
