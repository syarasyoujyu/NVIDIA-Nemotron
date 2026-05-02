import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

TRAIN_CSV = Path(os.environ.get("TRAIN_CSV", DATA_DIR / "train.csv"))
TEST_CSV = Path(os.environ.get("TEST_CSV", DATA_DIR / "test.csv"))
AUGMENTATIONS_DIR = DATA_DIR / "augmentations"
PROBLEMS_INDEX = DATA_DIR / "problems.jsonl"
PROBLEM_DIR = DATA_DIR / "problem"
REASONING_DIR = DATA_DIR / "reasoning"
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_INDEX = DATA_DIR / "corpus.jsonl"
CORPUS_INFER_DIR = DATA_DIR / "corpus_infer"
CORPUS_INFER_INDEX = DATA_DIR / "corpus_infer.jsonl"
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "data/tokenizer.json"))
INVESTIGATIONS_DIR = DATA_DIR / "investigations"
SFT_DIR = DATA_DIR / "training/sft"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
TOKEN_LIMIT = 8192
