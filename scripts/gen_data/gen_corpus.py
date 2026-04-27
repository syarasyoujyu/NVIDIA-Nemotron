"""reasoning/*.txt の推論テキストを使って合成学習コーパスを作成する。

各エントリーの completion は以下の形式:
    (推論テキスト)</think>\\boxed{(answer)}<|im_end|>

冒頭の <think>\\n はチャットテンプレートのプロンプト側に含まれるため、
推論テキストはそのまま続く形になる。

出力:
- corpus.jsonl                            - エントリーごとのメタデータインデックス
- corpus_token_counts.csv                 - エントリーごとの token count 明細
- corpus_unmasked_token_stats.json        - カテゴリ別 unmasked token count 集計
- corpus/<problem_id>/synthetic.jsonl     - マスク済み/非マスクを交互に並べたセグメントファイル

使い方:
    uv run corpus.py
"""

import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.basic.common import load_jsonl
from scripts.basic.const import (
    AUGMENTATIONS_DIR,
    CORPUS_DIR,
    CORPUS_INDEX,
    PROBLEMS_INDEX,
    PROMPT_SUFFIX,
    REASONING_DIR,
    TOKEN_LIMIT,
    TOKENIZER_PATH,
    TRAIN_CSV,
)

load_dotenv()

UNMASKED_TOKEN_THRESHOLDS = (2000, 4000, 6000, 8000)
TOKEN_COUNTS_PATH = CORPUS_INDEX.with_name("corpus_token_counts.csv")
UNMASKED_TOKEN_STATS_PATH = CORPUS_INDEX.with_name(
    "corpus_unmasked_token_stats.json"
)


def tokenize_prompt(
    prompt_text: str,
    chat_tokenizer: AutoTokenizer,
    *,
    suffix: str = PROMPT_SUFFIX,
) -> list[int]:
    """チャットテンプレートを使って問題プロンプトをトークナイズする（query.py と同じ方式）。"""
    messages = [{"role": "user", "content": prompt_text + suffix}]
    result = chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # enable_thinking=True のとき BatchEncoding が返る場合があるので list[int] に正規化する
    if not isinstance(result, list):
        result = result["input_ids"]
    return result


@dataclass
class CorpusEntry:
    problem_id: str
    category: str
    tokens: list[int]
    mask: list[int]
    prompt_token_count: int
    masked_token_count: int
    unmasked_token_count: int
    answer: str
    included: bool = False

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "segment": "synthetic.jsonl",
            "category": self.category,
            "prompt_token_count": self.prompt_token_count,
            "masked_token_count": self.masked_token_count,
            "unmasked_token_count": self.unmasked_token_count,
            "token_count": self.token_count,
            "answer": self.answer,
            "included": self.included,
        }


def build_segments(
    tokens: list[int],
    mask: list[int],
) -> list[dict]:
    """トークン列とマスクからセグメントリストを構築する。"""
    if not tokens:
        return []

    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"

    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append(
                {
                    "type": current_type,
                    "pos": seg_start,
                    "tokens": tokens[seg_start:i],
                }
            )
            seg_start = i
            current_type = token_type

    segments.append(
        {
            "type": current_type,
            "pos": seg_start,
            "tokens": tokens[seg_start:],
        }
    )

    return segments


def _threshold_key(threshold: int) -> str:
    return f"gte_{threshold}"


def _build_unmasked_token_stats(entries: list[CorpusEntry]) -> dict:
    """カテゴリ別の unmasked token count 統計を構築する。"""
    by_category: dict[str, list[CorpusEntry]] = defaultdict(list)
    for entry in entries:
        by_category[entry.category].append(entry)

    def summarize(category_entries: list[CorpusEntry]) -> dict[str, object]:
        unmasked_counts = [entry.unmasked_token_count for entry in category_entries]
        prompt_counts = [entry.prompt_token_count for entry in category_entries]
        unmasked_total = sum(unmasked_counts)
        prompt_total = sum(prompt_counts)
        summary: dict[str, object] = {
            "count": len(unmasked_counts),
            "prompt_token_total": prompt_total,
            "prompt_token_min": min(prompt_counts) if prompt_counts else 0,
            "prompt_token_max": max(prompt_counts) if prompt_counts else 0,
            "prompt_token_avg": (
                round(prompt_total / len(prompt_counts), 3) if prompt_counts else 0
            ),
            "unmasked_token_total": unmasked_total,
            "unmasked_token_min": min(unmasked_counts) if unmasked_counts else 0,
            "unmasked_token_max": max(unmasked_counts) if unmasked_counts else 0,
            "unmasked_token_avg": (
                round(unmasked_total / len(unmasked_counts), 3)
                if unmasked_counts
                else 0
            ),
        }
        for threshold in UNMASKED_TOKEN_THRESHOLDS:
            summary[_threshold_key(threshold)] = sum(
                count >= threshold for count in unmasked_counts
            )
        return summary

    return {
        "threshold_basis": "unmasked_token_count",
        "thresholds": list(UNMASKED_TOKEN_THRESHOLDS),
        "total": summarize(entries),
        "by_category": {
            category: summarize(category_entries)
            for category, category_entries in sorted(by_category.items())
        },
    }


def _write_token_count_outputs(entries: list[CorpusEntry]) -> dict:
    """Token count の明細 CSV とカテゴリ別 unmasked 集計 JSON を書き出す。"""
    with open(TOKEN_COUNTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "category",
                "prompt_token_count",
                "token_count",
                "masked_token_count",
                "unmasked_token_count",
                "included",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "problem_id": entry.problem_id,
                    "category": entry.category,
                    "prompt_token_count": entry.prompt_token_count,
                    "token_count": entry.token_count,
                    "masked_token_count": entry.masked_token_count,
                    "unmasked_token_count": entry.unmasked_token_count,
                    "included": entry.included,
                }
            )

    stats = _build_unmasked_token_stats(entries)
    with open(UNMASKED_TOKEN_STATS_PATH, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return stats


def main() -> None:
    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found. Run problems.py first.")
        return

    # トークナイザーを読み込む
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
    )

    # train.csv から問題プロンプトと回答を読み込む
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # 問題カテゴリを読み込む
    problem_cats: dict[str, str] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        problem_cats[prob_raw["id"]] = prob_raw["category"]

    # コーパスディレクトリを初期化する
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    entries: list[CorpusEntry] = []

    # 推論ファイルが存在する問題を処理する
    problem_ids = sorted(
        pid
        for pid in problem_cats
        if (REASONING_DIR / f"{pid}.txt").exists() and pid in prompts
    )

    for problem_id in tqdm(problem_ids):
        category = problem_cats[problem_id]
        answer = answers[problem_id]

        reasoning_text = (REASONING_DIR / f"{problem_id}.txt").read_text().rstrip("\n")

        # 推論テキスト内の \boxed{} から答えを抽出して整合させる
        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
        reasoning_answer = boxed_match[-1] if boxed_match else answer
        completion_text = (
            f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
        )
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids

        # プロンプトをトークナイズする（raw/ への依存なし）
        prompt_ids = tokenize_prompt(prompts[problem_id], chat_tokenizer)
        prompt_token_count = len(prompt_ids)

        all_tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        # トークン上限を超える場合は切り捨てる
        if len(all_tokens) > TOKEN_LIMIT:
            all_tokens = all_tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]

        unmasked_count = sum(mask)
        masked_count = len(mask) - unmasked_count

        entry = CorpusEntry(
            problem_id=problem_id,
            category=category,
            tokens=all_tokens,
            mask=mask,
            prompt_token_count=prompt_token_count,
            masked_token_count=masked_count,
            unmasked_token_count=unmasked_count,
            answer=answer,
            included=True,
        )

        # マスク済み/非マスクを交互に並べたセグメントを構築してファイルに書き出す
        segments = build_segments(all_tokens, mask)

        problem_dir = CORPUS_DIR / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        seg_path = problem_dir / "synthetic.jsonl"

        with open(seg_path, "w") as f:
            for seg in segments:
                json.dump(seg, f)
                f.write("\n")

        entries.append(entry)

    # augmentations/*.txt を処理する（推論なし・\boxed{} なし）
    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            # [category]・[prompt]・[completion] セクションをパースする
            category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")

            problem_id = aug_path.stem

            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = tokenizer.encode(
                completion_text, add_special_tokens=False
            ).ids

            prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer, suffix="")
            prompt_token_count = len(prompt_ids)

            all_tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            assert len(all_tokens) <= TOKEN_LIMIT, (
                f"augmented entry {problem_id} exceeds token limit: "
                f"{len(all_tokens)} > {TOKEN_LIMIT}"
            )

            unmasked_count = sum(mask)
            masked_count = len(mask) - unmasked_count

            entry = CorpusEntry(
                problem_id=problem_id,
                category=category,
                tokens=all_tokens,
                mask=mask,
                prompt_token_count=prompt_token_count,
                masked_token_count=masked_count,
                unmasked_token_count=unmasked_count,
                answer=completion,
                included=True,
            )

            segments = build_segments(all_tokens, mask)
            problem_dir = CORPUS_DIR / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as sf:
                for seg in segments:
                    json.dump(seg, sf)
                    sf.write("\n")

            entries.append(entry)

    entries.sort(key=lambda e: e.problem_id)

    # インデックス JSONL を書き出す
    with open(CORPUS_INDEX, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f)
            f.write("\n")

    unmasked_token_stats = _write_token_count_outputs(entries)

    # 統計を表示する
    cat_counts: dict[str, int] = {cat: 0 for cat in {e.category for e in entries}}
    cat_tokens: dict[str, int] = {cat: 0 for cat in cat_counts}
    cat_prompt_tokens: dict[str, int] = {cat: 0 for cat in cat_counts}
    for e in entries:
        cat_counts[e.category] += 1
        cat_tokens[e.category] += e.unmasked_token_count
        cat_prompt_tokens[e.category] += e.prompt_token_count

    total_unmasked = sum(e.unmasked_token_count for e in entries)
    total_masked = sum(e.masked_token_count for e in entries)
    total_prompt = sum(e.prompt_token_count for e in entries)
    max_tokens = max((e.token_count for e in entries), default=0)
    max_prompt_tokens = max((e.prompt_token_count for e in entries), default=0)

    print(f"Corpus (synthetic): {len(entries)} entries")
    print(f"Unmasked tokens: {total_unmasked:,}")
    print(f"Masked tokens:   {total_masked:,}")
    print(f"Prompt tokens:   {total_prompt:,}")
    print(f"Max seq length:  {max_tokens:,}")
    print(f"Max prompt len:  {max_prompt_tokens:,}")
    print(f"Token count details:      {TOKEN_COUNTS_PATH}")
    print(f"Unmasked token stats:     {UNMASKED_TOKEN_STATS_PATH}")
    print()
    for cat in sorted(cat_counts):
        stats = unmasked_token_stats["by_category"][cat]
        threshold_text = ", ".join(
            f">={threshold}: {stats[_threshold_key(threshold)]}"
            for threshold in UNMASKED_TOKEN_THRESHOLDS
        )
        print(
            f"  {cat}: {cat_counts[cat]} runs, "
            f"{cat_tokens[cat]:,} unmasked tokens, "
            f"{cat_prompt_tokens[cat]:,} prompt tokens "
            f"(unmasked {threshold_text})"
        )


if __name__ == "__main__":
    main()
