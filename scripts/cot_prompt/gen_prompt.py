"""規則発見済みの各問題に対して決定的な推論テキストを生成する。

規則が見つかった各問題について reasoning/<problem_id>.txt を作成する。
cryptarithm_guess はスキップする。推論はソルバーロジックを自然な
思考過程トレースとして反映する。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts.cot_prompt.numeral import reasoning_numeral

from scripts.cot_prompt.bit_manipulation import reasoning_bit_manipulation
from scripts.cot_prompt.cipher import reasoning_cipher
from scripts.cot_prompt.cryptarithm import reasoning_cryptarithm
from scripts.cot_prompt.equation_numeric import reasoning_equation_numeric
from scripts.cot_prompt.gravity import reasoning_gravity
from scripts.cot_prompt.store_types import Problem
from scripts.cot_prompt.unit_conversion import reasoning_unit_conversion

PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
INVESTIGATIONS_DIR = Path(__file__).parent / "investigations"
INVESTIGATION_CATEGORIES: set[str] = {
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}

SKIP_CATEGORIES: set[str] = set()

GENERATORS: dict[str, Callable] = {
    "gravity": reasoning_gravity,
    "unit_conversion": reasoning_unit_conversion,
    "cipher": reasoning_cipher,
    "bit_manipulation": reasoning_bit_manipulation,
    "numeral": reasoning_numeral,
    "equation_numeric_deduce": reasoning_equation_numeric,
    "equation_numeric_guess": reasoning_equation_numeric,
    "cryptarithm_deduce": reasoning_cryptarithm,
    "cryptarithm_guess": reasoning_cryptarithm,
}


def extract_answer(reasoning_text: str) -> str:
    """既存の答え抽出処理と同じ形で \\boxed{...} から答えを抽出する。"""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored_answer: str, predicted: str) -> bool:
    """答えが一致するかを検証する。

    数値の答えは一定の相対許容誤差（1e-2）内なら一致とみなす。
    それ以外は文字列として厳密に比較する（大文字小文字は無視）。

    例:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # 文字列を整える
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # 答えが2進文字列なら、文字列として厳密に比較する
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # 答えを浮動小数点数へ変換してみる
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # ゼロ付近の数値には小さな絶対許容誤差を使う
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # フォールバックとして大文字小文字を無視した文字列比較を行う
        return predicted.lower() == stored_answer.lower()


@dataclass
class CategoryCounts:
    rule_found: int = 0
    total: int = 0
    runtimes: list[float] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete-investigations",
        action="store_true",
        help="Delete investigation files when answer is correct",
    )
    args = parser.parse_args()

    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found.")
        return

    # 既存エントリを読み込んでフィールドを保持し、結果をマージして戻す
    existing: dict[str, dict] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                existing[entry["id"]] = entry

    if REASONING_DIR.exists():
        shutil.rmtree(REASONING_DIR)
    REASONING_DIR.mkdir(parents=True)
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)

    stats: dict[str, bool] = {}
    category_stats: dict[str, CategoryCounts] = {}
    generated = 0
    skipped = 0

    for entry in existing.values():
        pid = entry["id"]
        category = entry["category"]

        cat = category
        if cat not in category_stats:
            category_stats[cat] = CategoryCounts()
        category_stats[cat].total += 1

        if category in SKIP_CATEGORIES:
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        generator = GENERATORS.get(category)
        if not generator:
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        problem = Problem.load_from_json(pid)
        t0 = time.perf_counter()
        reasoning_text = generator(problem)
        elapsed = time.perf_counter() - t0
        category_stats[cat].runtimes.append(elapsed)

        if reasoning_text is None:
            # 調査ファイルを reasoning/ へコピーするフォールバックは行わない
            skipped += 1
            existing[pid]["status"] = "rule_unknown"
            existing[pid]["submission"] = ""
            continue

        submission = extract_answer(reasoning_text)
        result = compare_answer(problem.answer, submission)
        stats[pid] = result
        existing[pid]["status"] = "rule_found" if result else "rule_unknown"
        existing[pid]["submission"] = submission

        if result:
            category_stats[cat].rule_found += 1

        out_path = REASONING_DIR / f"{pid}.txt"
        with open(out_path, "w") as f:
            f.write(reasoning_text)

        if category in INVESTIGATION_CATEGORIES:
            inv_path = INVESTIGATIONS_DIR / f"{pid}.txt"
            if result and args.delete_investigations and inv_path.exists():
                inv_path.unlink()

        generated += 1

    # 調査ファイルがある問題の状態を更新する（まだ規則発見済みでない場合のみ）
    hypothesis_formed = 0
    for inv_path in INVESTIGATIONS_DIR.glob("*.txt"):
        pid = inv_path.stem
        if pid not in existing:
            continue
        if existing[pid]["status"] == "rule_found":
            continue
        existing[pid]["status"] = "hypothesis_formed"
        hypothesis_formed += 1

    # マージ済みの結果を problems.jsonl へ書き戻す
    with PROBLEMS_INDEX.open("w") as f:
        for entry in existing.values():
            entry.pop("has_investigation", None)
            f.write(json.dumps(entry) + "\n")

    # 精度統計を表示する
    total = sum(c.total for c in category_stats.values())
    rule_found = sum(c.rule_found for c in category_stats.values())
    print(f"\nGenerated {generated} reasoning files in {REASONING_DIR}/")
    if skipped:
        print(f"Skipped {skipped} (no generator for category)")
    if hypothesis_formed:
        print(
            f"Hypothesis formed: {hypothesis_formed} (investigation without reasoning)"
        )
    w = 64
    print(f"\n{'=' * w}")
    print(f"{'Category':<28} {'Found':>6} {'Total':>6} {'Accuracy':>10} {'Avg ms':>10}")
    print(f"{'-' * w}")
    all_runtimes: list[float] = []
    for category_name, counts in sorted(category_stats.items()):
        acc = counts.rule_found / counts.total * 100 if counts.total else 0
        avg_ms = (
            sum(counts.runtimes) / len(counts.runtimes) * 1000 if counts.runtimes else 0
        )
        all_runtimes.extend(counts.runtimes)
        acc_str = f"{acc:.1f}%"
        print(
            f"{category_name:<28} {counts.rule_found:>6} {counts.total:>6} {acc_str:>10} {avg_ms:>10.1f}"
        )
    print(f"{'-' * w}")
    overall_acc = rule_found / total * 100 if total else 0
    overall_avg_ms = sum(all_runtimes) / len(all_runtimes) * 1000 if all_runtimes else 0
    overall_acc_str = f"{overall_acc:.1f}%"
    print(
        f"{'TOTAL':<28} {rule_found:>6} {total:>6} {overall_acc_str:>10} {overall_avg_ms:>10.1f}"
    )
    print(f"{'=' * w}")
    print("\nIf you were given an example to fix, please verify that example.")
    print(
        "\nIf the user has previously asked to run corpus.py, you should run `uv run corpus.py`"
    )


if __name__ == "__main__":
    main()
