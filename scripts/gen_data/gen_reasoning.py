"""rule_found の各問題に対して決定的な推論テキストを生成する。

ルールが見つかった問題ごとに reasoning/<problem_id>.txt を作成する。
cryptarithm_guess はスキップ。推論はソルバーロジックを自然な
思考過程トレースとして反映する。

使い方:
    uv run reasoning.py
    uv run reasoning.py --delete-investigations   # 答えが正しい場合に調査ファイルを削除する
"""

import argparse
import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field

import pandas as pd

from scripts.basic.const import (
    INVESTIGATIONS_DIR,
    PROBLEMS_INDEX,
    REASONING_DIR,
    TRAIN_CSV,
)
from scripts.basic.types import GENERATORS, INVESTIGATION_CATEGORIES, SKIP_CATEGORIES
from scripts.cot_prompt.store_types import Problem


def extract_answer(reasoning_text: str) -> str:
    """\\boxed{...} から答えを抽出する（metric_reference.extract_final_answer と同じ方式）。"""
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


def write_reasoning_column_to_train_csv() -> int:
    """TRAIN_CSV に reasoning/<id>.txt の内容を reasoning カラムとして書き戻す。"""
    if not TRAIN_CSV.exists():
        print(f"No {TRAIN_CSV} found; skipped writing reasoning column.")
        return 0

    df = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    if "id" not in df.columns:
        print(f"No id column found in {TRAIN_CSV}; skipped writing reasoning column.")
        return 0

    reasoning_by_id = {
        path.stem: path.read_text(encoding="utf-8")
        for path in REASONING_DIR.glob("*.txt")
    }
    df["reasoning"] = df["id"].map(reasoning_by_id).fillna("")
    updated = int(df["reasoning"].astype(bool).sum())
    df.to_csv(TRAIN_CSV, index=False)

    return updated


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

    train_reasoning_rows = write_reasoning_column_to_train_csv()

    # 精度統計を表示する
    total = sum(c.total for c in category_stats.values())
    rule_found = sum(c.rule_found for c in category_stats.values())
    print(f"\nGenerated {generated} reasoning files in {REASONING_DIR}/")
    print(f"Updated reasoning column for {train_reasoning_rows} rows in {TRAIN_CSV}")
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
