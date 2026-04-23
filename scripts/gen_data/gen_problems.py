"""train.csv を読み込み、problems.jsonl と data/problem/<id>.json を生成する。

1. プロンプト文字列からカテゴリ・例示・質問を解析する
2. data/problem/<id>.json にフルの問題データを保存する
3. 推論ジェネレーターを実行して status を決定する
4. data/problems.jsonl にメタデータインデックスを書き出す

使い方:
    uv run gen_problems.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter

from scripts.basic.const import (
    INVESTIGATIONS_DIR,
    PROBLEM_DIR,
    PROBLEMS_INDEX,
    TRAIN_CSV,
)
from scripts.basic.types import GENERATORS
from scripts.cot_prompt.store_types import Example, Problem

# ---------------------------------------------------------------------------
# カテゴリ検出
# ---------------------------------------------------------------------------

def _detect_category(prompt: str) -> str:
    """プロンプトの先頭行からカテゴリを判定する。"""
    first = prompt.split("\n")[0]
    if "bit manipulation" in first:
        return "bit_manipulation"
    if "encryption" in first:
        return "cipher"
    if "numeral system" in first:
        return "numeral"
    if "unit conversion" in first:
        return "unit_conversion"
    if "gravitational" in first:
        return "gravity"
    if "transformation rules" in first:
        return _detect_equation_category(prompt)
    return "unknown"


def _detect_equation_category(prompt: str) -> str:
    """数値方程式か記号式か、deduce か guess かを判定する。"""
    ex_lines = [
        line.strip() for line in prompt.splitlines()
        if " = " in line and "determine" not in line.lower() and not line.startswith("Now")
    ]
    if not ex_lines:
        return "cryptarithm_deduce"

    lhs = ex_lines[0].split(" = ")[0].strip()
    is_numeric = bool(re.match(r"\d+\S\d+", lhs))

    if is_numeric:
        # 数値方程式: 例示の演算子と質問の演算子を比較
        ex_ops = set(re.findall(r"\d+(\D)\d+\s*=", prompt))
        q_match = re.search(r"determine the result for:\s*(\S+)", prompt)
        q_op_m = re.search(r"\d+(\D)\d+", q_match.group(1)) if q_match else None
        q_op = q_op_m.group(1) if q_op_m else None
        is_guess = q_op is not None and q_op not in ex_ops
        return "equation_numeric_guess" if is_guess else "equation_numeric_deduce"
    else:
        # 記号式: 各LHSの3文字目を演算子とみなす
        ex_ops = set()
        for ex_line in ex_lines:
            lhs_s = ex_line.split(" = ")[0].strip()
            if len(lhs_s) >= 3:
                ex_ops.add(lhs_s[2])
        q_match = re.search(r"determine the result for:\s*(\S+)", prompt)
        q_expr = q_match.group(1) if q_match else ""
        q_op = q_expr[2] if len(q_expr) >= 3 else None
        is_guess = q_op is not None and q_op not in ex_ops
        return "cryptarithm_guess" if is_guess else "cryptarithm_deduce"


# ---------------------------------------------------------------------------
# 例示・質問のパース
# ---------------------------------------------------------------------------

def _parse_bit_manipulation(prompt: str) -> tuple[list[Example], str]:
    """8ビット列の入出力ペアと質問ビット列を抽出する。"""
    examples = [
        Example(m.group(1), m.group(2))
        for m in re.finditer(r"([01]{8}) -> ([01]{8})", prompt)
    ]
    q = re.search(r"determine the output for:\s*([01]{8})", prompt)
    return examples, q.group(1) if q else ""


def _parse_cipher(prompt: str) -> tuple[list[Example], str]:
    """暗号文→平文の対応ペアと質問暗号文を抽出する。"""
    examples = []
    question = ""
    for line in prompt.splitlines():
        line = line.strip()
        if " -> " in line and not line.startswith("Now"):
            parts = line.split(" -> ", 1)
            # 8ビット列は bit_manipulation なので除外
            if len(parts) == 2 and not re.match(r"^[01]{8}$", parts[0].strip()):
                examples.append(Example(parts[0].strip(), parts[1].strip()))
        m = re.search(r"(?:decrypt|decode|decipher)[^:]*:\s*(.+)", line, re.IGNORECASE)
        if m:
            question = m.group(1).strip()
    return examples, question


def _parse_numeral(prompt: str) -> tuple[list[Example], str]:
    """アラビア数字→ローマ数字の対応ペアと質問数値を抽出する。"""
    examples = [
        Example(m.group(1), m.group(2))
        for m in re.finditer(r"(\d+)\s*->\s*([IVXLCDM]+)", prompt)
    ]
    q = re.search(r"write the number\s+(\d+)\s+in", prompt, re.IGNORECASE)
    return examples, q.group(1) if q else ""


def _parse_unit_conversion(prompt: str) -> tuple[list[Example], str]:
    """入力値→変換値のペアと質問入力値を抽出する。"""
    examples = [
        Example(m.group(1), m.group(2))
        for m in re.finditer(r"([\d.]+)\s+\S+\s+becomes\s+([\d.]+)", prompt)
    ]
    q = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
    return examples, q.group(1) if q else ""


def _parse_gravity(prompt: str) -> tuple[list[Example], str]:
    """時間→落下距離のペアと質問時間を抽出する。"""
    examples = [
        Example(m.group(1), m.group(2))
        for m in re.finditer(r"t\s*=\s*([\d.]+)s,\s*distance\s*=\s*([\d.]+)", prompt)
    ]
    # "Now, ..." 行の t= を質問として取る
    q_line = next((ln for ln in prompt.splitlines() if ln.startswith("Now")), "")
    q = re.search(r"t\s*=\s*([\d.]+)s", q_line)
    return examples, q.group(1) if q else ""


def _parse_equation(prompt: str) -> tuple[list[Example], str]:
    """数値方程式・記号式の入出力ペアと質問式を抽出する。"""
    examples = []
    ex_lines = [
        line.strip() for line in prompt.splitlines()
        if " = " in line and "determine" not in line.lower() and not line.startswith("Now")
    ]
    for ex_line in ex_lines:
        parts = ex_line.split(" = ", 1)
        if len(parts) == 2:
            examples.append(Example(parts[0].strip(), parts[1].strip()))
    q = re.search(r"determine the result for:\s*(.+)", prompt)
    return examples, q.group(1).strip() if q else ""


def _parse_prompt(category: str, prompt: str) -> tuple[list[Example], str]:
    """カテゴリに応じたパーサーを呼び出して (例示リスト, 質問) を返す。"""
    if category == "bit_manipulation":
        return _parse_bit_manipulation(prompt)
    if category == "cipher":
        return _parse_cipher(prompt)
    if category == "numeral":
        return _parse_numeral(prompt)
    if category == "unit_conversion":
        return _parse_unit_conversion(prompt)
    if category == "gravity":
        return _parse_gravity(prompt)
    # equation_numeric_* / cryptarithm_*
    return _parse_equation(prompt)


# ---------------------------------------------------------------------------
# 答え抽出と比較（check_cot.py と同方式）
# ---------------------------------------------------------------------------

def _extract_answer(reasoning_text: str) -> str:
    """\\boxed{...} から最後の答えを取り出す。"""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return ""


def _compare_answer(stored: str, predicted: str) -> bool:
    """答えが一致するかを検証する。数値は相対許容誤差 1e-2 以内なら一致。"""
    stored, predicted = stored.strip(), predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    PROBLEMS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 調査ファイルが存在する ID を事前収集
    investigation_ids = {p.stem for p in INVESTIGATIONS_DIR.glob("*.txt")}

    with open(TRAIN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} problems...")
    index_entries: list[dict] = []

    for i, row in enumerate(rows):
        pid = row["id"]
        prompt = row["prompt"]
        answer = row["answer"]

        # カテゴリ検出
        category = _detect_category(prompt)

        # 例示・質問のパース
        examples, question = _parse_prompt(category, prompt)

        # data/problem/<id>.json を書き出す
        problem_payload = {
            "id": pid,
            "category": category,
            "prompt": prompt,
            "answer": answer,
            "examples": [
                {"input_value": e.input_value, "output_value": e.output_value}
                for e in examples
            ],
            "question": question,
        }
        with open(PROBLEM_DIR / f"{pid}.jsonl", "w", encoding="utf-8") as f:
            json.dump(problem_payload, f, ensure_ascii=False)

        # ジェネレーターを実行して status を決定
        generator = GENERATORS.get(category)
        status = "rule_unknown"
        submission = ""

        if generator:
            problem = Problem(
                id=pid,
                category=category,  # type: ignore[arg-type]
                examples=examples,
                question=question,
                answer=answer,
                prompt=prompt,
            )
            try:
                reasoning_text = generator(problem)
                if reasoning_text is not None:
                    submission = _extract_answer(reasoning_text)
                    if _compare_answer(answer, submission):
                        status = "rule_found"
            except Exception:
                pass

        # ジェネレーターが失敗したが調査ファイルがある → hypothesis_formed
        if status == "rule_unknown" and pid in investigation_ids:
            status = "hypothesis_formed"

        index_entries.append({
            "id": pid,
            "category": category,
            "status": status,
            "submission": submission,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    # data/problems.jsonl を書き出す
    with open(PROBLEMS_INDEX, "w", encoding="utf-8") as f:
        for entry in index_entries:
            f.write(json.dumps(entry) + "\n")

    # 統計を表示する
    cat_counts = Counter(e["category"] for e in index_entries)
    status_counts = Counter(e["status"] for e in index_entries)
    rule_found = status_counts.get("rule_found", 0)
    total = len(index_entries)

    print(f"\nTotal: {total} problems")
    print(f"Status: {dict(status_counts)}")
    print(f"Accuracy: {rule_found}/{total} ({rule_found / total * 100:.1f}%)")
    print("\nCategories:")
    for cat, cnt in sorted(cat_counts.items()):
        found = sum(1 for e in index_entries if e["category"] == cat and e["status"] == "rule_found")
        print(f"  {cat}: {cnt} total, {found} rule_found")


if __name__ == "__main__":
    main()
