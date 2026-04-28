"""Evaluate JSONL generations produced by ``scripts/infer/infer.py``."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.gen_data.gen_reasoning import compare_answer


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from a model response."""
    if text is None:
        return "NOT_FOUND"

    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rate(numer: int, denom: int) -> float:
    return numer / denom if denom else 0.0


def _empty_bucket() -> dict[str, Any]:
    return {
        "samples": 0,
        "correct_samples": 0,
        "problems": set(),
        "correct_problems": set(),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_samples = len(rows)
    correct_samples = sum(1 for row in rows if row["correct"])

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)
    for row in rows:
        pid = row["problem_id"]
        category = row.get("category") or "unknown"
        by_problem[pid].append(row)
        bucket = by_category[category]
        bucket["samples"] += 1
        bucket["problems"].add(pid)
        if row["correct"]:
            bucket["correct_samples"] += 1
            bucket["correct_problems"].add(pid)

    problem_correct = {
        pid: any(row["correct"] for row in problem_rows)
        for pid, problem_rows in by_problem.items()
    }
    correct_problem_count = sum(problem_correct.values())

    category_summary = {}
    for category, bucket in sorted(by_category.items()):
        problems = len(bucket["problems"])
        correct_problems = len(bucket["correct_problems"])
        samples = bucket["samples"]
        correct = bucket["correct_samples"]
        category_summary[category] = {
            "samples": samples,
            "correct_samples": correct,
            "sample_accuracy": _rate(correct, samples),
            "problems": problems,
            "correct_problems": correct_problems,
            "pass_at_k": _rate(correct_problems, problems),
        }

    return {
        "samples": total_samples,
        "correct_samples": correct_samples,
        "sample_accuracy": _rate(correct_samples, total_samples),
        "problems": len(by_problem),
        "correct_problems": correct_problem_count,
        "pass_at_k": _rate(correct_problem_count, len(by_problem)),
        "by_category": category_summary,
    }


def evaluate_rows(generation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated = []
    for row in generation_rows:
        output_text = str(row.get("output_text", ""))
        expected = str(row.get("answer", ""))
        predicted = extract_final_answer(output_text)
        correct = compare_answer(expected, predicted)
        evaluated.append(
            {
                **row,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "correct": correct,
            }
        )
    return evaluated


def evaluate_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name("eval.jsonl")
    summary_path = (
        Path(summary_path) if summary_path else input_path.with_name("summary.json")
    )

    evaluated = evaluate_rows(load_jsonl(input_path))
    summary = build_summary(evaluated)

    write_jsonl(output_path, evaluated)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "sample_accuracy="
        f"{summary['sample_accuracy']:.4f} "
        f"({summary['correct_samples']}/{summary['samples']}), "
        "pass_at_k="
        f"{summary['pass_at_k']:.4f} "
        f"({summary['correct_problems']}/{summary['problems']})"
    )
    print(f"Wrote evaluated rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference generations.")
    parser.add_argument("input", type=Path, help="Path to generations.jsonl.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_file(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
