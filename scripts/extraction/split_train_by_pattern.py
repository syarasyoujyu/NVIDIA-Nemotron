#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


Rule = tuple[str, str, Callable[[str], bool]]


def first_line(text: str) -> str:
    return text.splitlines()[0].strip()


def _extract_examples_and_target(prompt: str) -> tuple[list[str], str | None]:
    """例題の左辺リストとターゲット入力を返す"""
    examples_lhs: list[str] = []
    target: str | None = None
    for line in prompt.splitlines():
        line = line.strip()
        if " = " in line and not any(
            kw in line for kw in ["Wonderland", "determine", "examples", "rules", "result", "few", "applied", "set of", "secret"]
        ):
            examples_lhs.append(line.split(" = ", 1)[0].strip())
        elif line.startswith("Now, determine the result for: "):
            target = line.split(": ", 1)[1].strip()
    return examples_lhs, target


def _get_operator(expr: str) -> str | None:
    m = re.match(r"^\d+([^\d]+)\d+$", expr.strip())
    return m.group(1) if m else None


def classify_deduce_guess(prompt: str, pattern: str) -> str:
    """deduce: ターゲットのルールが例題から推論可能 / guess: 不明な記号あり"""
    examples_lhs, target = _extract_examples_and_target(prompt)
    if not target or not examples_lhs:
        return "deduce"

    if pattern == "symbol_equation":
        example_chars = set("".join(examples_lhs))
        return "deduce" if all(c in example_chars for c in target) else "guess"

    if pattern == "numeric_equation":
        target_op = _get_operator(target)
        if not target_op:
            return "deduce"
        example_ops = {op for lhs in examples_lhs if (op := _get_operator(lhs))}
        return "deduce" if target_op in example_ops else "guess"

    return "deduce"


def _is_equation_with_numbers(prompt: str) -> bool:
    import re
    for line in prompt.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        if any(kw in line for kw in ["Wonderland", "determine", "examples", "rules", "result", "few", "applied", "set of", "secret"]):
            continue
        lhs = line.split("=", 1)[0]
        if re.search(r"\d", lhs):
            return True
    return False


RULES: list[Rule] = [
    (
        "bit_manipulation",
        "8-bit binary transformation",
        lambda prompt: "secret bit manipulation rule transforms 8-bit binary numbers"
        in prompt,
    ),
    (
        "text_decryption",
        "encrypted text to plain text",
        lambda prompt: "secret encryption rules are used on text" in prompt,
    ),
    (
        "roman_numeral",
        "integer to Roman numeral conversion",
        lambda prompt: "converted into a different numeral system" in prompt,
    ),
    (
        "unit_conversion",
        "secret unit conversion on measurements",
        lambda prompt: "secret unit conversion is applied to measurements" in prompt,
    ),
    (
        "gravity_distance",
        "falling distance from modified gravitational constant",
        lambda prompt: "gravitational constant has been secretly changed" in prompt,
    ),
    (
        "numeric_equation",
        "equation transformation with numeric operands",
        lambda prompt: "secret set of transformation rules is applied to equations"
        in prompt
        and _is_equation_with_numbers(prompt),
    ),
    (
        "symbol_equation",
        "equation transformation with symbol operands",
        lambda prompt: "secret set of transformation rules is applied to equations"
        in prompt
        and not _is_equation_with_numbers(prompt),
    ),
]


def classify_prompt(prompt: str) -> str | None:
    for label, _, matcher in RULES:
        if matcher(prompt):
            return label
    return None


def split_dataset(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")

        rows_by_pattern: dict[str, list[dict[str, str]]] = defaultdict(list)
        first_line_counts: Counter[str] = Counter()
        unmatched_rows: list[dict[str, str]] = []

        for row in reader:
            prompt = row["prompt"]
            pattern = classify_prompt(prompt)
            row = dict(row)
            if pattern is None:
                unmatched_rows.append(
                    {
                        "id": row.get("id", ""),
                        "prompt_first_line": first_line(prompt),
                    }
                )
                continue

            row["pattern"] = pattern
            rows_by_pattern[pattern].append(row)
            first_line_counts[first_line(prompt)] += 1

    output_fieldnames = list(fieldnames) + ["pattern"]
    summary = []

    _DEDUCE_GUESS_PATTERNS = {"numeric_equation", "symbol_equation"}

    for label, description, _ in RULES:
        rows = rows_by_pattern.get(label, [])
        pattern_dir = output_dir / label
        pattern_dir.mkdir(parents=True, exist_ok=True)
        out_path = pattern_dir / "rows.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary.append(
            {
                "pattern": label,
                "description": description,
                "count": len(rows),
                "output_file": str(out_path.relative_to(output_dir)),
            }
        )

        if label in _DEDUCE_GUESS_PATTERNS:
            for subcat in ("deduce", "guess"):
                subcat_rows = [
                    r for r in rows
                    if classify_deduce_guess(r["prompt"], label) == subcat
                ]
                subcat_dir = pattern_dir / subcat
                subcat_dir.mkdir(parents=True, exist_ok=True)
                subcat_path = subcat_dir / "rows.csv"
                with subcat_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=output_fieldnames)
                    writer.writeheader()
                    writer.writerows(subcat_rows)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": str(input_csv),
                "total_rows": sum(len(rows) for rows in rows_by_pattern.values()),
                "patterns": summary,
                "prompt_first_line_counts": dict(first_line_counts),
                "unmatched_count": len(unmatched_rows),
                "unmatched_ids": [row["id"] for row in unmatched_rows],
                "unmatched_rows": unmatched_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    unmatched_path = output_dir / "unmatched_cases.csv"
    with unmatched_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt_first_line"])
        writer.writeheader()
        writer.writerows(unmatched_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split NVIDIA Nemotron train.csv into files by prompt pattern."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/patterns"),
        help="Directory where the split CSV files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(args.input, args.output_dir)
    print(f"Saved split files to {args.output_dir}")


if __name__ == "__main__":
    main()
