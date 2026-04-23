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

    if pattern == "cryptarithm":
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
        "cipher",
        "encrypted text to plain text",
        lambda prompt: "secret encryption rules are used on text" in prompt,
    ),
    (
        "numeral",
        "integer to Roman numeral conversion",
        lambda prompt: "converted into a different numeral system" in prompt,
    ),
    (
        "unit_conversion",
        "secret unit conversion on measurements",
        lambda prompt: "secret unit conversion is applied to measurements" in prompt,
    ),
    (
        "gravity",
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
        "cryptarithm",
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


DEDUCE_GUESS_PATTERNS = {"numeric_equation", "cryptarithm"}


def classify_category(prompt: str) -> str | None:
    pattern = classify_prompt(prompt)
    if pattern is None:
        return None
    if pattern in DEDUCE_GUESS_PATTERNS:
        return f"{pattern}_{classify_deduce_guess(prompt, pattern)}"
    return pattern


def write_dataset_with_category(input_csv: Path, output_csv: Path) -> dict[str, object]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    category_counts: Counter[str] = Counter()
    first_line_counts: Counter[str] = Counter()
    unmatched_rows: list[dict[str, str]] = []
    total_rows = 0

    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")
        if "prompt" not in fieldnames:
            raise ValueError(f"No prompt column found in {input_csv}")

        output_fieldnames = list(fieldnames)
        if "category" not in output_fieldnames:
            output_fieldnames.append("category")

        with output_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                prompt = row["prompt"]
                category = classify_category(prompt)
                if category is None:
                    category = "unmatched"
                    unmatched_rows.append(
                        {
                            "id": row.get("id", ""),
                            "prompt_first_line": first_line(prompt),
                        }
                    )

                row = dict(row)
                row["category"] = category
                writer.writerow({name: row.get(name, "") for name in output_fieldnames})

                category_counts[category] += 1
                first_line_counts[first_line(prompt)] += 1

    return {
        "input_file": str(input_csv),
        "output_file": str(output_csv),
        "total_rows": total_rows,
        "category_counts": dict(sorted(category_counts.items())),
        "prompt_first_line_counts": dict(first_line_counts),
        "unmatched_count": len(unmatched_rows),
        "unmatched_ids": [row["id"] for row in unmatched_rows],
        "unmatched_rows": unmatched_rows,
    }


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

        if label in DEDUCE_GUESS_PATTERNS:
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
        description=(
            "Add detailed pattern category labels to NVIDIA Nemotron train/test CSVs."
        )
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to the source train CSV file.",
    )
    parser.add_argument(
        "--input",
        dest="train_input",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("data/test.csv"),
        help="Path to the source test CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/patterns"),
        help="Directory where train_pattern.csv and test_pattern.csv will be written.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Path to the output train CSV. Defaults to OUTPUT_DIR/train_pattern.csv.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Path to the output test CSV. Defaults to OUTPUT_DIR/test_pattern.csv.",
    )
    parser.add_argument(
        "--legacy-split",
        action="store_true",
        help=(
            "Also write the old train-only pattern directory split under OUTPUT_DIR. "
            "This preserves the previous rows.csv layout for report generation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_output = args.train_output or args.output_dir / "train_pattern.csv"
    test_output = args.test_output or args.output_dir / "test_pattern.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        write_dataset_with_category(args.train_input, train_output),
        write_dataset_with_category(args.test_input, test_output),
    ]

    summary_path = args.output_dir / "category_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "outputs": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.legacy_split:
        split_dataset(args.train_input, args.output_dir)

    print(f"Saved categorized CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
