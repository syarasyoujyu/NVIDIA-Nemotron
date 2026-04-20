#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


Rule = tuple[str, str, Callable[[str], bool]]


def first_line(text: str) -> str:
    return text.splitlines()[0].strip()


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
        "equation_transformation",
        "symbol-string transformation rules",
        lambda prompt: "secret set of transformation rules is applied to equations"
        in prompt,
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
