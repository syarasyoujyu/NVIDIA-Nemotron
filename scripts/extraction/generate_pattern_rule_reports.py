#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extraction.patterns import get_extractors, get_validators
from scripts.extraction.patterns.base import format_markdown_table, load_rows


def generate_reports(input_dir: Path, output_dir: Path, sample_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for extractor in get_extractors():
        source_path = input_dir / extractor.pattern_name / "rows.csv"
        rows = load_rows(source_path)
        parsed_records = [extractor.parse_row(row) for row in rows]
        index_rows.append(
            extractor.write_outputs(output_dir, parsed_records, sample_size)
        )

    index_md = ["# パターン別ルールレポート", ""]
    index_md.append(
        format_markdown_table(
            ["パターン", "件数", "ディレクトリ", "csv", "jsonl", "markdown"],
            [
                [
                    row["pattern"],
                    row["count"],
                    row["directory"],
                    row["csv"],
                    row["jsonl"],
                    row["markdown"],
                ]
                for row in index_rows
            ],
        )
    )
    index_md.append("")
    (output_dir / "README.md").write_text("\n".join(index_md), encoding="utf-8")


def generate_unmatched(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for validator in get_validators():
        source_path = input_dir / validator.pattern_name / "rows.csv"
        rows = load_rows(source_path)
        index_rows.append(validator.write_outputs(output_dir, rows))

    readme_lines = ["# unmatched 一覧", ""]
    readme_lines.append(
        format_markdown_table(
            ["パターン", "総件数", "一致件数", "未一致件数", "unmatched_csv", "summary_json"],
            [
                [
                    row["pattern"],
                    row["total_rows"],
                    row["matched_rows"],
                    row["unmatched_rows"],
                    row["unmatched_csv"],
                    row["summary_json"],
                ]
                for row in index_rows
            ],
        )
    )
    readme_lines.append("")
    (output_dir / "README_unmatched.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="パターン別のレポート生成と unmatched 生成をまとめて行う。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/patterns"),
        help="パターンごとの rows.csv を置いたディレクトリ。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/patterns"),
        help="レポート出力先ディレクトリ。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="各 Markdown レポートに含めるサンプル件数。",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "reports", "unmatched"],
        default="all",
        help="`all` は report と unmatched を両方生成する。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ran = []
    if args.mode in {"all", "reports"}:
        generate_reports(args.input_dir, args.output_dir, args.sample_size)
        ran.append("reports")
    if args.mode in {"all", "unmatched"}:
        generate_unmatched(args.input_dir, args.output_dir)
        ran.append("unmatched")
    print(f"Saved {', '.join(ran)} to {args.output_dir}")


if __name__ == "__main__":
    main()
