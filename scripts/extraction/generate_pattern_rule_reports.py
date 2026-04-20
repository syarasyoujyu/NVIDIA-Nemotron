#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.extraction.extract_pattern import get_extractors
from scripts.extraction.extract_pattern.base import format_markdown_table, load_rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="パターン別のルールレポートと解析済み入出力データを生成する。"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_reports(args.input_dir, args.output_dir, args.sample_size)
    print(f"Saved pattern reports to {args.output_dir}")


if __name__ == "__main__":
    main()
