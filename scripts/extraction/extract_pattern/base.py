from __future__ import annotations

import csv
import json
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    escaped_rows = []
    for row in rows:
        escaped_rows.append(
            [str(cell).replace("|", "\\|").replace("\n", "<br>") for cell in row]
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in escaped_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class PatternExtractor:
    pattern_name = ""

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        raise NotImplementedError

    def parse_row(self, row: dict[str, str]) -> dict:
        parsed = self.parse_prompt(row["prompt"], row["answer"])
        return {
            "id": row["id"],
            "pattern": self.pattern_name,
            **parsed,
        }

    def build_sample_section(self, records: list[dict], sample_size: int) -> str:
        sections = [f"# {self.pattern_name}", "", f"総件数: {len(records)}", ""]

        for record in records[:sample_size]:
            sections.append(f"## {record['id']}")
            sections.append("")
            sections.append("### 規則の要約")
            sections.append(record["rule_summary"])
            sections.append("")
            sections.append("### 入出力表")

            table_rows = []
            for idx, pair in enumerate(record["examples"], start=1):
                table_rows.append([str(idx), pair["input"], pair["output"]])
            table_rows.append(["target", record["target_input"], record["answer"]])

            sections.append(
                format_markdown_table(["行", "入力", "出力"], table_rows)
            )
            sections.append("")
            sections.append("### 関係図")
            sections.append("```text")
            sections.append(record["relation_diagram"])
            sections.append("```")
            sections.append("")

            analysis = record.get("analysis")
            if analysis is not None:
                sections.append("### 補足情報")
                sections.append("```json")
                sections.append(json.dumps(analysis, ensure_ascii=False, indent=2))
                sections.append("```")
                sections.append("")

        return "\n".join(sections).rstrip() + "\n"

    def write_outputs(
        self, output_root: Path, records: list[dict], sample_size: int
    ) -> dict[str, str]:
        pattern_dir = output_root / self.pattern_name
        pattern_dir.mkdir(parents=True, exist_ok=True)

        csv_path = pattern_dir / "rows.csv"
        jsonl_path = pattern_dir / "parsed.jsonl"
        readme_path = pattern_dir / "README.md"

        write_jsonl(jsonl_path, records)
        readme_path.write_text(
            self.build_sample_section(records, sample_size),
            encoding="utf-8",
        )

        return {
            "pattern": self.pattern_name,
            "count": str(len(records)),
            "directory": self.pattern_name,
            "csv": f"{self.pattern_name}/rows.csv",
            "jsonl": f"{self.pattern_name}/parsed.jsonl",
            "markdown": f"{self.pattern_name}/README.md",
        }
