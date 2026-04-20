from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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


def decimal_places(number_text: str) -> int:
    if "." not in number_text:
        return 0
    return len(number_text.split(".", 1)[1])


def rounded_interval(number_text: str) -> tuple[float, float]:
    value = float(number_text)
    places = decimal_places(number_text)
    half_unit = 0.5 * (10 ** (-places))
    eps = 1e-12
    return value - half_unit - eps, value + half_unit + eps


@dataclass
class MatchResult:
    matched: bool
    matched_families: list[str]
    reason: str


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

            sections.append(format_markdown_table(["行", "入力", "出力"], table_rows))
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


class PatternValidator:
    pattern_name = ""

    def family_names(self) -> list[str]:
        raise NotImplementedError

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        raise NotImplementedError

    def validate_row(self, row: dict[str, str]) -> MatchResult:
        matched_families = [
            family_name
            for family_name in self.family_names()
            if self.matches_family(row, family_name)
        ]
        if matched_families:
            return MatchResult(
                matched=True,
                matched_families=matched_families,
                reason="matched",
            )
        return MatchResult(
            matched=False,
            matched_families=[],
            reason="no_assumed_family_matched_all_examples_and_answer",
        )

    def write_outputs(self, output_root: Path, rows: list[dict[str, str]]) -> dict[str, str]:
        pattern_dir = output_root / self.pattern_name
        pattern_dir.mkdir(parents=True, exist_ok=True)

        unmatched_rows = []
        matched_count = 0
        for row in rows:
            result = self.validate_row(row)
            if result.matched:
                matched_count += 1
                continue
            unmatched_rows.append(
                {
                    "id": row["id"],
                    "pattern": self.pattern_name,
                    "reason": result.reason,
                    "families_checked": "|".join(self.family_names()),
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                }
            )

        unmatched_path = pattern_dir / "unmatched.csv"
        with unmatched_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "pattern",
                    "reason",
                    "families_checked",
                    "prompt",
                    "answer",
                ],
            )
            writer.writeheader()
            writer.writerows(unmatched_rows)

        summary_path = pattern_dir / "unmatched_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "pattern": self.pattern_name,
                    "total_rows": len(rows),
                    "matched_rows": matched_count,
                    "unmatched_rows": len(unmatched_rows),
                    "coverage": matched_count / len(rows) if rows else 0.0,
                    "families_checked": self.family_names(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "pattern": self.pattern_name,
            "total_rows": str(len(rows)),
            "matched_rows": str(matched_count),
            "unmatched_rows": str(len(unmatched_rows)),
            "unmatched_csv": f"{self.pattern_name}/unmatched.csv",
            "summary_json": f"{self.pattern_name}/unmatched_summary.json",
        }
