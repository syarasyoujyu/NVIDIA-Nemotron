"""Build categorized raw datasets for pattern inspection and CoT generation.

Outputs:
  - data/generated/train.csv: generated or parsed train rows usable by gen_problems.py
  - data/generated/test.csv: generated or parsed test rows usable as a test CSV
  - data/generated/patterns/train_pattern.csv: generated or parsed train rows plus category
  - data/generated/patterns/test_pattern.csv: generated or parsed test rows plus category
  - data/generated/patterns/train_raw.jsonl: parsed examples/question payloads for CoT tools
  - data/generated/patterns/test_raw.jsonl: parsed examples/question payloads for CoT tools
  - data/generated/patterns/raw_summary.json: counts and unmatched rows

The category and parsing responsibilities live in scripts/gen_data/types/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.basic.const import TEST_CSV, TRAIN_CSV
from scripts.basic.types import GENERATORS
from scripts.cot_prompt.store_types import Problem
from scripts.gen_data.types import RawProblemRecord, build_record, generate_record

DEFAULT_OUTPUT_DIR = (
    Path("data/generated") / datetime.now().strftime("%Y-%m-%d-%H-%M") / "patterns"
)
CATEGORY_ORDER = (
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
)
DEFAULT_TRAIN_COUNTS = {
    "bit_manipulation": 1602,
    "cipher": 1576,
    "cryptarithm_deduce": 329,
    "cryptarithm_guess": 494,
    "equation_numeric_deduce": 596,
    "equation_numeric_guess": 136,
    "gravity": 1597,
    "numeral": 1576,
    "unit_conversion": 1594,
}
DEFAULT_TEST_COUNTS = {
    "bit_manipulation": 2,
    "cipher": 1,
}


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text else ""


def _read_source_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        if "id" not in reader.fieldnames:
            raise ValueError(f"No id column found in {path}")
        if "prompt" not in reader.fieldnames:
            raise ValueError(f"No prompt column found in {path}")
        return list(reader.fieldnames), list(reader)


def _pattern_fieldnames(source_fieldnames: list[str]) -> list[str]:
    fields = list(source_fieldnames)
    if "category" not in fields:
        fields.append("category")
    return fields


def _parse_counts(value: str | None, default: dict[str, int]) -> dict[str, int]:
    if value is None or not value.strip():
        return dict(default)
    text = value.strip()
    if text.startswith("["):
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("Count list must be a JSON list")
        if len(raw) != len(CATEGORY_ORDER):
            raise ValueError(
                f"Count list must have {len(CATEGORY_ORDER)} values in category order: "
                f"{','.join(CATEGORY_ORDER)}"
            )
        return {category: int(count) for category, count in zip(CATEGORY_ORDER, raw)}
    if text.startswith("{"):
        raw = json.loads(text)
        return {category: int(raw.get(category, 0)) for category in CATEGORY_ORDER}

    counts = {category: 0 for category in CATEGORY_ORDER}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Counts must be a JSON list, JSON object, or comma-separated category=count pairs"
            )
        category, count = part.split("=", 1)
        category = category.strip()
        if category not in counts:
            raise ValueError(f"Unknown category in counts: {category}")
        counts[category] = int(count)
    return counts


def _new_problem_id(rng: random.Random, used_ids: set[str]) -> str:
    while True:
        problem_id = f"{rng.getrandbits(32):08x}"
        if problem_id not in used_ids:
            used_ids.add(problem_id)
            return problem_id


def _extract_answer(reasoning_text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return ""


def _compare_answer(expected: str, predicted: str) -> bool:
    expected = expected.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", expected):
        return predicted.lower() == expected.lower()
    try:
        return math.isclose(float(expected), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == expected.lower()


def _verify_cot_records(
    records: list[RawProblemRecord],
    enabled: bool,
    max_examples: int = 20,
) -> dict[str, object]:
    if not enabled:
        return {"enabled": False}

    checked = 0
    correct = 0
    skipped = 0
    mismatches: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    by_category: dict[str, dict[str, int]] = {}

    for record in records:
        bucket = by_category.setdefault(
            record.category,
            {"checked": 0, "correct": 0, "incorrect": 0, "errors": 0, "skipped": 0},
        )
        if not record.answer:
            skipped += 1
            bucket["skipped"] += 1
            continue
        generator = GENERATORS.get(record.category)
        if generator is None:
            skipped += 1
            bucket["skipped"] += 1
            if len(errors) < max_examples:
                errors.append(
                    {
                        "id": record.id,
                        "category": record.category,
                        "error": "no cot_prompt generator",
                    }
                )
            continue

        checked += 1
        bucket["checked"] += 1
        problem = Problem(
            id=record.id,
            category=record.category,  # type: ignore[arg-type]
            examples=record.examples,
            question=record.question,
            answer=record.answer,
            prompt=record.prompt,
        )
        try:
            reasoning = generator(problem)
            if reasoning is None:
                submission = ""
            else:
                submission = _extract_answer(reasoning)
            is_correct = _compare_answer(record.answer, submission)
        except Exception as exc:
            bucket["errors"] += 1
            if len(errors) < max_examples:
                errors.append(
                    {
                        "id": record.id,
                        "category": record.category,
                        "error": str(exc),
                    }
                )
            continue

        if is_correct:
            correct += 1
            bucket["correct"] += 1
        else:
            bucket["incorrect"] += 1
            if len(mismatches) < max_examples:
                mismatches.append(
                    {
                        "id": record.id,
                        "category": record.category,
                        "answer": record.answer,
                        "submission": submission,
                    }
                )

    incorrect = checked - correct - sum(bucket["errors"] for bucket in by_category.values())
    return {
        "enabled": True,
        "checked": checked,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "errors": sum(bucket["errors"] for bucket in by_category.values()),
        "accuracy": correct / checked if checked else 0.0,
        "by_category": dict(sorted(by_category.items())),
        "mismatches": mismatches,
        "error_examples": errors,
    }


def _verify_cot_record(record: RawProblemRecord) -> tuple[bool, str]:
    if not record.answer:
        return False, ""
    generator = GENERATORS.get(record.category)
    if generator is None:
        return False, ""
    problem = Problem(
        id=record.id,
        category=record.category,  # type: ignore[arg-type]
        examples=record.examples,
        question=record.question,
        answer=record.answer,
        prompt=record.prompt,
    )
    reasoning = generator(problem)
    submission = _extract_answer(reasoning) if reasoning is not None else ""
    return _compare_answer(record.answer, submission), submission


def _write_pattern_csv(
    path: Path,
    records: list[RawProblemRecord],
    source_fieldnames: list[str],
    include_answer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _pattern_fieldnames(source_fieldnames)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.pattern_row(fieldnames, include_answer=include_answer))


def _write_raw_jsonl(
    path: Path,
    records: list[RawProblemRecord],
    include_answer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.raw_payload(include_answer=include_answer), ensure_ascii=False))
            f.write("\n")


def _write_dataset_csv(
    path: Path,
    records: list[RawProblemRecord],
    include_answer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "prompt"]
    if include_answer:
        fieldnames.append("answer")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"id": record.id, "prompt": record.prompt}
            if include_answer:
                row["answer"] = record.answer
            writer.writerow(row)


def _build_records(
    rows: list[dict[str, str]],
) -> tuple[list[RawProblemRecord], list[dict[str, str]]]:
    records: list[RawProblemRecord] = []
    unmatched: list[dict[str, str]] = []
    for row in rows:
        try:
            records.append(build_record(row))
        except ValueError:
            unmatched.append(
                {
                    "id": row.get("id", ""),
                    "prompt_first_line": _first_line(row.get("prompt", "")),
                }
            )
    return records, unmatched


def _summarize(
    source_path: Path | str,
    pattern_path: Path,
    raw_path: Path,
    records: list[RawProblemRecord],
    unmatched: list[dict[str, str]],
    cot_verification: dict[str, object],
) -> dict[str, object]:
    category_counts = Counter(record.category for record in records)
    return {
        "input_file": str(source_path),
        "pattern_file": str(pattern_path),
        "raw_file": str(raw_path),
        "total_rows": len(records) + len(unmatched),
        "matched_rows": len(records),
        "unmatched_count": len(unmatched),
        "category_counts": dict(sorted(category_counts.items())),
        "unmatched_ids": [row["id"] for row in unmatched],
        "unmatched_rows": unmatched,
        "cot_verification": cot_verification,
    }


def build_dataset(
    source_path: Path,
    dataset_path: Path,
    pattern_path: Path,
    raw_path: Path,
    include_answer: bool,
    verify_cot: bool,
) -> dict[str, object]:
    source_fieldnames, rows = _read_source_csv(source_path)
    records, unmatched = _build_records(rows)
    _write_dataset_csv(dataset_path, records, include_answer=include_answer)
    _write_pattern_csv(
        pattern_path,
        records,
        source_fieldnames=source_fieldnames,
        include_answer=include_answer,
    )
    _write_raw_jsonl(raw_path, records, include_answer=include_answer)
    cot_verification = _verify_cot_records(records, enabled=verify_cot and include_answer)
    return _summarize(source_path, pattern_path, raw_path, records, unmatched, cot_verification)


def build_generated_dataset(
    source_name: str,
    dataset_path: Path,
    pattern_path: Path,
    raw_path: Path,
    counts: dict[str, int],
    include_answer: bool,
    seed: int,
    verify_cot: bool,
    require_cot_correct: bool,
    max_attempts_per_record: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    used_ids: set[str] = set()
    records: list[RawProblemRecord] = []
    for category in CATEGORY_ORDER:
        for _ in range(counts.get(category, 0)):
            last_record: RawProblemRecord | None = None
            for attempt in range(1, max_attempts_per_record + 1):
                problem_id = _new_problem_id(rng, used_ids)
                record = generate_record(category, rng, problem_id)
                last_record = record
                if not require_cot_correct:
                    records.append(record)
                    break
                is_correct, _ = _verify_cot_record(record)
                if is_correct:
                    records.append(record)
                    break
            else:
                assert last_record is not None
                raise RuntimeError(
                    f"Could not generate CoT-verifiable record for {category} "
                    f"after {max_attempts_per_record} attempts"
                )
    rng.shuffle(records)

    source_fieldnames = ["id", "prompt", "answer"] if include_answer else ["id", "prompt"]
    _write_dataset_csv(dataset_path, records, include_answer=include_answer)
    _write_pattern_csv(
        pattern_path,
        records,
        source_fieldnames=source_fieldnames,
        include_answer=include_answer,
    )
    _write_raw_jsonl(raw_path, records, include_answer=include_answer)
    cot_verification = _verify_cot_records(records, enabled=verify_cot)
    return _summarize(source_name, pattern_path, raw_path, records, [], cot_verification)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cot_prompt-compatible raw pattern datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "parse"],
        default="generate",
        help="generate creates synthetic rows from scratch; parse categorizes existing CSV files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-counts",
        default=None,
        help=(
            "Generated train counts. Use JSON or comma-separated category=count pairs. "
            "Defaults to the original category distribution."
        ),
    )
    parser.add_argument(
        "--test-counts",
        default=None,
        help="Generated test counts. Defaults to bit_manipulation=2,cipher=1.",
    )
    parser.add_argument("--train-input", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test-input", type=Path, default=TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-csv-output", type=Path, default=None)
    parser.add_argument("--test-csv-output", type=Path, default=None)
    parser.add_argument("--train-pattern-output", type=Path, default=None)
    parser.add_argument("--test-pattern-output", type=Path, default=None)
    parser.add_argument("--train-raw-output", type=Path, default=None)
    parser.add_argument("--test-raw-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--verify-cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run scripts/cot_prompt generators after building rows and compare submissions with answers.",
    )
    parser.add_argument(
        "--fail-on-cot-mismatch",
        action="store_true",
        help="Exit with an error if CoT verification finds mismatches or generator errors.",
    )
    parser.add_argument(
        "--require-cot-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When generating rows, keep retrying until scripts/cot_prompt produces the stored answer.",
    )
    parser.add_argument(
        "--max-attempts-per-record",
        type=int,
        default=100,
        help="Maximum retries for one generated row when --require-cot-correct is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pattern = args.train_pattern_output or output_dir / "train_pattern.csv"
    test_pattern = args.test_pattern_output or output_dir / "test_pattern.csv"
    train_raw = args.train_raw_output or output_dir / "train_raw.jsonl"
    test_raw = args.test_raw_output or output_dir / "test_raw.jsonl"
    summary_output = args.summary_output or output_dir / "raw_summary.json"
    train_csv = args.train_csv_output or output_dir.parent / "train.csv"
    test_csv = args.test_csv_output or output_dir.parent / "test.csv"

    if args.mode == "generate":
        train_counts = _parse_counts(args.train_counts, DEFAULT_TRAIN_COUNTS)
        test_counts = _parse_counts(args.test_counts, DEFAULT_TEST_COUNTS)
        summaries = [
            build_generated_dataset(
                "generated:train",
                train_csv,
                train_pattern,
                train_raw,
                counts=train_counts,
                include_answer=True,
                seed=args.seed,
                verify_cot=args.verify_cot,
                require_cot_correct=args.require_cot_correct,
                max_attempts_per_record=args.max_attempts_per_record,
            ),
            build_generated_dataset(
                "generated:test",
                test_csv,
                test_pattern,
                test_raw,
                counts=test_counts,
                include_answer=True,
                seed=args.seed + 1,
                verify_cot=args.verify_cot,
                require_cot_correct=args.require_cot_correct,
                max_attempts_per_record=args.max_attempts_per_record,
            ),
        ]
    else:
        summaries = [
            build_dataset(
                args.train_input,
                train_csv,
                train_pattern,
                train_raw,
                include_answer=True,
                verify_cot=args.verify_cot,
            ),
            build_dataset(
                args.test_input,
                test_csv,
                test_pattern,
                test_raw,
                include_answer=False,
                verify_cot=args.verify_cot,
            ),
        ]
    summary_output.write_text(
        json.dumps({"outputs": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved raw pattern data to {output_dir}")
    for summary in summaries:
        print(
            f"  {summary['pattern_file']}: "
            f"{summary['matched_rows']}/{summary['total_rows']} rows, "
            f"{summary['unmatched_count']} unmatched"
        )
        cot = summary.get("cot_verification", {})
        if isinstance(cot, dict) and cot.get("enabled"):
            print(
                "    CoT verification: "
                f"{cot.get('correct', 0)}/{cot.get('checked', 0)} correct, "
                f"{cot.get('incorrect', 0)} incorrect, "
                f"{cot.get('errors', 0)} errors"
            )

    if args.fail_on_cot_mismatch:
        failing = []
        for summary in summaries:
            cot = summary.get("cot_verification", {})
            if not isinstance(cot, dict) or not cot.get("enabled"):
                continue
            if int(cot.get("incorrect", 0)) > 0 or int(cot.get("errors", 0)) > 0:
                failing.append(summary["pattern_file"])
        if failing:
            raise SystemExit(
                "CoT verification failed for: " + ", ".join(str(path) for path in failing)
            )


if __name__ == "__main__":
    main()
