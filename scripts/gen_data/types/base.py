from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from scripts.cot_prompt.store_types import Example


@dataclass(frozen=True)
class RawProblemRecord:
    """A source row enriched for both CSV grouping and CoT prompt generation."""

    id: str
    prompt: str
    answer: str
    category: str
    examples: list[Example]
    question: str

    def pattern_row(self, fieldnames: list[str], include_answer: bool) -> dict[str, str]:
        row = {
            "id": self.id,
            "prompt": self.prompt,
            "category": self.category,
        }
        if include_answer:
            row["answer"] = self.answer
        return {field: row.get(field, "") for field in fieldnames}

    def raw_payload(self, include_answer: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "examples": [example.to_payload() for example in self.examples],
            "question": self.question,
        }
        if include_answer:
            payload["answer"] = self.answer
        return payload


class TypeDataBuilder(Protocol):
    category: str
    description: str
    generated_categories: tuple[str, ...]

    def matches(self, prompt: str) -> bool:
        ...

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        ...

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        ...


def make_record(
    row: dict[str, str],
    category: str,
    examples: list[Example],
    question: str,
) -> RawProblemRecord:
    return RawProblemRecord(
        id=row.get("id", ""),
        prompt=row.get("prompt", ""),
        answer=row.get("answer", ""),
        category=category,
        examples=examples,
        question=question,
    )


def make_generated_record(
    problem_id: str,
    category: str,
    prompt: str,
    answer: str,
    examples: list[Example],
    question: str,
) -> RawProblemRecord:
    return RawProblemRecord(
        id=problem_id,
        prompt=prompt,
        answer=answer,
        category=category,
        examples=examples,
        question=question,
    )
