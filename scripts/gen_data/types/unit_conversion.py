from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record


class UnitConversionBuilder:
    category = "unit_conversion"
    generated_categories = ("unit_conversion",)
    description = "secret unit conversion on measurements"

    def matches(self, prompt: str) -> bool:
        return "secret unit conversion is applied to measurements" in prompt

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        prompt = row.get("prompt", "")
        examples = [
            Example(match.group(1), match.group(2))
            for match in re.finditer(r"([\d.]+)\s+\S+\s+becomes\s+([\d.]+)", prompt)
        ]
        question_match = re.search(r"convert the following measurement:\s*([\d.]+)", prompt)
        question = question_match.group(1) if question_match else ""
        return make_record(row, self.category, examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        factor = rng.choice([0.25, 0.4, 0.5, 0.66, 0.75, 1.2, 1.5, 2.5])
        values = [round(rng.uniform(3, 80), 2) for _ in range(7)]

        def fmt(value: float) -> str:
            return f"{value:.2f}"

        examples = [
            Example(fmt(value), fmt(value * factor))
            for value in values[:5]
        ]
        question = fmt(values[5])
        answer = fmt(values[5] * factor)
        unit = "m"
        lines = [
            "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:",
            *[f"{example.input_value} {unit} becomes {example.output_value}" for example in examples],
            f"Now, convert the following measurement: {question} {unit}",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
