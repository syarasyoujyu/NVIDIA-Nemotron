from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record


def _to_roman(value: int) -> str:
    parts = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for amount, symbol in parts:
        while value >= amount:
            value -= amount
            out.append(symbol)
    return "".join(out)


class NumeralBuilder:
    category = "numeral"
    generated_categories = ("numeral",)
    description = "integer to Roman numeral conversion"

    def matches(self, prompt: str) -> bool:
        return "converted into a different numeral system" in prompt

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        prompt = row.get("prompt", "")
        examples = [
            Example(match.group(1), match.group(2))
            for match in re.finditer(r"(\d+)\s*->\s*([IVXLCDM]+)", prompt)
        ]
        question_match = re.search(r"write the number\s+(\d+)\s+in", prompt, re.IGNORECASE)
        question = question_match.group(1) if question_match else ""
        return make_record(row, self.category, examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        values = rng.sample(range(1, 151), 6)
        example_values = values[:5]
        question_value = values[5]
        examples = [
            Example(str(value), _to_roman(value))
            for value in example_values
        ]
        question = str(question_value)
        answer = _to_roman(question_value)
        lines = [
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
            "Some examples are given below:",
            *[f"{example.input_value} -> {example.output_value}" for example in examples],
            f"Now, write the number {question} in the Wonderland numeral system.",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
