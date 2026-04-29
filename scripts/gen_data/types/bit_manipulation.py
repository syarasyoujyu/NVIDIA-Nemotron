from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record


class BitManipulationBuilder:
    category = "bit_manipulation"
    generated_categories = ("bit_manipulation",)
    description = "8-bit binary transformation"

    def matches(self, prompt: str) -> bool:
        return "secret bit manipulation rule transforms 8-bit binary numbers" in prompt

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        prompt = row.get("prompt", "")
        examples = [
            Example(match.group(1), match.group(2))
            for match in re.finditer(r"([01]{8}) -> ([01]{8})", prompt)
        ]
        question_match = re.search(r"determine the output for:\s*([01]{8})", prompt)
        question = question_match.group(1) if question_match else ""
        return make_record(row, self.category, examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        rule = rng.choice(("not", "rotl", "rotr", "xor"))
        xor_mask = rng.randrange(1, 256)

        def bits(value: int) -> str:
            return format(value & 0xFF, "08b")

        def transform(value: int) -> int:
            if rule == "not":
                return value ^ 0xFF
            if rule == "rotl":
                return ((value << 1) & 0xFF) | (value >> 7)
            if rule == "rotr":
                return (value >> 1) | ((value & 1) << 7)
            return value ^ xor_mask

        values = rng.sample(range(256), 10)
        example_values = values[:8]
        question_value = values[8]
        examples = [
            Example(bits(value), bits(transform(value)))
            for value in example_values
        ]
        question = bits(question_value)
        answer = bits(transform(question_value))
        lines = [
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
            "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
            "and possibly majority or choice functions.",
            "",
            "Here are some examples of input -> output:",
            *[f"{example.input_value} -> {example.output_value}" for example in examples],
            "",
            f"Now, determine the output for: {question}",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
