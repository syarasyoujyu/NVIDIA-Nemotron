from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record


class GravityBuilder:
    category = "gravity"
    generated_categories = ("gravity",)
    description = "falling distance from modified gravitational constant"

    def matches(self, prompt: str) -> bool:
        return "gravitational constant has been secretly changed" in prompt

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        prompt = row.get("prompt", "")
        examples = [
            Example(match.group(1), match.group(2))
            for match in re.finditer(r"t\s*=\s*([\d.]+)s,\s*distance\s*=\s*([\d.]+)", prompt)
        ]
        question_line = next((line for line in prompt.splitlines() if line.startswith("Now")), "")
        question_match = re.search(r"t\s*=\s*([\d.]+)s", question_line)
        question = question_match.group(1) if question_match else ""
        return make_record(row, self.category, examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        g = rng.choice([7.2, 8.4, 9.8, 12.0, 15.6, 18.0])
        times = [round(rng.uniform(0.8, 6.0), 2) for _ in range(7)]

        def dist(t: float) -> str:
            return f"{0.5 * g * t * t:.2f}"

        def fmt(t: float) -> str:
            return f"{t:.2f}"

        examples = [
            Example(fmt(t), dist(t))
            for t in times[:5]
        ]
        question = fmt(times[5])
        answer = dist(times[5])
        lines = [
            "In Alice's Wonderland, the gravitational constant has been secretly changed. "
            "Here are some example observations:",
            *[
                f"For t = {example.input_value}s, distance = {example.output_value} m"
                for example in examples
            ],
            f"Now, determine the falling distance for t = {question}s given d = 0.5*g*t^2.",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
