from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record

_WORDS = (
    "alice",
    "queen",
    "dragon",
    "castle",
    "forest",
    "student",
    "teacher",
    "wizard",
    "rabbit",
    "garden",
    "secret",
    "door",
    "book",
    "river",
    "valley",
    "golden",
    "silver",
    "reads",
    "finds",
    "opens",
    "dreams",
    "creates",
    "follows",
    "inside",
    "near",
)


def _shift_text(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
        else:
            out.append(ch)
    return "".join(out)


def _phrase(rng: random.Random) -> str:
    while True:
        phrase = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 5)))
        if len(phrase) <= 30:
            return phrase


class CipherBuilder:
    category = "cipher"
    generated_categories = ("cipher",)
    description = "encrypted text to plain text"

    def matches(self, prompt: str) -> bool:
        return "secret encryption rules are used on text" in prompt

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        prompt = row.get("prompt", "")
        examples: list[Example] = []
        question = ""
        for line in prompt.splitlines():
            line = line.strip()
            if " -> " in line and not line.startswith("Now"):
                left, right = line.split(" -> ", 1)
                if not re.fullmatch(r"[01]{8}", left.strip()):
                    examples.append(Example(left.strip(), right.strip()))
            match = re.search(r"(?:decrypt|decode|decipher)[^:]*:\s*(.+)", line, re.IGNORECASE)
            if match:
                question = match.group(1).strip()
        return make_record(row, self.category, examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        shift = rng.randint(1, 25)
        plain_examples = [_phrase(rng) for _ in range(5)]
        answer = _phrase(rng)
        examples = [
            Example(_shift_text(plain, shift), plain)
            for plain in plain_examples
        ]
        question = _shift_text(answer, shift)
        lines = [
            "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:",
            *[f"{example.input_value} -> {example.output_value}" for example in examples],
            f"Now, decrypt the following text: {question}",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
