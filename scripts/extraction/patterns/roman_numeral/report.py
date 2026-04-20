from __future__ import annotations

import re

from ..base import PatternExtractor


ROMAN_TOKENS = [
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


def to_roman(value: int) -> str:
    result = []
    remaining = value
    for token_value, token in ROMAN_TOKENS:
        while remaining >= token_value:
            result.append(token)
            remaining -= token_value
    return "".join(result)


def decompose_roman(value: int) -> list[str]:
    tokens = []
    remaining = value
    for token_value, token in ROMAN_TOKENS:
        while remaining >= token_value:
            tokens.append(token)
            remaining -= token_value
    return tokens


class RomanPromptExtractor(PatternExtractor):
    pattern_name = "roman_numeral"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        examples = [
            {"input": left, "output": right}
            for left, right in re.findall(r"(\d+) -> ([IVXLCDM]+)", prompt)
        ]
        target = re.search(
            r"Now, write the number (\d+) in the Wonderland numeral system\.", prompt
        )
        target_value = int(target.group(1)) if target else 0
        tokens = decompose_roman(target_value)

        diagram = [f"{pair['input']} -> {pair['output']}" for pair in examples]
        diagram.append(f"{target_value} -> {' + '.join(tokens)} -> {answer}  [対象]")

        return {
            "examples": examples,
            "target_input": str(target_value),
            "answer": answer,
            "rule_summary": (
                "標準的なローマ数字への変換です。ターゲット値については、"
                "どのローマ数字トークンの組み合わせで構成されるかも示しています。"
            ),
            "relation_diagram": "\n".join(diagram),
            "analysis": {
                "target_tokens": tokens,
                "recomputed_answer": to_roman(target_value),
            },
        }


EXTRACTOR = RomanPromptExtractor()
