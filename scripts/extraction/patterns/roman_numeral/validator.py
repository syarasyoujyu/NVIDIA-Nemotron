from __future__ import annotations

import re

from ..base import PatternValidator


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


class RomanValidator(PatternValidator):
    pattern_name = "roman_numeral"

    def family_names(self) -> list[str]:
        return ["standard_roman_numeral"]

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        examples = re.findall(r"(\d+) -> ([IVXLCDM]+)", row["prompt"])
        for number_text, roman in examples:
            if to_roman(int(number_text)) != roman:
                return False

        target = re.search(
            r"Now, write the number (\d+) in the Wonderland numeral system\.",
            row["prompt"],
        )
        if target is None:
            return False
        return to_roman(int(target.group(1))) == row["answer"]


VALIDATOR = RomanValidator()
