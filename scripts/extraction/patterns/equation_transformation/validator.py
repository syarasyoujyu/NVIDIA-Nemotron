from __future__ import annotations

from collections import Counter

from ..base import PatternValidator


def is_subsequence(source: str, target: str) -> bool:
    idx = 0
    for char in source:
        if idx < len(target) and char == target[idx]:
            idx += 1
    return idx == len(target)


def multiset_subset(source: str, target: str) -> bool:
    return not (Counter(target) - Counter(source))


class EquationValidator(PatternValidator):
    pattern_name = "equation_transformation"

    def family_names(self) -> list[str]:
        return [
            "subsequence_selection",
            "multiset_subset_rewrite",
        ]

    def _parse_pairs(self, row: dict[str, str]) -> list[tuple[str, str]]:
        pairs = []
        target_input = None
        for line in row["prompt"].splitlines():
            if " = " in line:
                left, right = line.split(" = ", 1)
                pairs.append((left, right))
            elif line.startswith("Now, determine the result for: "):
                target_input = line.split(": ", 1)[1]
        if target_input is None:
            return []
        pairs.append((target_input, row["answer"]))
        return pairs

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        pairs = self._parse_pairs(row)
        if not pairs:
            return False

        if family_name == "subsequence_selection":
            return all(is_subsequence(source, target) for source, target in pairs)
        if family_name == "multiset_subset_rewrite":
            return all(multiset_subset(source, target) for source, target in pairs)
        return False


VALIDATOR = EquationValidator()
