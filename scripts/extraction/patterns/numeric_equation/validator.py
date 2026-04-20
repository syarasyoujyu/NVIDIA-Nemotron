from __future__ import annotations

from ..base import PatternValidator


class NumericEquationValidator(PatternValidator):
    pattern_name = "numeric_equation"

    def family_names(self) -> list[str]:
        return []

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        return False


VALIDATOR = NumericEquationValidator()
