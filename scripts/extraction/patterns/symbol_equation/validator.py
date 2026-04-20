from __future__ import annotations

from ..base import PatternValidator


class SymbolEquationValidator(PatternValidator):
    pattern_name = "symbol_equation"

    def family_names(self) -> list[str]:
        return []

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        return False


class SymbolEquationDeduceValidator(SymbolEquationValidator):
    pattern_name = "symbol_equation/deduce"


class SymbolEquationGuessValidator(SymbolEquationValidator):
    pattern_name = "symbol_equation/guess"


VALIDATOR = SymbolEquationValidator()
DEDUCE_VALIDATOR = SymbolEquationDeduceValidator()
GUESS_VALIDATOR = SymbolEquationGuessValidator()
