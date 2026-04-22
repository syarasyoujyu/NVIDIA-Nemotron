from __future__ import annotations

import re

from ..base import PatternValidator


class SymbolEquationValidator(PatternValidator):
    pattern_name = "cryptarithm"

    def family_names(self) -> list[str]:
        return []

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        return False

    def _extra_unmatched_fields(self, row: dict[str, str]) -> dict[str, str]:
        # プロンプト中の記号（英数字・空白以外）をユニーク収集
        symbols = sorted(set(re.findall(r'[^\w\s]', row["prompt"])))
        return {"equation_symbols": "|".join(symbols) if symbols else ""}


class SymbolEquationDeduceValidator(SymbolEquationValidator):
    pattern_name = "cryptarithm/deduce"


class SymbolEquationGuessValidator(SymbolEquationValidator):
    pattern_name = "cryptarithm/guess"


VALIDATOR = SymbolEquationValidator()
DEDUCE_VALIDATOR = SymbolEquationDeduceValidator()
GUESS_VALIDATOR = SymbolEquationGuessValidator()
