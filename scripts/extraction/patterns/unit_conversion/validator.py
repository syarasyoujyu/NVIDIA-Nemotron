from __future__ import annotations

import re

from ..base import PatternValidator, rounded_interval


def intersect_intervals(intervals: list[tuple[float, float]]) -> bool:
    low = max(low for low, _ in intervals)
    high = min(high for _, high in intervals)
    return low <= high


class UnitValidator(PatternValidator):
    pattern_name = "unit_conversion"

    def family_names(self) -> list[str]:
        return ["multiplicative_linear_with_rounding"]

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        pairs = re.findall(
            r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)",
            row["prompt"],
        )
        target = re.search(
            r"Now, convert the following measurement: ([0-9]+(?:\.[0-9]+)?) m",
            row["prompt"],
        )
        if target is None:
            return False

        intervals = []
        for input_text, output_text in pairs:
            x = float(input_text)
            if x == 0:
                return False
            y_low, y_high = rounded_interval(output_text)
            intervals.append((float(y_low) / x, float(y_high) / x))

        target_x = float(target.group(1))
        if target_x == 0:
            return False
        ans_low, ans_high = rounded_interval(row["answer"])
        intervals.append((float(ans_low) / target_x, float(ans_high) / target_x))

        return intersect_intervals(intervals)


VALIDATOR = UnitValidator()
