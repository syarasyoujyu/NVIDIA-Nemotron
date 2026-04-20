from __future__ import annotations

import re

from ..base import PatternValidator, rounded_interval


def intersect_intervals(intervals: list[tuple[float, float]]) -> bool:
    low = max(low for low, _ in intervals)
    high = min(high for _, high in intervals)
    return low <= high


class GravityValidator(PatternValidator):
    pattern_name = "gravity_distance"

    def family_names(self) -> list[str]:
        return ["constant_g_in_d_equals_half_g_t_squared"]

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        pairs = re.findall(
            r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m",
            row["prompt"],
        )
        target = re.search(
            r"Now, determine the falling distance for t = ([0-9]+(?:\.[0-9]+)?)s",
            row["prompt"],
        )
        if target is None:
            return False

        intervals = []
        for time_text, dist_text in pairs:
            t = float(time_text)
            if t == 0:
                return False
            d_low, d_high = rounded_interval(dist_text)
            denom = 0.5 * t * t
            intervals.append((float(d_low) / denom, float(d_high) / denom))

        target_t = float(target.group(1))
        if target_t == 0:
            return False
        ans_low, ans_high = rounded_interval(row["answer"])
        denom = 0.5 * target_t * target_t
        intervals.append((float(ans_low) / denom, float(ans_high) / denom))

        return intersect_intervals(intervals)


VALIDATOR = GravityValidator()
