from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from ..base import PatternExtractor


def decimal_places(number_text: str) -> int:
    if "." not in number_text:
        return 0
    return len(number_text.split(".", 1)[1])


def rounded_interval(number_text: str) -> tuple[Decimal, Decimal]:
    value = Decimal(number_text)
    half_unit = Decimal("0.5") * (Decimal(10) ** (-decimal_places(number_text)))
    eps = Decimal("1e-18")
    return value - half_unit - eps, value + half_unit + eps


def quantize_string(value: Decimal, places: int = 2) -> str:
    quantum = Decimal("1").scaleb(-places)
    return str(value.quantize(quantum, rounding=ROUND_HALF_UP))


def interval_midpoint(low: Decimal, high: Decimal) -> Decimal:
    return (low + high) / 2


class GravityPromptExtractor(PatternExtractor):
    pattern_name = "gravity_distance"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        examples = [
            {"input": left, "output": right}
            for left, right in re.findall(
                r"For t = ([0-9]+(?:\.[0-9]+)?)s, distance = ([0-9]+(?:\.[0-9]+)?) m",
                prompt,
            )
        ]
        target = re.search(
            r"Now, determine the falling distance for t = ([0-9]+(?:\.[0-9]+)?)s",
            prompt,
        )
        target_time_text = target.group(1) if target else "0"
        target_time = Decimal(target_time_text)

        pairs = [(Decimal(pair["input"]), Decimal(pair["output"])) for pair in examples]
        gravity_intervals = []
        for pair in examples:
            t = Decimal(pair["input"])
            d_low, d_high = rounded_interval(pair["output"])
            denom = Decimal("0.5") * t * t
            gravity_intervals.append((d_low / denom, d_high / denom))

        gravity_low = max(low for low, _ in gravity_intervals)
        gravity_high = min(high for _, high in gravity_intervals)

        answer_low, answer_high = rounded_interval(answer)
        target_denom = Decimal("0.5") * target_time * target_time
        target_gravity_low = answer_low / target_denom
        target_gravity_high = answer_high / target_denom
        consistent_low = max(gravity_low, target_gravity_low)
        consistent_high = min(gravity_high, target_gravity_high)
        target_consistent = consistent_low <= consistent_high

        gravity = (
            interval_midpoint(consistent_low, consistent_high)
            if target_consistent
            else interval_midpoint(gravity_low, gravity_high)
        )
        predicted = quantize_string(Decimal("0.5") * gravity * target_time * target_time)

        diagram_lines = [
            f"t --square--> t^2 --* {float(Decimal('0.5') * gravity):.6f} --> d",
            f"estimated g = {float(gravity):.6f}",
        ]
        for t, d in pairs[:6]:
            diagram_lines.append(
                f"{float(t):.2f}s --square--> {float(t*t):.4f} --* {float(Decimal('0.5') * gravity):.6f} --> {float(d):.2f}m"
            )
        diagram_lines.append(
            f"{float(target_time):.2f}s --square--> {float(target_time*target_time):.4f} --* {float(Decimal('0.5') * gravity):.6f} --> {answer}m  [対象]"
        )

        analysis_rows = []
        for t, d in pairs:
            analysis_rows.append(
                {
                    "time_s": float(t),
                    "distance_m": float(d),
                    "estimated_g": round(float(Decimal(2) * d / (t * t)), 6) if t else None,
                }
            )

        return {
            "examples": examples,
            "target_input": f"{float(target_time):.2f}",
            "answer": answer,
            "rule_summary": (
                f"距離は d = 0.5 * g * t^2 に従います。"
                f"例題すべてと両立する g の区間は {float(gravity_low):.6f} から {float(gravity_high):.6f} で、"
                f"代表値として {float(gravity):.6f} を使っています。"
                f"この代表値でターゲット解答まで一致するかは {predicted == answer} です。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": {
                "gravity": float(gravity),
                "gravity_interval": [float(gravity_low), float(gravity_high)],
                "target_gravity_interval": [
                    float(target_gravity_low),
                    float(target_gravity_high),
                ],
                "gravity_interval_with_target": [
                    float(consistent_low),
                    float(consistent_high),
                ]
                if target_consistent
                else None,
                "target_consistent_with_representative": predicted == answer,
                "target_consistent_interval_exists": target_consistent,
                "predicted_target_output": predicted,
                "rows": analysis_rows,
            },
        }


EXTRACTOR = GravityPromptExtractor()
