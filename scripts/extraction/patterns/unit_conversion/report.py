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


class UnitPromptExtractor(PatternExtractor):
    pattern_name = "unit_conversion"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        examples = [
            {"input": left, "output": right}
            for left, right in re.findall(
                r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)", prompt
            )
        ]
        target = re.search(
            r"Now, convert the following measurement: ([0-9]+(?:\.[0-9]+)?) m", prompt
        )
        target_value_text = target.group(1) if target else "0"
        target_value = Decimal(target_value_text)

        pairs = [(Decimal(pair["input"]), Decimal(pair["output"])) for pair in examples]
        factor_intervals = []
        for pair in examples:
            x = Decimal(pair["input"])
            y_low, y_high = rounded_interval(pair["output"])
            factor_intervals.append((y_low / x, y_high / x))

        factor_low = max(low for low, _ in factor_intervals)
        factor_high = min(high for _, high in factor_intervals)

        answer_low, answer_high = rounded_interval(answer)
        target_factor_low = answer_low / target_value
        target_factor_high = answer_high / target_value
        consistent_low = max(factor_low, target_factor_low)
        consistent_high = min(factor_high, target_factor_high)
        target_consistent = consistent_low <= consistent_high

        factor = (
            interval_midpoint(consistent_low, consistent_high)
            if target_consistent
            else interval_midpoint(factor_low, factor_high)
        )
        predicted = quantize_string(target_value * factor)

        diagram_lines = [f"x --* {float(factor):.6f} --> y"]
        for x, y in pairs[:6]:
            diagram_lines.append(f"{float(x):.2f} --* {float(factor):.6f} --> {float(y):.2f}")
        diagram_lines.append(
            f"{float(target_value):.2f} --* {float(factor):.6f} --> {answer}  [対象]"
        )

        analysis_rows = []
        for x, y in pairs:
            analysis_rows.append(
                {
                    "input_m": float(x),
                    "output": float(y),
                    "estimated_output": quantize_string(x * factor),
                    "ratio": round(float(y / x), 6) if x else None,
                }
            )

        return {
            "examples": examples,
            "target_input": f"{float(target_value):.2f}",
            "answer": answer,
            "rule_summary": (
                f"各行ごとにほぼ一定の倍率で変換される線形変換です。"
                f"例題すべてと両立する倍率区間は {float(factor_low):.6f} から {float(factor_high):.6f} で、"
                f"代表値として {float(factor):.6f} を使っています。"
                f"この代表値でターゲット解答まで一致するかは {predicted == answer} です。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": {
                "factor": float(factor),
                "factor_interval": [float(factor_low), float(factor_high)],
                "target_factor_interval": [float(target_factor_low), float(target_factor_high)],
                "factor_interval_with_target": [float(consistent_low), float(consistent_high)]
                if target_consistent
                else None,
                "target_consistent_with_representative": predicted == answer,
                "target_consistent_interval_exists": target_consistent,
                "predicted_target_output": predicted,
                "rows": analysis_rows,
            },
        }


EXTRACTOR = UnitPromptExtractor()
