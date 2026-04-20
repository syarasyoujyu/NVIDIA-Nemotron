from __future__ import annotations

import re
from statistics import mean

from .base import PatternExtractor


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
        target_value = float(target.group(1)) if target else 0.0

        pairs = [(float(pair["input"]), float(pair["output"])) for pair in examples]
        multipliers = [y / x for x, y in pairs if x != 0]
        factor = mean(multipliers) if multipliers else 0.0
        predicted = round(target_value * factor, 2)

        diagram_lines = [f"x --* {factor:.6f} --> y"]
        for x, y in pairs[:6]:
            diagram_lines.append(f"{x:.2f} --* {factor:.6f} --> {y:.2f}")
        diagram_lines.append(
            f"{target_value:.2f} --* {factor:.6f} --> {answer}  [対象]"
        )

        analysis_rows = []
        for x, y in pairs:
            analysis_rows.append(
                {
                    "input_m": x,
                    "output": y,
                    "estimated_output": round(x * factor, 2),
                    "ratio": round(y / x, 6) if x else None,
                }
            )

        return {
            "examples": examples,
            "target_input": f"{target_value:.2f}",
            "answer": answer,
            "rule_summary": (
                f"各行ごとにほぼ一定の倍率で変換される線形変換です。推定倍率は {factor:.6f} です。"
                "表では観測値と倍率ベースの推定値を比較できます。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": {
                "factor": factor,
                "predicted_target_output": f"{predicted:.2f}",
                "rows": analysis_rows,
            },
        }


EXTRACTOR = UnitPromptExtractor()
