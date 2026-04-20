from __future__ import annotations

import re
from statistics import mean

from .base import PatternExtractor


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
        target_time = float(target.group(1)) if target else 0.0

        pairs = [(float(pair["input"]), float(pair["output"])) for pair in examples]
        gravities = [2 * d / (t * t) for t, d in pairs if t != 0]
        gravity = mean(gravities) if gravities else 0.0
        predicted = round(0.5 * gravity * target_time * target_time, 2)

        diagram_lines = [
            f"t --square--> t^2 --* {0.5 * gravity:.6f} --> d",
            f"estimated g = {gravity:.6f}",
        ]
        for t, d in pairs[:6]:
            diagram_lines.append(
                f"{t:.2f}s --square--> {t*t:.4f} --* {0.5 * gravity:.6f} --> {d:.2f}m"
            )
        diagram_lines.append(
            f"{target_time:.2f}s --square--> {target_time*target_time:.4f} --* {0.5 * gravity:.6f} --> {answer}m  [対象]"
        )

        analysis_rows = []
        for t, d in pairs:
            analysis_rows.append(
                {
                    "time_s": t,
                    "distance_m": d,
                    "estimated_g": round(2 * d / (t * t), 6) if t else None,
                }
            )

        return {
            "examples": examples,
            "target_input": f"{target_time:.2f}",
            "answer": answer,
            "rule_summary": (
                f"距離は d = 0.5 * g * t^2 に従います。各行ごとに g が異なり、"
                f"この行の推定 g は {gravity:.6f} です。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": {
                "gravity": gravity,
                "predicted_target_output": f"{predicted:.2f}",
                "rows": analysis_rows,
            },
        }


EXTRACTOR = GravityPromptExtractor()
