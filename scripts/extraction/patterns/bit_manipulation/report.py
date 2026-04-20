from __future__ import annotations

import re

from ..base import PatternExtractor


class BitPromptExtractor(PatternExtractor):
    pattern_name = "bit_manipulation"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        examples = [
            {"input": left, "output": right}
            for left, right in re.findall(r"([01]{8}) -> ([01]{8})", prompt)
        ]
        target = re.search(r"Now, determine the output for: ([01]{8})", prompt)
        target_input = target.group(1) if target else ""

        rows = []
        for pair in examples:
            input_positions = [str(i) for i, bit in enumerate(pair["input"]) if bit == "1"]
            output_positions = [str(i) for i, bit in enumerate(pair["output"]) if bit == "1"]
            rows.append(
                {
                    "input": pair["input"],
                    "output": pair["output"],
                    "input_one_positions": input_positions,
                    "output_one_positions": output_positions,
                    "input_one_count": len(input_positions),
                    "output_one_count": len(output_positions),
                }
            )

        diagram_lines = [f"{row['input']} -> {row['output']}" for row in rows[:8]]
        diagram_lines.append(f"{target_input} -> {answer}  [対象]")

        return {
            "examples": examples,
            "target_input": target_input,
            "answer": answer,
            "rule_summary": (
                "8ビット列から8ビット列への変換です。プロンプトには回転・シフト・"
                "AND/OR/XOR/NOT などの候補が書かれていますが、例題だけでは一意に決まりません。"
                "そのため、ここでは入力と出力の対応だけを中立に並べています。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": rows,
        }


EXTRACTOR = BitPromptExtractor()
