from __future__ import annotations

import re

from .base import PatternExtractor


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
            delta = format(int(pair["input"], 2) ^ int(pair["output"], 2), "08b")
            flipped = [str(i) for i, bit in enumerate(delta) if bit == "1"]
            rows.append(
                {
                    "input": pair["input"],
                    "output": pair["output"],
                    "xor_delta": delta,
                    "flipped_output_positions": flipped,
                }
            )

        diagram_lines = [
            f"{row['input']} -> {row['output']}  xor={row['xor_delta']}  flip_pos={','.join(row['flipped_output_positions']) or '-'}"
            for row in rows[:8]
        ]
        diagram_lines.append(f"{target_input} -> {answer}  [対象]")

        return {
            "examples": examples,
            "target_input": target_input,
            "answer": answer,
            "rule_summary": (
                "8ビット列から8ビット列への変換です。例題ごとの入力と出力に加えて、"
                "入力と出力の XOR 差分も併記しています。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": rows,
        }


EXTRACTOR = BitPromptExtractor()
