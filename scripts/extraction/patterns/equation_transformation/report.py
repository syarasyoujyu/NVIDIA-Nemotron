from __future__ import annotations

import difflib
from collections import Counter

from ..base import PatternExtractor


class EquationPromptExtractor(PatternExtractor):
    pattern_name = "equation_transformation"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        examples = []
        target_text = ""
        for line in prompt.splitlines():
            if " = " in line:
                left, right = line.split(" = ", 1)
                examples.append({"input": left, "output": right})
            elif line.startswith("Now, determine the result for: "):
                target_text = line.split(": ", 1)[1]

        analysis_rows = []
        diagram_lines = []
        for pair in examples[:8]:
            input_chars = Counter(pair["input"])
            output_chars = Counter(pair["output"])
            common_chars = sorted((input_chars & output_chars).elements())
            removed_chars = sorted((input_chars - output_chars).elements())
            added_chars = sorted((output_chars - input_chars).elements())

            diff_ops = []
            for opcode in difflib.SequenceMatcher(
                a=pair["input"], b=pair["output"]
            ).get_opcodes():
                tag, i1, i2, j1, j2 = opcode
                if tag == "equal":
                    continue
                diff_ops.append(
                    {
                        "op": tag,
                        "from": pair["input"][i1:i2],
                        "to": pair["output"][j1:j2],
                    }
                )

            analysis_rows.append(
                {
                    "input": pair["input"],
                    "output": pair["output"],
                    "common_chars": common_chars,
                    "removed_chars": removed_chars,
                    "added_chars": added_chars,
                    "diff_ops": diff_ops,
                }
            )

            diagram_lines.append(
                f"{pair['input']} -> {pair['output']}  common={''.join(common_chars) or '-'}  removed={''.join(removed_chars) or '-'}  added={''.join(added_chars) or '-'}"
            )

        diagram_lines.append(f"{target_text} -> {answer}  [対象]")

        return {
            "examples": examples,
            "target_input": target_text,
            "answer": answer,
            "rule_summary": (
                "記号列の書き換え問題です。関係図では各例題について、"
                "共通記号、削除された記号、追加された記号、差分操作を要約しています。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": analysis_rows,
        }


EXTRACTOR = EquationPromptExtractor()
