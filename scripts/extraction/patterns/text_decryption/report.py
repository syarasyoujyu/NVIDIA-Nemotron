from __future__ import annotations

from collections import Counter, defaultdict

from ..base import PatternExtractor


class TextPromptExtractor(PatternExtractor):
    pattern_name = "text_decryption"

    def parse_prompt(self, prompt: str, answer: str) -> dict:
        lines = prompt.splitlines()
        examples = []
        target_text = ""
        for line in lines:
            if " -> " in line:
                left, right = line.split(" -> ", 1)
                examples.append({"input": left, "output": right})
            elif line.startswith("Now, decrypt the following text: "):
                target_text = line.split(": ", 1)[1]

        char_map: dict[str, Counter[str]] = defaultdict(Counter)
        word_map: dict[str, str] = {}
        for pair in examples:
            enc_words = pair["input"].split()
            dec_words = pair["output"].split()
            for enc_word, dec_word in zip(enc_words, dec_words):
                word_map[enc_word] = dec_word
                if len(enc_word) == len(dec_word):
                    for enc_char, dec_char in zip(enc_word, dec_word):
                        char_map[enc_char][dec_char] += 1

        char_lines = []
        for enc_char in sorted(char_map):
            outputs = ", ".join(
                f"{plain}:{count}" for plain, count in char_map[enc_char].most_common(4)
            )
            char_lines.append(f"{enc_char} -> {outputs}")

        word_rows = []
        for enc_word, dec_word in list(word_map.items())[:12]:
            word_rows.append({"encrypted": enc_word, "decrypted": dec_word})

        diagram_lines = [f"{pair['input']} -> {pair['output']}" for pair in examples[:6]]
        diagram_lines.append(f"{target_text} -> {answer}  [対象]")
        diagram_lines.append("")
        diagram_lines.append("文字対応のヒント:")
        diagram_lines.extend(char_lines[:20])

        return {
            "examples": examples,
            "target_input": target_text,
            "answer": answer,
            "rule_summary": (
                "単語単位の例題から、同じ長さの暗号語と復号語を位置合わせして、"
                "文字対応のヒント表を作っています。"
            ),
            "relation_diagram": "\n".join(diagram_lines),
            "analysis": {
                "word_map_samples": word_rows,
                "char_map": {k: dict(v) for k, v in char_map.items()},
            },
        }


EXTRACTOR = TextPromptExtractor()
