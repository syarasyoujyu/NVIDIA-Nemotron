from __future__ import annotations

import re

from ..base import PatternValidator


def consistent_substitution(pairs: list[tuple[str, str]]) -> bool:
    cipher_to_plain: dict[str, str] = {}
    plain_to_cipher: dict[str, str] = {}

    for cipher_text, plain_text in pairs:
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        if len(cipher_words) != len(plain_words):
            return False

        for cipher_word, plain_word in zip(cipher_words, plain_words):
            if len(cipher_word) != len(plain_word):
                return False
            for cipher_char, plain_char in zip(cipher_word, plain_word):
                prev_plain = cipher_to_plain.get(cipher_char)
                if prev_plain is not None and prev_plain != plain_char:
                    return False
                cipher_to_plain[cipher_char] = plain_char

                prev_cipher = plain_to_cipher.get(plain_char)
                if prev_cipher is not None and prev_cipher != cipher_char:
                    return False
                plain_to_cipher[plain_char] = cipher_char
    return True


class TextValidator(PatternValidator):
    pattern_name = "text_decryption"

    def family_names(self) -> list[str]:
        return ["monoalphabetic_substitution_with_word_boundaries"]

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        pairs = re.findall(r"([a-z ]+) -> ([a-z ]+)", row["prompt"])
        target = re.search(r"Now, decrypt the following text: ([a-z ]+)", row["prompt"])
        if target is None:
            return False

        all_pairs = [(cipher, plain) for cipher, plain in pairs]
        all_pairs.append((target.group(1), row["answer"]))
        return consistent_substitution(all_pairs)


VALIDATOR = TextValidator()
