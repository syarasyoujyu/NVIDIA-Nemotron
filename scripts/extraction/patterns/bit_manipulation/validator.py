from __future__ import annotations

import re

from ..base import PatternValidator


def bit(value: str, index: int) -> int:
    return int(value[index])


def unary_candidates():
    yield ("const0", lambda x: 0)
    yield ("const1", lambda x: 1)
    for i in range(8):
        yield (f"id_{i}", lambda x, i=i: bit(x, i))
        yield (f"not_{i}", lambda x, i=i: 1 - bit(x, i))


def binary_candidates():
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            yield (f"and_{i}_{j}", lambda x, i=i, j=j: bit(x, i) & bit(x, j))
            yield (f"or_{i}_{j}", lambda x, i=i, j=j: bit(x, i) | bit(x, j))
            yield (f"xor_{i}_{j}", lambda x, i=i, j=j: bit(x, i) ^ bit(x, j))
            yield (
                f"and_not_{i}_{j}",
                lambda x, i=i, j=j: bit(x, i) & (1 - bit(x, j)),
            )
            yield (
                f"or_not_{i}_{j}",
                lambda x, i=i, j=j: bit(x, i) | (1 - bit(x, j)),
            )
            yield (
                f"xor_not_{i}_{j}",
                lambda x, i=i, j=j: bit(x, i) ^ (1 - bit(x, j)),
            )


class BitValidator(PatternValidator):
    pattern_name = "bit_manipulation"
    _candidate_functions = list(unary_candidates()) + list(binary_candidates())

    def family_names(self) -> list[str]:
        return [
            "unary_or_constant_per_output_bit",
            "local_unary_binary_boolean_rule",
        ]

    def _parse_pairs(self, row: dict[str, str]) -> list[tuple[str, str]]:
        pairs = re.findall(r"([01]{8}) -> ([01]{8})", row["prompt"])
        target = re.search(r"Now, determine the output for: ([01]{8})", row["prompt"])
        if target is None:
            return []
        pairs.append((target.group(1), row["answer"]))
        return pairs

    def _output_bit_supported(
        self,
        examples: list[tuple[str, str]],
        output_index: int,
        allow_binary: bool,
    ) -> bool:
        candidates = unary_candidates() if not allow_binary else self._candidate_functions
        for _, fn in candidates:
            ok = True
            for input_bits, output_bits in examples:
                if fn(input_bits) != int(output_bits[output_index]):
                    ok = False
                    break
            if ok:
                return True
        return False

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        examples = self._parse_pairs(row)
        if not examples:
            return False

        allow_binary = family_name == "local_unary_binary_boolean_rule"
        return all(
            self._output_bit_supported(examples, output_index, allow_binary)
            for output_index in range(8)
        )


VALIDATOR = BitValidator()
