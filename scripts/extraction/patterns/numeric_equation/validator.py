from __future__ import annotations

import json
import re
from pathlib import Path

from ..base import PatternValidator

# (rev_input, swap_operands, out_mode) の12通りのモード
# out_mode: "none" = そのまま, "num_rev" = 数値反転（符号維持・桁反転）, "full_rev" = 全反転（符号含む文字列反転）
_REVERSAL_MODE_NAMES: dict[tuple[bool, bool, str], str] = {
    (False, False, "none"):     "no_reversal",
    (True,  False, "none"):     "reverse_input",
    (False, True,  "none"):     "swap",
    (True,  True,  "none"):     "reverse_input_swap",
    (False, False, "num_rev"):  "num_reverse_output",
    (True,  False, "num_rev"):  "reverse_input_num_reverse_output",
    (False, True,  "num_rev"):  "swap_num_reverse_output",
    (True,  True,  "num_rev"):  "reverse_input_swap_num_reverse_output",
    (False, False, "full_rev"): "full_reverse_output",
    (True,  False, "full_rev"): "reverse_input_full_reverse_output",
    (False, True,  "full_rev"): "swap_full_reverse_output",
    (True,  True,  "full_rev"): "reverse_input_swap_full_reverse_output",
}

_ARITH_OPS = [
    ("add",      lambda a, b: a + b),
    ("sub",      lambda a, b: a - b),
    ("mul",      lambda a, b: a * b),
    ("abs_diff", lambda a, b: abs(a - b)),
]

_OP_LABELS_JP = {
    "add":      "足し算",
    "sub":      "引き算",
    "mul":      "掛け算",
    "abs_diff": "絶対差",
    "concat":   "連結",
}


def _offset_str(offset: int) -> str:
    if offset == 0:
        return ""
    return f" + {offset}" if offset > 0 else f" − {-offset}"


def _flow_label(op_name: str, offset: int, rev_in: bool, swap: bool, out_mode: str) -> str:
    parts = []
    if rev_in:
        parts.append("反転")
    if swap:
        parts.append("交換")
    parts.append(_OP_LABELS_JP[op_name] + _offset_str(offset))
    if out_mode == "num_rev":
        parts.append("数値反転")
    elif out_mode == "full_rev":
        parts.append("全反転")
    return " → ".join(parts)


def _parse_lhs(lhs: str) -> tuple[str, str, str] | None:
    m = re.match(r'^(\d+)([^\d]+)(\d+)$', lhs.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _rhs_to_int(rhs: str) -> int | None:
    try:
        return int(rhs)
    except ValueError:
        return None


def _find_consistent_offset(
    examples: list[tuple[str, str, str, str]],
    arith_fn,
    rev_in: bool, swap: bool, out_mode: str,
) -> int | None:
    if not examples:
        return None

    def raw_result(a_str: str, b_str: str) -> int:
        a_s = a_str[::-1] if rev_in else a_str
        b_s = b_str[::-1] if rev_in else b_str
        if swap:
            a_s, b_s = b_s, a_s
        return arith_fn(int(a_s), int(b_s))

    def expected_rhs(raw: int, offset: int, op_char: str) -> str | None:
        val = raw + offset
        if out_mode == "full_rev":
            if val >= 0:
                return str(val)[::-1]
            return (op_char + str(-val))[::-1]
        if out_mode == "num_rev":
            if val >= 0:
                return str(val)[::-1]
            # 数値反転: 符号はそのまま、桁のみ反転
            return op_char + str(-val)[::-1]
        # "none"
        if val >= 0:
            return str(val)
        return op_char + str(-val)

    # 最初の例からオフセットを決定
    a, o, b, r = examples[0]
    raw0 = raw_result(a, b)
    if out_mode == "full_rev":
        # rhs = rev(str(raw+offset)) or rev(op+str(|raw+offset|))
        r_as_int = _rhs_to_int(r[::-1])
        if r_as_int is None:
            rev_r = r[::-1]
            if rev_r.startswith(o) and rev_r[len(o):].lstrip('0').isdigit():
                r_as_int = -int(rev_r[len(o):]) if rev_r[len(o):] else None
    elif out_mode == "num_rev":
        # 正数: rhs = rev(str(val)) → r[::-1] がそのまま val
        r_as_int = _rhs_to_int(r[::-1])
        if r_as_int is None:
            # 負数: rhs = op + rev(str(-val)) → r[len(o):] の桁を再反転
            if r.startswith(o) and r[len(o):].lstrip('0').isdigit():
                r_as_int = -int(r[len(o):][::-1]) if r[len(o):] else None
    else:
        r_as_int = _rhs_to_int(r)
        if r_as_int is None and r.startswith(o) and r[len(o):].lstrip('0').isdigit():
            r_as_int = -int(r[len(o):]) if r[len(o):] else None
    if r_as_int is None:
        return None
    offset = r_as_int - raw0

    # 例が1つだけの場合は offset=0 のみ許可
    if len(examples) == 1:
        return offset if offset == 0 else None

    # 残りの例で検証
    for a, o, b, r in examples[1:]:
        raw = raw_result(a, b)
        exp = expected_rhs(raw, offset, o)
        if exp is None or exp != r:
            return None

    return offset


def _example_matches_concat(
    a_str: str, b_str: str, rhs: str,
    rev_in: bool, swap: bool, out_mode: str,
) -> bool:
    a_s = a_str[::-1] if rev_in else a_str
    b_s = b_str[::-1] if rev_in else b_str
    if swap:
        a_s, b_s = b_s, a_s
    # concat結果は桁のみ（符号なし）なので num_rev と full_rev は同じ
    for concat in [a_s + b_s, b_s + a_s]:
        expected = concat[::-1] if out_mode != "none" else concat
        if expected == rhs:
            return True
    return False


def _op_group_operation(
    examples: list[tuple[str, str, str, str]],
    rev_in: bool, swap: bool, out_mode: str,
    allow_nonzero_offset: bool = True,
) -> tuple[str, int] | None:
    for op_name, fn in _ARITH_OPS:
        offset = _find_consistent_offset(examples, fn, rev_in, swap, out_mode)
        if offset is not None:
            if not allow_nonzero_offset and offset != 0:
                continue
            return (op_name, offset)
    if all(_example_matches_concat(a, b, r, rev_in, swap, out_mode) for a, _, b, r in examples):
        return ("concat", 0)
    return None


def _best_mode_for_group(
    examples: list[tuple[str, str, str, str]],
) -> tuple[tuple[bool, bool, str], str, int] | None:
    """演算子グループに対して最初に一致した (mode, op_name, offset) を返す。
    優先順位: offset=0 の全モード → 非ゼロoffset の全モード
    out_mode の順: none → num_rev → full_rev
    """
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=False)
        if result is not None:
            return (mode, result[0], result[1])
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=True)
        if result is not None:
            return (mode, result[0], result[1])
    return None


def _parse_op_groups(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str, str, str]]] | None:
    op_groups: dict[str, list[tuple[str, str, str, str]]] = {}
    for lhs, rhs in pairs:
        parsed = _parse_lhs(lhs)
        if parsed is None:
            return None
        a_str, op, b_str = parsed
        op_groups.setdefault(op, []).append((a_str, op, b_str, rhs))
    return op_groups


class NumericEquationValidator(PatternValidator):
    pattern_name = "numeric_equation"

    def family_names(self) -> list[str]:
        # 演算子ごとに独立してモードを決定する（混在を許容）
        return ["per_operator_any_mode"]

    def _parse_pairs(self, row: dict[str, str]) -> list[tuple[str, str]]:
        pairs = []
        target_input = None
        for line in row["prompt"].splitlines():
            if " = " in line:
                left, right = line.split(" = ", 1)
                pairs.append((left.strip(), right.strip()))
            elif line.startswith("Now, determine the result for: "):
                target_input = line.split(": ", 1)[1].strip()
        if target_input is None:
            return []
        pairs.append((target_input, row["answer"]))
        return pairs

    def matches_family(self, row: dict[str, str], family_name: str) -> bool:
        if family_name != "per_operator_any_mode":
            return False
        pairs = self._parse_pairs(row)
        if not pairs:
            return False
        op_groups = _parse_op_groups(pairs)
        if op_groups is None:
            return False
        # 各演算子グループが独立していずれかのモード+演算で説明できるか
        return all(
            _best_mode_for_group(examples) is not None
            for examples in op_groups.values()
        )

    def _build_matched_entry(self, row: dict[str, str]) -> dict | None:
        result = self.validate_row(row)
        if not result.matched:
            return None

        pairs = self._parse_pairs(row)
        op_groups = _parse_op_groups(pairs)
        if op_groups is None:
            return None

        # 演算子ごとに独立してモードを決定しフローラベルを生成
        operator_flows = {}
        for op_char, examples in op_groups.items():
            best = _best_mode_for_group(examples)
            if best:
                mode, op_name, offset = best
                rev_in, swap, out_mode = mode
                operator_flows[op_char] = _flow_label(op_name, offset, rev_in, swap, out_mode)

        return {
            "id": row["id"],
            "pattern": self.pattern_name,
            "operator_flows": operator_flows,
        }

    def write_outputs(self, output_root: Path, rows: list[dict[str, str]]) -> dict[str, str]:
        result = super().write_outputs(output_root, rows)

        pattern_dir = output_root / self.pattern_name
        matched_entries = [
            entry for row in rows
            if (entry := self._build_matched_entry(row)) is not None
        ]
        matched_path = pattern_dir / "matched.jsonl"
        with matched_path.open("w", encoding="utf-8") as f:
            for entry in matched_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return result


class NumericEquationDeduceValidator(NumericEquationValidator):
    pattern_name = "numeric_equation/deduce"


class NumericEquationGuessValidator(NumericEquationValidator):
    pattern_name = "numeric_equation/guess"


VALIDATOR = NumericEquationValidator()
DEDUCE_VALIDATOR = NumericEquationDeduceValidator()
GUESS_VALIDATOR = NumericEquationGuessValidator()
