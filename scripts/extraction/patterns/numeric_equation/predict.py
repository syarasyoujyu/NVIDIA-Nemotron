"""別解候補の予測。

answerを使わずに例示のみから導出できるルールが複数ある場合、
各ルールがターゲット入力に対して何を予測するかを計算する。
"""
from __future__ import annotations

from .constants import _ARITH_OPS, _REVERSAL_MODE_NAMES
from .matching import (
    _op_group_abs_num_rev_op,
    _op_group_abs_rev_op_suffix,
    _op_group_operation,
)


def _predict(
    a_str: str, b_str: str, op_char: str,
    op_name: str, offset: int,
    rev_in: bool, swap: bool, out_mode: str,
) -> str | None:
    """指定ルールで (a_str op b_str) の予測出力を返す。

    入力変換 → 演算 → 出力変換 の順に適用する。
    連結系は先頭候補のみ返す（swap で両方試す前提）。
    計算できない場合は None。
    """
    a_s = a_str[::-1] if rev_in else a_str
    b_s = b_str[::-1] if rev_in else b_str
    if swap:
        a_s, b_s = b_s, a_s

    if op_name == "concat":
        concat = a_s + b_s
        return concat[::-1] if out_mode != "none" else concat
    if op_name == "concat_strip":
        concat = a_s + b_s
        stripped = concat.lstrip('0') or '0'
        return stripped[::-1] if out_mode != "none" else stripped

    fn_map = {op: fn for op, fn in _ARITH_OPS}
    fn = fn_map.get(op_name)
    if fn is None:
        return None

    raw = fn(int(a_s), int(b_s))
    val = raw + offset

    if out_mode == "num_rev":
        # 数値反転: 絶対値の桁を逆順にし、負なら演算子符号を前置
        if val >= 0:
            return str(val)[::-1]
        return op_char + str(-val)[::-1]
    if out_mode == "num_rev_sfx":
        # 数値反転サフィックス: 絶対値反転＋末尾に演算子記号
        return str(abs(val))[::-1] + op_char
    if out_mode == "full_rev":
        # 全反転: 符号含む文字列ごと逆順
        if val >= 0:
            return str(val)[::-1]
        return (op_char + str(-val))[::-1]
    # "none"
    if val >= 0:
        return str(val)
    return op_char + str(-val)


def _all_modes_for_group(
    examples: list[tuple[str, str, str, str]],
) -> list[tuple[tuple[bool, bool, str], str, int]]:
    """例示に対して成立する全ての (mode, op_name, offset) を返す。

    _best_mode_for_group と同じ探索順（フェーズ1 → 1.5 → 2）で
    成立するものを全て収集する（最初の1件で止まらない）。
    重複は除外する。
    """
    seen: set[tuple] = set()
    results: list[tuple[tuple[bool, bool, str], str, int]] = []

    def _add(mode, op_name, offset):
        key = (mode, op_name, offset)
        if key not in seen:
            seen.add(key)
            results.append(key)

    # フェーズ1: offset=0
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=False)
        if result is not None:
            _add(mode, result[0], result[1])

    # フェーズ1.5: 特殊出力パターン
    for rev_in in (False, True):
        for swap in (False, True):
            for op_name, fn in _ARITH_OPS:
                if _op_group_abs_num_rev_op(examples, fn, rev_in, swap):
                    _add((rev_in, swap, "num_rev"), op_name, 0)
    for rev_in in (False, True):
        for swap in (False, True):
            for op_name, fn in _ARITH_OPS:
                if _op_group_abs_rev_op_suffix(examples, fn, rev_in, swap):
                    _add((rev_in, swap, "num_rev_sfx"), op_name, 0)

    # フェーズ2: offset ±1
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=True)
        if result is not None:
            _add(mode, result[0], result[1])

    return results
