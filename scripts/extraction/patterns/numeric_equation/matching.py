"""演算子グループに対するモードマッチング。

各演算子グループ（同じ記号を使う例の集合）に対して、
どの変換ルールが全例示と一致するかを検索する。

## マッチング優先順位

1. **フェーズ1（offset=0）**: 全12モード × 全演算で offset=0 のみ試す
   - モード順: (rev_in, swap, out_mode) の辞書順
   - 演算順: add → sub → mul → abs_diff → neg_abs_diff → mod → concat → concat_strip

2. **フェーズ1.5（特殊出力パターン）**: 通常の算術では表現できない出力形式を試す
   - abs_diff_op_sign : rhs = op + str(|a-b|)
   - abs_num_rev_op   : rhs = op + str(abs(result))[::-1]（数値反転＋演算子符号プレフィックス）
   - abs_rev_op_suffix: rhs = str(abs(result))[::-1] + op（数値反転＋演算子符号サフィックス）

3. **フェーズ2（offset ±1）**: フェーズ1と同じ順で offset -1/+1 も許容して再試行

## オフセット制約
調整値は -1, 0, +1 のみ許可。最初の例示からオフセットを決定し、残りの例示で検証する。
"""
from __future__ import annotations

from .constants import _ARITH_OPS, _REVERSAL_MODE_NAMES


def _find_consistent_offset(
    examples: list[tuple[str, str, str, str]],
    arith_fn,
    rev_in: bool, swap: bool, out_mode: str,
) -> int | None:
    """全例示に一致する一定のオフセットを求める。

    最初の例示から offset = rhs_as_int - raw_result を計算し、
    -1/0/+1 の範囲内かつ残り全例示でも成立する場合に offset を返す。
    成立しない場合は None。
    """
    if not examples:
        return None

    def raw_result(a_str: str, b_str: str) -> int:
        a_s = a_str[::-1] if rev_in else a_str
        b_s = b_str[::-1] if rev_in else b_str
        if swap:
            a_s, b_s = b_s, a_s
        return arith_fn(int(a_s), int(b_s))

    def expected_rhs(raw: int, offset: int, op_char: str) -> str | None:
        """(raw + offset) を out_mode に従って文字列化する。"""
        val = raw + offset
        if out_mode == "full_rev":
            if val >= 0:
                return str(val)[::-1]
            return (op_char + str(-val))[::-1]
        if out_mode == "num_rev":
            if val >= 0:
                return str(val)[::-1]
            # 負数: 符号はそのまま、桁のみ逆順
            return op_char + str(-val)[::-1]
        # "none"
        if val >= 0:
            return str(val)
        return op_char + str(-val)

    # 最初の例示からオフセットを推定
    a, o, b, r = examples[0]
    raw0 = raw_result(a, b)
    if out_mode == "full_rev":
        r_as_int = _rhs_to_int_local(r[::-1])
        if r_as_int is None:
            rev_r = r[::-1]
            if rev_r.startswith(o) and rev_r[len(o):].lstrip('0').isdigit():
                r_as_int = -int(rev_r[len(o):]) if rev_r[len(o):] else None
    elif out_mode == "num_rev":
        r_as_int = _rhs_to_int_local(r[::-1])
        if r_as_int is None:
            if r.startswith(o) and r[len(o):].lstrip('0').isdigit():
                r_as_int = -int(r[len(o):][::-1]) if r[len(o):] else None
    else:
        r_as_int = _rhs_to_int_local(r)
        if r_as_int is None and r.startswith(o) and r[len(o):].lstrip('0').isdigit():
            r_as_int = -int(r[len(o):]) if r[len(o):] else None
    if r_as_int is None:
        return None
    offset = r_as_int - raw0

    if offset not in (-1, 0, 1):
        return None

    # 残りの例示で検証
    for a, o, b, r in examples[1:]:
        raw = raw_result(a, b)
        exp = expected_rhs(raw, offset, o)
        if exp is None or exp != r:
            return None

    return offset


def _rhs_to_int_local(rhs: str) -> int | None:
    try:
        return int(rhs)
    except ValueError:
        return None


def _example_matches_concat(
    a_str: str, b_str: str, rhs: str,
    rev_in: bool, swap: bool, out_mode: str,
) -> bool:
    """1例示が連結演算（文字列結合）で説明できるか検証する。

    a+b と b+a の両方を試み、out_mode に従い出力変換を適用して rhs と比較する。
    concat は数値でなく文字列を扱うため num_rev と full_rev は同一動作。
    """
    a_s = a_str[::-1] if rev_in else a_str
    b_s = b_str[::-1] if rev_in else b_str
    if swap:
        a_s, b_s = b_s, a_s
    for concat in [a_s + b_s, b_s + a_s]:
        expected = concat[::-1] if out_mode != "none" else concat
        if expected == rhs:
            return True
    return False


def _example_matches_concat_strip(
    a_str: str, b_str: str, rhs: str,
    rev_in: bool, swap: bool, out_mode: str,
) -> bool:
    """1例示が連結＋先頭0除去演算で説明できるか検証する。

    連結後に先頭の "0" を除去（例: "0133" → "133"）してから rhs と比較する。
    """
    a_s = a_str[::-1] if rev_in else a_str
    b_s = b_str[::-1] if rev_in else b_str
    if swap:
        a_s, b_s = b_s, a_s
    for concat in [a_s + b_s, b_s + a_s]:
        stripped = concat.lstrip('0') or '0'
        expected = stripped[::-1] if out_mode != "none" else stripped
        if expected == rhs:
            return True
    return False



def _op_group_abs_num_rev_op(
    examples: list[tuple[str, str, str, str]],
    arith_fn,
    rev_in: bool, swap: bool,
) -> bool:
    """全例示が「演算子符号前置＋絶対値数値反転」パターンか検証する。

    rhs = op + str(abs(result))[::-1] の形式（数値反転）。
    例: 16-61 → raw=-45, abs=45, rev="54" → rhs = "-54"
    """
    for a_str, op_char, b_str, rhs in examples:
        a_s = a_str[::-1] if rev_in else a_str
        b_s = b_str[::-1] if rev_in else b_str
        if swap:
            a_s, b_s = b_s, a_s
        raw = arith_fn(int(a_s), int(b_s))
        if rhs != op_char + str(abs(raw))[::-1]:
            return False
    return True


def _op_group_abs_rev_op_suffix(
    examples: list[tuple[str, str, str, str]],
    arith_fn,
    rev_in: bool, swap: bool,
) -> bool:
    """全例示が「絶対値数値反転＋演算子符号後置」パターンか検証する。

    rhs = str(abs(result))[::-1] + op の形式（数値反転＋サフィックス）。
    """
    for a_str, op_char, b_str, rhs in examples:
        a_s = a_str[::-1] if rev_in else a_str
        b_s = b_str[::-1] if rev_in else b_str
        if swap:
            a_s, b_s = b_s, a_s
        raw = arith_fn(int(a_s), int(b_s))
        if rhs != str(abs(raw))[::-1] + op_char:
            return False
    return True


def _op_group_operation(
    examples: list[tuple[str, str, str, str]],
    rev_in: bool, swap: bool, out_mode: str,
    allow_nonzero_offset: bool = True,
) -> tuple[str, int] | None:
    """指定モードで全例示に一致する演算名とオフセットを返す。

    算術演算（add/sub/mul/abs_diff/neg_abs_diff/mod）を順に試し、
    次に concat、concat_strip を試す。
    allow_nonzero_offset=False の場合は offset=0 のみ許可。
    いずれも成立しなければ None。
    """
    for op_name, fn in _ARITH_OPS:
        offset = _find_consistent_offset(examples, fn, rev_in, swap, out_mode)
        if offset is not None:
            if not allow_nonzero_offset and offset != 0:
                continue
            return (op_name, offset)
    if all(_example_matches_concat(a, b, r, rev_in, swap, out_mode) for a, _, b, r in examples):
        return ("concat", 0)
    if all(_example_matches_concat_strip(a, b, r, rev_in, swap, out_mode) for a, _, b, r in examples):
        return ("concat_strip", 0)
    return None


def _best_mode_for_group(
    examples: list[tuple[str, str, str, str]],
) -> tuple[tuple[bool, bool, str], str, int] | None:
    """演算子グループに対して最初に一致した (mode, op_name, offset) を返す。

    ## 探索順序
    1. フェーズ1: 全12モードで offset=0 のみ試す（最もシンプルなルールを優先）
    2. フェーズ1.5: 特殊出力パターン（abs_diff_op_sign / abs_num_rev_op / abs_rev_op_suffix）
    3. フェーズ2: 全12モードで offset ±1 も許容して再試行

    最初に見つかったものを返し、それ以降の探索は行わない。
    どのパターンにも一致しない場合は None。
    """
    # フェーズ1: offset=0 のみ
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=False)
        if result is not None:
            return (mode, result[0], result[1])

    # フェーズ1.5: 特殊出力パターン
    for rev_in in (False, True):
        for swap in (False, True):
            for op_name, fn in _ARITH_OPS:
                if _op_group_abs_num_rev_op(examples, fn, rev_in, swap):
                    return ((rev_in, swap, "num_rev"), op_name, 0)
    for rev_in in (False, True):
        for swap in (False, True):
            for op_name, fn in _ARITH_OPS:
                if _op_group_abs_rev_op_suffix(examples, fn, rev_in, swap):
                    return ((rev_in, swap, "num_rev_sfx"), op_name, 0)

    # フェーズ2: offset ±1 も許容
    for mode in _REVERSAL_MODE_NAMES:
        rev_in, swap, out_mode = mode
        result = _op_group_operation(examples, rev_in, swap, out_mode, allow_nonzero_offset=True)
        if result is not None:
            return (mode, result[0], result[1])

    return None
