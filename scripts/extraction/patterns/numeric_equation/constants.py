"""演算モードおよび演算種別の定数定義。

入力反転・オペランド交換・出力変換の組み合わせ (rev_in, swap, out_mode) を
「モード」と呼ぶ。out_mode の種類:
  - "none"        : 出力をそのまま文字列化
  - "num_rev"     : 数値反転（桁のみ逆順）
  - "num_rev_sfx" : 数値反転＋末尾に演算子記号
  - "full_rev"    : 全反転（符号含む文字列ごと逆順）
"""
from __future__ import annotations

# (rev_input, swap_operands, out_mode) → モード識別名
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


def _mod(a: int, b: int) -> int:
    """大きい方を小さい方で割った余り（常に max÷min）。"""
    lo, hi = (a, b) if a <= b else (b, a)
    return hi % lo if lo != 0 else 10 ** 18


# 試行する算術演算の一覧。優先順に並んでいる。
_ARITH_OPS: list[tuple[str, object]] = [
    ("add",          lambda a, b: a + b),
    ("sub",          lambda a, b: a - b),
    ("mul",          lambda a, b: a * b),
    ("abs_diff",     lambda a, b: abs(a - b)),
    ("neg_abs_diff", lambda a, b: -(abs(a - b))),
    ("mod",          _mod),
]

# 演算名 → 日本語表示ラベル
_OP_LABELS_JP: dict[str, str] = {
    "add":               "足し算",
    "sub":               "引き算",
    "mul":               "掛け算",
    "abs_diff":          "引き算（絶対値）",
    "neg_abs_diff":      "引き算（－絶対値）",
    "mod":               "剰余（大÷小）",
    "concat":            "連結",
    "concat_strip":      "連結（先頭0除去）",
}
