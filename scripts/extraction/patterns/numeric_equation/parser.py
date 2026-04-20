"""プロンプトから方程式ペアを解析するパーサー。

プロンプトの各行は「左辺 = 右辺」形式の例示、または
「Now, determine the result for: 左辺」形式のターゲット入力からなる。

左辺は「数字 演算子記号 数字」の形式（例: "12+34", "5*7"）。
右辺は整数文字列または演算子記号プレフィックス付きの文字列（例: "-54", "/17"）。
"""
from __future__ import annotations

import re


def _parse_lhs(lhs: str) -> tuple[str, str, str] | None:
    """左辺文字列を (左オペランド, 演算子記号, 右オペランド) に分解する。

    「先頭の数字列 + 非数字記号列 + 末尾の数字列」という形式を想定。
    解析できない場合は None を返す。
    """
    m = re.match(r'^(\d+)([^\d]+)(\d+)$', lhs.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _rhs_to_int(rhs: str) -> int | None:
    """右辺文字列を整数に変換する。変換できない場合は None。"""
    try:
        return int(rhs)
    except ValueError:
        return None


def _parse_op_groups(
    pairs: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str, str, str]]] | None:
    """(左辺, 右辺) ペアのリストを演算子記号ごとのグループに分類する。

    各グループは (左オペランド, 演算子, 右オペランド, 右辺文字列) のリスト。
    左辺のパースに失敗した場合は None を返す。
    """
    op_groups: dict[str, list[tuple[str, str, str, str]]] = {}
    for lhs, rhs in pairs:
        parsed = _parse_lhs(lhs)
        if parsed is None:
            return None
        a_str, op, b_str = parsed
        op_groups.setdefault(op, []).append((a_str, op, b_str, rhs))
    return op_groups
