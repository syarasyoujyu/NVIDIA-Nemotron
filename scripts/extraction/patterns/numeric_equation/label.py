"""フローラベル生成ユーティリティ。

各演算子グループに対して決定したルールを
人間が読める日本語の「フロー文字列」に変換する。

例: (op_name="sub", offset=-1, rev_in=True, swap=False, out_mode="num_rev")
    → "反転 → 引き算 − 1 → 数値反転"
"""
from __future__ import annotations

from .constants import _OP_LABELS_JP


def _offset_str(offset: int) -> str:
    """オフセットを文字列に変換する。0 の場合は空文字。"""
    if offset == 0:
        return ""
    return f" + {offset}" if offset > 0 else f" − {-offset}"


def _flow_label(op_name: str, offset: int, rev_in: bool, swap: bool, out_mode: str) -> str:
    """演算ルールを日本語フローラベルに変換する。

    ステップを「反転 → 交換 → 演算(±offset) → 出力変換」の順に連結する。
    該当するステップのみ出力し、矢印で区切る。
    """
    parts = []
    if rev_in:
        parts.append("反転")
    if swap:
        parts.append("交換")
    parts.append(_OP_LABELS_JP[op_name] + _offset_str(offset))
    if out_mode in ("num_rev", "num_rev_sfx"):
        parts.append("数値反転")
    elif out_mode == "full_rev":
        parts.append("全反転")
    return " → ".join(parts)


def _flow_sort_key(flow: str) -> tuple[int, int, int, str]:
    """別解表示用のフロー優先順位を返す。"""
    arrow_count = flow.count("→")
    parenthetical_count = flow.count("（")
    full_reverse_last = 1 if flow.endswith("全反転") else 0
    return (arrow_count, parenthetical_count, full_reverse_last, flow)
