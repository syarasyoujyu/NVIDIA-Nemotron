"""数値式の推論生成器（extraction パターン準拠）。

## 推論フロー
1. 出力形式を確認し反転タイプ（全反転 / 数値反転）を決定する
2. [反転→演算→反転, 演算, 反転→交換→演算→反転] × 全演算子 × common演算 を試す
3. 2で解けなければ同じモードループ × rare+common演算 を試す
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from scripts.cot_prompt.store_types import Problem

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


# ──────────────────────────── 演算定義 ────────────────────────────

def _mod_fn(a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return hi % lo if lo != 0 else 10 ** 18


# extraction の _ARITH_OPS と同順（Phase 1 用・offset=0）
_COMMON_ARITH: list[tuple[str, Callable[[int, int], int]]] = [
    ("add",          lambda a, b: a + b),
    ("sub",          lambda a, b: a - b),
    ("mul",          lambda a, b: a * b),
    ("abs_diff",     lambda a, b: abs(a - b)),
    ("neg_abs_diff", lambda a, b: -(abs(a - b))),
]

# Phase 2 追加分（offset ±1 を組み込み済み + 桁外演算）
_RARE_ARITH: list[tuple[str, Callable[[int, int], int]]] = [
    ("add+1",   lambda a, b: a + b + 1),
    ("add-1",   lambda a, b: a + b - 1),
    ("sub+1",   lambda a, b: a - b + 1),
    ("sub-1",   lambda a, b: a - b - 1),
    ("mul+1",   lambda a, b: a * b + 1),
    ("mul-1",   lambda a, b: a * b - 1),
    ("mod_ab",  lambda a, b: a % b if b != 0 else 10 ** 18),
    ("mod_ba",  lambda a, b: b % a if a != 0 else 10 ** 18),
]


def _digit_ops(sa: str, sb: str) -> list[tuple[str, str]]:
    """2桁専用演算の候補リスト（演算名, 結果文字列）。"""
    if len(sa) != 2 or len(sb) != 2:
        return []
    d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
    return [
        ("digit_abs_diff",  str(abs(d1 - d3)) + str(abs(d2 - d4))),
        ("digit_add_mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)),
        ("digit_sub_mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)),
        ("cross_mul",       str(d1 * d3 + d2 * d4)),
        ("cross_mul_rev",   str(d1 * d4 + d2 * d3)),
        ("digit_mul",       str(d1 * d3) + str(d2 * d4)),
        ("digit_mul_rev",   str(d1 * d4) + str(d2 * d3)),
        ("digit_sum_diff",  str((d1 + d2) - (d3 + d4))),
        ("digit_sum_sum",   str((d1 + d2) + (d3 + d4))),
        ("digit_prod_diff", str(d1 * d2 - d3 * d4)),
        ("digit_prod_sum",  str(d1 * d2 + d3 * d4)),
        ("determinant",     str(d1 * d4 - d2 * d3)),
        ("abs_det",         str(abs(d1 * d4 - d2 * d3))),
    ]


# ──────────────────────────── 変換ユーティリティ ────────────────────────────

def _transform_ops(a: str, b: str, rev_in: bool, swap: bool) -> tuple[str, str] | None:
    """入力反転・交換を適用し (a', b') を返す。先頭0なら None。"""
    a_t = a[::-1] if rev_in else a
    b_t = b[::-1] if rev_in else b
    if swap:
        a_t, b_t = b_t, a_t
    if (len(a_t) > 1 and a_t.startswith("0")) or (len(b_t) > 1 and b_t.startswith("0")):
        return None
    return a_t, b_t


def _fmt_out(val: int, op_char: str, out_mode: str) -> str:
    """算術値を out_mode に従って文字列化する。"""
    if out_mode == "full_rev":
        if val >= 0:
            return str(val)[::-1]
        return (op_char + str(-val))[::-1]
    if out_mode == "num_rev":
        if val >= 0:
            return str(val)[::-1]
        return op_char + str(-val)[::-1]
    # "none"
    if val >= 0:
        return str(val)
    return op_char + str(-val)


def _mode_label(rev_in: bool, swap: bool, out_mode: str) -> str:
    parts: list[str] = []
    if rev_in:
        parts.append("反転")
    if swap:
        parts.append("交換")
    parts.append("演算")
    if out_mode == "full_rev":
        parts.append("全反転")
    elif out_mode == "num_rev":
        parts.append("数値反転")
    return " → ".join(parts)


# ──────────────────────────── 1演算子グループのテスト ────────────────────────────

@dataclass
class _MatchResult:
    op_name: str
    rev_in: bool
    swap: bool
    out_mode: str
    op_char: str
    concat_ab: bool = field(default=True)  # concat: True=a||b, False=b||a


def _test_arith(
    examples: list[tuple[str, str, str]],
    op_char: str,
    rev_in: bool, swap: bool, out_mode: str,
    op_name: str, fn: Callable[[int, int], int],
    lines: list[str], indent: str,
) -> bool:
    """算術演算1種を全例示でテストし、CoT行を追記する。True = 全一致。"""
    parts: list[str] = []
    for a, b, exp in examples:
        ops = _transform_ops(a, b, rev_in, swap)
        if ops is None:
            lines.append(f"{indent}{op_name}: leading-zero operand → skip")
            return False
        a_t, b_t = ops
        raw = fn(int(a_t), int(b_t))
        out = _fmt_out(raw, op_char, out_mode)
        ok = out == exp
        t_desc = f"{a}→{a_t},{b}→{b_t}" if rev_in else f"{a_t},{b_t}"
        parts.append(f"f({t_desc})={raw}→{out}" + ("✓" if ok else f"✗(exp:{exp})"))
        if not ok:
            lines.append(f"{indent}{op_name}: {', '.join(parts)} → wrong")
            return False
    lines.append(f"{indent}{op_name}: {', '.join(parts)} → match")
    return True


def _test_concat(
    examples: list[tuple[str, str, str]],
    rev_in: bool, swap: bool, out_mode: str,
    strip: bool,
    lines: list[str], indent: str,
) -> tuple[bool, bool]:
    """連結演算をテストする。(成否, ab順かどうか) を返す。"""
    op_name = "concat_strip" if strip else "concat"
    parts: list[str] = []
    ab_order = True
    for idx, (a, b, exp) in enumerate(examples):
        a_t = a[::-1] if rev_in else a
        b_t = b[::-1] if rev_in else b
        if swap:
            a_t, b_t = b_t, a_t
        matched_out: str | None = None
        for use_ab, cat in [(True, a_t + b_t), (False, b_t + a_t)]:
            s = cat.lstrip("0") or "0" if strip else cat
            out = s[::-1] if out_mode != "none" else s
            if out == exp:
                matched_out = out
                if idx == 0:
                    ab_order = use_ab
                break
        if matched_out is None:
            cat0 = a_t + b_t
            s0 = cat0.lstrip("0") or "0" if strip else cat0
            out0 = s0[::-1] if out_mode != "none" else s0
            parts.append(f"f({a_t}||{b_t})→{out0}✗(exp:{exp})")
            lines.append(f"{indent}{op_name}: {', '.join(parts)} → wrong")
            return False, True
        parts.append(f"f({a_t}||{b_t})→{matched_out}✓")
    lines.append(f"{indent}{op_name}: {', '.join(parts)} → match")
    return True, ab_order


def _test_digit_ops_group(
    examples: list[tuple[str, str, str]],
    rev_in: bool, swap: bool,
    lines: list[str], indent: str,
) -> str | None:
    """2桁専用演算を全例示でテストし、マッチした演算名を返す（None = 不一致）。"""
    if not all(len(a) == 2 and len(b) == 2 for a, b, _ in examples):
        return None
    a0, b0, exp0 = examples[0]
    a_t0 = a0[::-1] if rev_in else a0
    b_t0 = b0[::-1] if rev_in else b0
    if swap:
        a_t0, b_t0 = b_t0, a_t0
    candidates = [name for name, val in _digit_ops(a_t0, b_t0) if val == exp0]
    for name in candidates:
        parts: list[str] = []
        ok_all = True
        for a, b, exp in examples:
            a_t = a[::-1] if rev_in else a
            b_t = b[::-1] if rev_in else b
            if swap:
                a_t, b_t = b_t, a_t
            d_map = dict(_digit_ops(a_t, b_t))
            out = d_map.get(name, "?")
            ok = out == exp
            parts.append(f"f({a_t},{b_t})={out}" + ("✓" if ok else f"✗(exp:{exp})"))
            if not ok:
                ok_all = False
                break
        status = "match" if ok_all else "wrong"
        lines.append(f"{indent}{name}: {', '.join(parts)} → {status}")
        if ok_all:
            return name
    return None


# ──────────────────────────── モード探索フェーズ ────────────────────────────

def _search_phase(
    by_op: dict[str, list[tuple[str, str, str]]],
    modes: list[tuple[bool, bool, str]],
    phase_label: str,
    use_rare: bool,
    lines: list[str],
) -> dict[str, _MatchResult] | None:
    """指定モード × 全演算子 × 演算を試し、全演算子が一致するモードを返す。"""
    arith_ops = _COMMON_ARITH + (_RARE_ARITH if use_rare else [])

    lines.append("")
    lines.append("=" * 56)
    lines.append(phase_label)
    lines.append("=" * 56)

    for rev_in, swap, out_mode in modes:
        label = _mode_label(rev_in, swap, out_mode)
        lines.append("")
        lines.append(f"[Mode: {label}]")

        found_per_op: dict[str, _MatchResult] = {}
        mode_ok = True

        for op_char in sorted(by_op.keys()):
            group = by_op[op_char]
            ex_str = ", ".join(f"{a}{op_char}{b}={out}" for a, b, out in group)
            lines.append(f"  Operator 【{op_char}】: {ex_str}")
            indent = "    "

            matched: str | None = None
            concat_ab = True

            # 算術演算
            for op_name, fn in arith_ops:
                if _test_arith(group, op_char, rev_in, swap, out_mode, op_name, fn, lines, indent):
                    matched = op_name
                    break

            # 連結演算
            if matched is None:
                for strip in (False, True):
                    ok, ab = _test_concat(group, rev_in, swap, out_mode, strip, lines, indent)
                    if ok:
                        matched = "concat_strip" if strip else "concat"
                        concat_ab = ab
                        break

            # 2桁専用演算（rare フェーズのみ、出力変換なし）
            if matched is None and use_rare and out_mode == "none":
                matched = _test_digit_ops_group(group, rev_in, swap, lines, indent)

            if matched is None:
                lines.append(f"  → 【{op_char}】: no match → mode failed")
                mode_ok = False
                break

            lines.append(f"  → 【{op_char}】: matched with 【{matched}】")
            found_per_op[op_char] = _MatchResult(
                op_name=matched,
                rev_in=rev_in, swap=swap, out_mode=out_mode,
                op_char=op_char, concat_ab=concat_ab,
            )

        if mode_ok and len(found_per_op) == len(by_op):
            lines.append(f"→ All operators matched under mode 【{label}】!")
            return found_per_op
        elif mode_ok:
            lines.append(f"→ Mode 【{label}】: failed (not all operators covered)")

    lines.append("→ No solution found in this phase.")
    return None


# ──────────────────────────── 解の適用 ────────────────────────────

def _apply_match(match: _MatchResult, a_str: str, b_str: str) -> tuple[str, list[str]]:
    steps: list[str] = []
    ops = _transform_ops(a_str, b_str, match.rev_in, match.swap)
    if ops is None:
        return "?", ["leading zero in input → cannot compute"]
    a_t, b_t = ops

    if match.rev_in:
        steps.append(f"Reverse inputs: {a_str}→{a_t}, {b_str}→{b_t}")
    if match.swap:
        steps.append(f"Swap: a={a_t}, b={b_t}")

    op = match.op_name

    if op in ("concat", "concat_strip"):
        cat = (a_t + b_t) if match.concat_ab else (b_t + a_t)
        if op == "concat_strip":
            cat = cat.lstrip("0") or "0"
        result = cat[::-1] if match.out_mode != "none" else cat
        steps.append(f"{op}: {a_t}||{b_t} = {cat} → {result}")
        return result, steps

    # 2桁専用
    d_map = dict(_digit_ops(a_t, b_t))
    if op in d_map:
        result = d_map[op]
        steps.append(f"{op}: f({a_t},{b_t}) = {result}")
        return result, steps

    # 算術
    fn_map = dict(_COMMON_ARITH + _RARE_ARITH)
    fn = fn_map.get(op)
    if fn is None:
        return "?", [f"unknown operation: {op}"]
    raw = fn(int(a_t), int(b_t))
    out = _fmt_out(raw, match.op_char, match.out_mode)
    steps.append(f"{op}: f({a_t},{b_t}) = {raw}")
    if match.out_mode != "none":
        rev_label = "全反転" if match.out_mode == "full_rev" else "数値反転"
        steps.append(f"Apply {rev_label}: {raw} → {out}")
    else:
        steps.append(f"Output: {out}")
    return out, steps


# ──────────────────────────── メイン ────────────────────────────

def reasoning_equation_numeric(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples:")

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))
        lines.append(f"  {ex.input_value} = {ex.output_value}")

    if not parsed:
        return None

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    q_match = _EXPR_RE.fullmatch(str(problem.question))
    q_op = q_match.group(2) if q_match else None

    # ──── Step 1: 出力形式の確認（全反転 vs 数値反転） ────
    lines.append("")
    lines.append("Step 1: Analyze outputs to determine reversal encoding")
    all_outs = [out for _, _, _, out in parsed]
    lines.append(f"All outputs: {', '.join(all_outs)}")

    # 演算子記号が出力末尾にある → 全反転（(op+abs_val) を文字列ごと反転）
    # 出力末尾にない         → 数値反転（桁のみ反転、負は op + reversed_digits）
    has_suffix = False
    suffix_example = ""
    for op_char, group in by_op.items():
        for a, b, out in group:
            if len(out) > 1 and out[-1] == op_char:
                has_suffix = True
                suffix_example = f"{a}{op_char}{b}={out}"
                break
        if has_suffix:
            break

    if has_suffix:
        lines.append(f"Output ending with operator symbol found: {suffix_example}")
        lines.append("→ Reversal (if any) uses full reversal.")
        lines.append("  Negative encoding: (op + abs_digits) reversed as a whole string.")
        out_mode = "full_rev"
    else:
        lines.append("No output ends with an operator symbol.")
        lines.append("→ Reversal (if any) uses numeric reversal.")
        lines.append("  Negative encoding: op_char + reversed_abs_digits.")
        out_mode = "num_rev"

    # モードリスト: [反転→演算→(全/数値)反転, 演算, 反転→交換→演算→(全/数値)反転]
    modes: list[tuple[bool, bool, str]] = [
        (False, False, "none"),
        (True,  False, out_mode),
        (True,  True,  out_mode),
    ]

    # ──── Step 2: common演算 × 全モード × 全演算子 ────
    result = _search_phase(by_op, modes, "Step 2: Common operations (offset=0)", False, lines)

    # ──── Step 3: rare+common演算 × 全モード × 全演算子 ────
    if result is None:
        result = _search_phase(by_op, modes, "Step 3: All operations (rare + offset)", True, lines)

    if result is None:
        return None

    # ──── 質問に適用 ────
    if q_match is None or q_op is None:
        return None

    qa, qb = q_match.group(1), q_match.group(3)
    lines.append("")
    lines.append(f"Applying to {problem.question}:")

    if q_op in result:
        found = result[q_op]
    else:
        # 質問演算子が例示にない → 最初の演算子の設定で abs_diff を使用
        first = next(iter(result.values()))
        lines.append(
            f"  Operator 【{q_op}】 not in examples. "
            f"Using abs_diff with mode 【{_mode_label(first.rev_in, first.swap, first.out_mode)}】."
        )
        found = _MatchResult(
            op_name="abs_diff",
            rev_in=first.rev_in, swap=first.swap, out_mode=first.out_mode,
            op_char=q_op,
        )

    result_val, steps = _apply_match(found, qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: 【{result_val}】")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)
