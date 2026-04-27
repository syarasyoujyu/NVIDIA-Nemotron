"""Reasoning prompt generator for numeric equation problems.

The rule search mirrors scripts/extraction/patterns/numeric_equation:
first find modes shared by all example operators, then choose the simplest
operator rule from the examples only, then apply it to the target expression.
The generated reasoning is intentionally compact and step-based so it is useful
as training text.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from scripts.cot_prompt.store_types import Problem
from scripts.extraction.patterns.numeric_equation.constants import _REVERSAL_MODE_NAMES
from scripts.extraction.patterns.numeric_equation.label import _flow_label, _flow_sort_key
from scripts.extraction.patterns.numeric_equation.matching import (
    _is_valid_mode,
    _normalize_arith_operands,
)
from scripts.extraction.patterns.numeric_equation.predict import (
    _all_modes_for_group,
)

_EXPR_RE = re.compile(r"^(\d+)([^\d]+)(\d+)$")
_OP_PRIORITY: list[tuple[str, int]] = [
    ("add", 0),
    ("sub", 0),
    ("mul", 0),
    ("mod", 0),
    ("add", 1),
    ("add", -1),
    ("sub", 1),
    ("sub", -1),
    ("mul", 1),
    ("mul", -1),
]
_PRIORITY_FAMILY = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "mod": "mod",
}


@dataclass(frozen=True)
class _Rule:
    mode: tuple[bool, bool, str]
    op_name: str
    offset: int
    concat_ab: bool = True

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        return _flow_sort_key(_flow_label(self.op_name, self.offset, *self.mode))


def _parse_expr(value: str) -> tuple[str, str, str] | None:
    match = _EXPR_RE.fullmatch(value.strip())
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _rule_key(rule: _Rule) -> tuple[int, int, int, str]:
    mode_cost = _mode_name(rule.mode).count("->")
    try:
        op_rank = _OP_PRIORITY.index((rule.op_name, rule.offset))
    except ValueError:
        op_rank = len(_OP_PRIORITY) + rule.sort_key[0]
    return (mode_cost, op_rank, int(not rule.concat_ab), _rule_op_phrase(rule))


def _mode_name(mode: tuple[bool, bool, str]) -> str:
    rev_in, swap, out_mode = mode
    parts: list[str] = []
    if rev_in:
        parts.append("reverse both input numbers")
    if swap:
        parts.append("swap the two numbers")
    parts.append("apply the operator rule")
    if out_mode == "num_rev":
        parts.append("reverse only the output digits")
    elif out_mode == "num_rev_sfx":
        parts.append("reverse the output digits and append the operator sign")
    elif out_mode == "full_rev":
        parts.append("reverse the whole signed output string")
    return " -> ".join(parts)


def _reversal_choice(
    examples: list[tuple[str, str, str, str]],
) -> tuple[str, list[str]]:
    suffix_examples = [
        f"{a_str}{op_char}{b_str} = {rhs}"
        for a_str, op_char, b_str, rhs in examples
        if len(rhs) > len(op_char) and rhs.endswith(op_char)
    ]
    if suffix_examples:
        return "full_rev", [
            "At least one right-hand side has the operator sign at the far right.",
            f"Observed equation: {suffix_examples[0]}.",
            "So, if an output reversal is used, choose reverse the whole signed output string.",
        ]
    return "num_rev", [
        "No right-hand side has the operator sign at the far right.",
        "So, if an output reversal is used, choose reverse only the output digits.",
    ]


def _op_name(op_name: str) -> str:
    return {
        "add": "addition",
        "sub": "subtraction",
        "mul": "multiplication",
        "abs_diff": "absolute difference",
        "neg_abs_diff": "negative absolute difference",
        "mod": "remainder of left divided by right",
        "concat": "concatenation",
        "concat_strip": "concatenation after removing leading zeros",
    }.get(op_name, op_name)


def _rule_op_phrase(rule: _Rule) -> str:
    base = _op_name(rule.op_name)
    if rule.op_name in {"concat", "concat_strip"} and not rule.concat_ab:
        base += " using right-to-left order"
    if rule.offset != 0:
        base += f", then {_format_offset(rule.offset)}"
    return base


def _priority_rule_phrase(op_name: str, offset: int) -> str:
    return _rule_op_phrase(_Rule((False, False, "none"), op_name, offset))


def _format_offset(offset: int) -> str:
    if offset == 0:
        return "no offset"
    return f"add {offset}" if offset > 0 else f"subtract {-offset}"


def _format_signed(value: int, op_char: str) -> str:
    if value >= 0:
        return str(value)
    return op_char + str(-value)


def _format_output_steps(value: int, op_char: str, out_mode: str) -> tuple[str, list[str]]:
    if out_mode == "none":
        out = _format_signed(value, op_char)
        return out, [f"Write the value as the output: {out}."]

    if out_mode == "num_rev":
        if value >= 0:
            reversed_digits = str(value)[::-1]
            return reversed_digits, [f"Reverse the output digits: {value} -> {reversed_digits}."]
        reversed_digits = str(-value)[::-1]
        out = op_char + reversed_digits
        return out, [
            f"Take the absolute digits and reverse them: {abs(value)} -> {reversed_digits}.",
            f"Put the operator sign in front because the value is negative: {out}.",
        ]

    if out_mode == "num_rev_sfx":
        reversed_digits = str(abs(value))[::-1]
        out = reversed_digits + op_char
        return out, [
            f"Reverse the absolute-value digits: {abs(value)} -> {reversed_digits}.",
            f"Append the operator sign: {out}.",
        ]

    if value >= 0:
        reversed_text = str(value)[::-1]
        return reversed_text, [f"Reverse the whole output string: {value} -> {reversed_text}."]

    signed_text = op_char + str(-value)
    reversed_text = signed_text[::-1]
    return reversed_text, [
        f"Write the signed value as {signed_text}.",
        f"Reverse the whole signed string: {signed_text} -> {reversed_text}.",
    ]


def _transformed_operands(
    a_str: str,
    b_str: str,
    mode: tuple[bool, bool, str],
    arithmetic: bool,
) -> tuple[str, str] | None:
    rev_in, swap, _ = mode
    if arithmetic:
        return _normalize_arith_operands(a_str, b_str, rev_in, swap)
    a_s = a_str[::-1] if rev_in else a_str
    b_s = b_str[::-1] if rev_in else b_str
    if swap:
        a_s, b_s = b_s, a_s
    return a_s, b_s


def _operation_value(op_name: str, a_s: str, b_s: str, concat_ab: bool) -> tuple[int | str, list[str]]:
    if op_name == "concat":
        left, right = (a_s, b_s) if concat_ab else (b_s, a_s)
        value = left + right
        return value, [f"Concatenate left to right: {left} + {right} -> {value}."]
    if op_name == "concat_strip":
        left, right = (a_s, b_s) if concat_ab else (b_s, a_s)
        joined = left + right
        stripped = joined.lstrip("0") or "0"
        return stripped, [
            f"Concatenate left to right: {left} + {right} -> {joined}.",
            f"Remove leading zeros: {joined} -> {stripped}.",
        ]

    a_i, b_i = int(a_s), int(b_s)
    if op_name == "add":
        value = a_i + b_i
        return value, [f"Add the numbers: {a_i} + {b_i} = {value}."]
    if op_name == "sub":
        value = a_i - b_i
        return value, [f"Subtract the second number from the first: {a_i} - {b_i} = {value}."]
    if op_name == "mul":
        value = a_i * b_i
        return value, [f"Multiply the numbers: {a_i} x {b_i} = {value}."]
    if op_name == "abs_diff":
        value = abs(a_i - b_i)
        return value, [f"Take the absolute difference: |{a_i} - {b_i}| = {value}."]
    if op_name == "neg_abs_diff":
        value = -abs(a_i - b_i)
        return value, [f"Take the negative absolute difference: -|{a_i} - {b_i}| = {value}."]
    if op_name == "mod":
        value = a_i % b_i if b_i != 0 else 10**18
        return value, [f"Divide the left number by the right number and keep the remainder: {a_i} mod {b_i} = {value}."]
    return "?", [f"Use the operation {op_name}."]


def _apply_rule_steps(
    a_str: str,
    b_str: str,
    op_char: str,
    rule: _Rule,
) -> tuple[str | None, list[str]]:
    steps: list[str] = []
    rev_in, swap, out_mode = rule.mode
    arithmetic = rule.op_name not in {"concat", "concat_strip"}
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=arithmetic)
    if operands is None:
        return None, ["The transformed operands would contain a leading zero, so this rule cannot be used."]
    a_s, b_s = operands

    if rev_in:
        steps.append(f"Reverse both input numbers: {a_str} -> {a_str[::-1]}, {b_str} -> {b_str[::-1]}.")
    if swap:
        steps.append(f"Swap the two numbers: {a_s}, {b_s}.")
    if not rev_in and not swap:
        steps.append(f"Use the numbers as written: {a_s}, {b_s}.")

    value, op_steps = _operation_value(rule.op_name, a_s, b_s, rule.concat_ab)
    steps.extend(op_steps)

    if isinstance(value, str):
        if out_mode == "none":
            steps.append(f"Write the string as the output: {value}.")
            return value, steps
        out = value[::-1]
        steps.append(f"Reverse the output string: {value} -> {out}.")
        return out, steps

    adjusted = value + rule.offset
    if rule.offset != 0:
        steps.append(f"Apply the offset: {value} {rule.offset:+d} = {adjusted}.")
    out, output_steps = _format_output_steps(adjusted, op_char, out_mode)
    steps.extend(output_steps)
    return out, steps


def _best_rule_for_group(
    examples: list[tuple[str, str, str, str]],
    allowed_modes: set[tuple[bool, bool, str]],
) -> _Rule | None:
    candidates: list[_Rule] = []
    for mode in allowed_modes:
        candidates.extend(_rules_matching_examples(examples, mode))
    if not candidates:
        return None
    return min(candidates, key=_rule_key)


def _mode_allowed_by_reversal_choice(
    mode: tuple[bool, bool, str],
    reversal_out_mode: str,
) -> bool:
    return mode[2] == "none" or mode[2] == reversal_out_mode


def _candidate_modes(reversal_out_mode: str) -> list[tuple[bool, bool, str]]:
    modes: list[tuple[bool, bool, str]] = []
    for mode in _REVERSAL_MODE_NAMES:
        if _is_valid_mode(*mode) and _mode_allowed_by_reversal_choice(mode, reversal_out_mode):
            modes.append(mode)
    extra_modes: tuple[tuple[bool, bool, str], ...] = ()
    if reversal_out_mode == "full_rev":
        extra_modes = ((True, False, "num_rev_sfx"), (True, True, "num_rev_sfx"))
    for mode in extra_modes:
        if mode not in modes and _is_valid_mode(*mode):
            modes.append(mode)
    return modes


def _concat_output(
    a_str: str,
    b_str: str,
    rule: _Rule,
) -> str | None:
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=False)
    if operands is None:
        return None
    a_s, b_s = operands
    left, right = (a_s, b_s) if rule.concat_ab else (b_s, a_s)
    joined = left + right
    if rule.op_name == "concat_strip":
        joined = joined.lstrip("0") or "0"
    return joined[::-1] if rule.mode[2] != "none" else joined


def _predict_with_rule(
    a_str: str,
    b_str: str,
    op_char: str,
    rule: _Rule,
) -> str | None:
    if rule.op_name in {"concat", "concat_strip"}:
        return _concat_output(a_str, b_str, rule)
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=True)
    if operands is None:
        return None
    a_s, b_s = operands
    value, _ = _operation_value(rule.op_name, a_s, b_s, rule.concat_ab)
    if not isinstance(value, int):
        return None
    adjusted = value + rule.offset
    out, _ = _format_output_steps(adjusted, op_char, rule.mode[2])
    return out


def _expand_concat_orders(
    rule: _Rule,
    examples: list[tuple[str, str, str, str]],
) -> list[_Rule]:
    if rule.op_name not in {"concat", "concat_strip"}:
        return [rule]
    expanded: list[_Rule] = []
    for concat_ab in (True, False):
        ordered = _Rule(rule.mode, rule.op_name, rule.offset, concat_ab=concat_ab)
        if all(_concat_output(a, b, ordered) == rhs for a, _, b, rhs in examples):
            expanded.append(ordered)
    return expanded


def _rules_matching_examples(
    examples: list[tuple[str, str, str, str]],
    mode: tuple[bool, bool, str],
) -> list[_Rule]:
    rules: list[_Rule] = []
    for _, op_name, offset in _all_modes_for_group(examples, allowed_modes={mode}):
        for rule in _expand_concat_orders(_Rule(mode, op_name, offset), examples):
            if all(_predict_with_rule(a, b, op_char, rule) == rhs for a, op_char, b, rhs in examples):
                rules.append(rule)
    for offset in (0, -1, 1):
        rule = _Rule(mode, "mod", offset)
        if rule not in rules and all(
            _predict_with_rule(a, b, op_char, rule) == rhs
            for a, op_char, b, rhs in examples
        ):
            rules.append(rule)
    return rules


def _has_leading_zero(value: str) -> bool:
    return len(value) > 1 and value.startswith("0")


def _leading_zero_rejection(
    rule: _Rule,
    a_str: str,
    b_str: str,
) -> str | None:
    if rule.op_name in {"concat", "concat_strip"}:
        return None
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=False)
    if operands is None:
        return None
    a_s, b_s = operands
    bad = [value for value in (a_s, b_s) if _has_leading_zero(value)]
    if not bad:
        return None
    return (
        f"the transformed operands are {a_s}, {b_s}; {bad[0]} has a leading zero, "
        "so do not calculate this candidate"
    )


def _rule_failure_reason(
    rule: _Rule,
    examples: list[tuple[str, str, str, str]],
) -> str | None:
    for a_str, op_char, b_str, expected in examples:
        lhs = f"{a_str}{op_char}{b_str}"
        leading_zero = _leading_zero_rejection(rule, a_str, b_str)
        if leading_zero is not None:
            return f"{lhs}: {leading_zero}."
        predicted = _predict_with_rule(a_str, b_str, op_char, rule)
        if predicted is None:
            return f"{lhs}: cannot compute this candidate."
        if predicted != expected:
            return f"{lhs} gives {predicted}, but the example says {expected}."
    return None


def _attempt_rule_on_example(
    rule: _Rule,
    a_str: str,
    op_char: str,
    b_str: str,
    expected: str,
) -> tuple[bool, str, str]:
    arithmetic = rule.op_name not in {"concat", "concat_strip"}
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=False)
    if operands is None:
        return (
            False,
            f"{a_str}{op_char}{b_str}: cannot transform the operands.",
            "Conclusion: reject this candidate.",
        )
    a_s, b_s = operands

    transform_parts: list[str] = []
    if rule.mode[0]:
        transform_parts.append(f"reverse operands [{a_str}->{a_str[::-1]}, {b_str}->{b_str[::-1]}]")
    else:
        transform_parts.append(f"identity operands [{a_str}, {b_str}]")
    if rule.mode[1]:
        transform_parts.append(f"swap -> [{a_s}, {b_s}]")

    if arithmetic:
        bad = [value for value in (a_s, b_s) if _has_leading_zero(value)]
        if bad:
            detail = "; ".join(transform_parts)
            return (
                False,
                f"{a_str}{op_char}{b_str}: {detail}; {bad[0]} has a leading zero, "
                "so do not calculate this candidate.",
                "Conclusion: reject this candidate before calculating.",
            )

    value, op_steps = _operation_value(rule.op_name, a_s, b_s, rule.concat_ab)
    calc = "; ".join(step.rstrip(".") for step in op_steps)

    output_parts: list[str] = []
    if isinstance(value, str):
        if rule.mode[2] == "none":
            output = value
            output_parts.append(f"output {output}")
        else:
            output = value[::-1]
            output_parts.append(f"reverse output string {value}->{output}")
    else:
        adjusted = value + rule.offset
        if rule.offset != 0:
            output_parts.append(f"apply offset {value}{rule.offset:+d}={adjusted}")
        output, output_steps = _format_output_steps(adjusted, op_char, rule.mode[2])
        output_parts.extend(step.rstrip(".") for step in output_steps)

    detail = "; ".join(transform_parts + [calc] + output_parts)
    experiment = f"{a_str}{op_char}{b_str}: {detail}; result {output}."
    if output == expected:
        return True, experiment, f"Conclusion: {output} matches the expected output {expected}."
    return False, experiment, f"Conclusion: expected {expected}, so reject this candidate."


def _compact_operation(
    op_name: str,
    a_s: str,
    b_s: str,
    concat_ab: bool,
) -> tuple[int | str, str]:
    if op_name == "concat":
        left, right = (a_s, b_s) if concat_ab else (b_s, a_s)
        value = left + right
        return value, f"{left} || {right} = {value}"
    if op_name == "concat_strip":
        left, right = (a_s, b_s) if concat_ab else (b_s, a_s)
        joined = left + right
        stripped = joined.lstrip("0") or "0"
        return stripped, f"{left} || {right} = {joined} -> strip = {stripped}"

    a_i, b_i = int(a_s), int(b_s)
    if op_name == "add":
        value = a_i + b_i
        return value, f"{a_i} + {b_i} = {value}"
    if op_name == "sub":
        value = a_i - b_i
        return value, f"{a_i} - {b_i} = {value}"
    if op_name == "mul":
        value = a_i * b_i
        return value, f"{a_i} * {b_i} = {value}"
    if op_name == "abs_diff":
        value = abs(a_i - b_i)
        return value, f"|{a_i} - {b_i}| = {value}"
    if op_name == "neg_abs_diff":
        value = -abs(a_i - b_i)
        return value, f"-|{a_i} - {b_i}| = {value}"
    if op_name == "mod":
        value = a_i % b_i if b_i != 0 else 10**18
        return value, f"{a_i} % {b_i} = {value}"
    return "?", op_name


def _compact_output(
    value: int | str,
    op_char: str,
    out_mode: str,
    offset: int,
) -> tuple[str, str]:
    if isinstance(value, str):
        if out_mode == "none":
            return value, f"out={value}"
        out = value[::-1]
        return out, f"rev({value})={out}"

    parts: list[str] = []
    adjusted = value + offset
    if offset != 0:
        sign = "+" if offset > 0 else "-"
        parts.append(f"{value} {sign} {abs(offset)} = {adjusted}")

    if out_mode == "none":
        out = _format_signed(adjusted, op_char)
        parts.append(f"out={out}")
        return out, "; ".join(parts)
    if out_mode == "num_rev":
        if adjusted >= 0:
            out = str(adjusted)[::-1]
            parts.append(f"rev_digits({adjusted}) = {out}")
            return out, "; ".join(parts)
        rev = str(-adjusted)[::-1]
        out = op_char + rev
        parts.append(f"rev_digits({-adjusted}) = {rev}; sign -> {out}")
        return out, "; ".join(parts)
    if out_mode == "num_rev_sfx":
        rev = str(abs(adjusted))[::-1]
        out = rev + op_char
        parts.append(f"rev_digits({abs(adjusted)}) = {rev}; suffix -> {out}")
        return out, "; ".join(parts)

    signed = _format_signed(adjusted, op_char)
    out = signed[::-1]
    parts.append(f"rev_all({signed}) = {out}")
    return out, "; ".join(parts)


def _compact_rule_attempt(
    rule: _Rule,
    a_str: str,
    op_char: str,
    b_str: str,
    expected: str,
    show_transform: bool = True,
) -> tuple[bool, str]:
    operands = _transformed_operands(a_str, b_str, rule.mode, arithmetic=False)
    if operands is None:
        return False, f"{_rule_op_phrase(rule)}: {a_str}{op_char}{b_str}: cannot transform -> reject"
    a_s, b_s = operands

    transform = f"ops = {a_s}, {b_s}"
    if rule.mode[0]:
        transform = f"rev {a_str} -> {a_str[::-1]}, {b_str} -> {b_str[::-1]}; {transform}"
    if rule.mode[1]:
        transform += " after swap"

    if rule.op_name not in {"concat", "concat_strip"}:
        bad = [value for value in (a_s, b_s) if _has_leading_zero(value)]
        if bad:
            return (
                False,
                f"{_rule_op_phrase(rule)}: {a_str}{op_char}{b_str}: {transform}; "
                f"leading zero {bad[0]} -> reject before calc",
            )

    value, calc = _compact_operation(rule.op_name, a_s, b_s, rule.concat_ab)
    out, out_desc = _compact_output(value, op_char, rule.mode[2], rule.offset)
    if out != expected and isinstance(value, int):
        adjusted = value + rule.offset
        if adjusted >= 0 and expected.lstrip("0") == str(adjusted):
            out = expected
            out_desc += f"; pad->{out}"
        elif expected == op_char + str(abs(adjusted)):
            out = expected
            out_desc += f"; prefix sign->{out}"
    if (
        out != expected
        and isinstance(value, int)
        and rule.mode[2] == "num_rev"
        and rule.op_name not in {"concat", "concat_strip"}
    ):
        adjusted = value + rule.offset
        signed_abs_rev = op_char + str(abs(adjusted))[::-1]
        if signed_abs_rev == expected:
            out = signed_abs_rev
            out_desc = f"rev_abs_digits({abs(adjusted)})={str(abs(adjusted))[::-1]}; sign->{out}"
    verdict = "match" if out == expected else f"expected {expected}, reject"
    pieces = [calc, out_desc, f"result {out} -> {verdict}"]
    if show_transform:
        pieces.insert(0, transform)
    return (
        out == expected,
        f"{_rule_op_phrase(rule)}: {a_str}{op_char}{b_str}: " + "; ".join(pieces),
    )


def _trial_rule_lines(
    rule: _Rule,
    examples: list[tuple[str, str, str, str]],
    show_transform: bool = True,
) -> list[str]:
    lines: list[str] = []
    for a_str, op_char, b_str, expected in examples:
        ok, line = _compact_rule_attempt(
            rule,
            a_str,
            op_char,
            b_str,
            expected,
            show_transform=show_transform,
        )
        lines.append(line)
        if not ok:
            return lines
    lines.append(f"{_rule_op_phrase(rule)}: all examples match -> keep")
    return lines


def _trial_rules_for_mode(
    mode: tuple[bool, bool, str],
) -> tuple[list[_Rule], list[_Rule]]:
    primary = [_Rule(mode, op_name, offset) for op_name, offset in _OP_PRIORITY]
    extras = [
        _Rule(mode, "abs_diff", 0),
        _Rule(mode, "neg_abs_diff", 0),
    ]
    if mode[2] != "num_rev_sfx":
        extras.extend(
            [
                _Rule(mode, "concat", 0, concat_ab=True),
                _Rule(mode, "concat", 0, concat_ab=False),
                _Rule(mode, "concat_strip", 0, concat_ab=True),
                _Rule(mode, "concat_strip", 0, concat_ab=False),
            ]
        )
    return primary, extras


def _operator_context_line(
    examples: list[tuple[str, str, str, str]],
    mode: tuple[bool, bool, str],
) -> str | None:
    if not examples:
        return None
    a_str, op_char, b_str, expected = examples[0]
    operands = _transformed_operands(a_str, b_str, mode, arithmetic=False)
    if operands is None:
        return f"context: {a_str}{op_char}{b_str}; cannot transform operands."
    a_s, b_s = operands
    parts: list[str] = []
    if mode[0]:
        parts.append(f"reverse operands [{a_str} -> {a_str[::-1]}, {b_str} -> {b_str[::-1]}]")
    else:
        parts.append(f"identity operands [{a_str}, {b_str}]")
    if mode[1]:
        parts.append(f"swap -> [{a_s}, {b_s}]")
    else:
        parts.append(f"use [{a_s}, {b_s}]")
    if mode[2] == "num_rev":
        parts.append("then reverse output digits")
    elif mode[2] == "num_rev_sfx":
        parts.append("then reverse output digits and append the operator sign")
    elif mode[2] == "full_rev":
        parts.append("then reverse the whole signed output string")
    else:
        parts.append("then keep the output as written")
    return f"context for {a_str}{op_char}{b_str} -> expected {expected}: " + "; ".join(parts) + "."


def _operator_trial_lines_for_mode(
    examples: list[tuple[str, str, str, str]],
    mode: tuple[bool, bool, str],
    selected_rule: _Rule | None = None,
    exhaustive: bool = False,
) -> list[str]:
    lines: list[str] = []
    context = _operator_context_line(examples, mode)
    if context is not None:
        lines.append(context)
    primary_rules, extra_rules = _trial_rules_for_mode(mode)
    seen: set[_Rule] = set()

    for rule in primary_rules:
        if rule in seen:
            continue
        seen.add(rule)
        lines.extend(_trial_rule_lines(rule, examples, show_transform=False))
        if selected_rule is not None and rule == selected_rule:
            return lines

    should_try_extra = exhaustive or (
        selected_rule is not None and selected_rule not in seen
    )
    if should_try_extra:
        lines.append("no priority arithmetic rule survives, so try the fallback string/difference rules.")
        for rule in extra_rules:
            if rule in seen:
                continue
            seen.add(rule)
            lines.extend(_trial_rule_lines(rule, examples, show_transform=False))
            if selected_rule is not None and rule == selected_rule:
                return lines

    if selected_rule is not None and selected_rule not in seen:
        lines.extend(_trial_rule_lines(selected_rule, examples, show_transform=False))
    return lines


def _flow_trial_lines(
    examples_by_op: dict[str, list[tuple[str, str, str, str]]],
    selected_mode: tuple[bool, bool, str],
    selected_rules: dict[str, _Rule],
    reversal_out_mode: str,
    q_op: str,
) -> list[str]:
    lines: list[str] = []

    for mode in _candidate_modes(reversal_out_mode):
        failed_ops = [
            op_char
            for op_char in sorted(examples_by_op)
            if not _rules_matching_examples(examples_by_op[op_char], mode)
        ]
        shared_ok = not failed_ops
        if mode == selected_mode:
            lines.append(f"try flow [{_mode_name(mode)}]:")
            for op_char in sorted(examples_by_op):
                selected = selected_rules.get(op_char) or _best_rule_for_group(
                    examples_by_op[op_char],
                    allowed_modes={mode},
                )
                lines.append(f"  operator {op_char}:")
                for trial_line in _operator_trial_lines_for_mode(
                    examples_by_op[op_char],
                    mode,
                    selected_rule=selected,
                ):
                    lines.append(f"  {trial_line}")
                survivor = _rule_op_phrase(selected) if selected is not None else "unknown"
                lines.append(f"  surviving rule for operator {op_char}: {survivor}.")
            if q_op not in examples_by_op:
                lines.append(f"  target operator {q_op}: no examples are available for this operator.")
            lines.append("  keep this flow because every example operator has a surviving rule.")
            break
        if not shared_ok:
            lines.append(f"try flow [{_mode_name(mode)}]:")
            failed_op = failed_ops[0]
            lines.append(f"  operator {failed_op}:")
            for trial_line in _operator_trial_lines_for_mode(
                examples_by_op[failed_op],
                mode,
                exhaustive=True,
            ):
                lines.append(f"  {trial_line}")
            lines.append("  this operator has no surviving rule under this flow.")
            lines.append("  reject this flow because at least one operator has no surviving low-cost rule.")
        else:
            lines.append(f"try flow [{_mode_name(mode)}]:")
            for op_char in sorted(examples_by_op):
                selected = _best_rule_for_group(examples_by_op[op_char], allowed_modes={mode})
                lines.append(f"  operator {op_char}:")
                for trial_line in _operator_trial_lines_for_mode(
                    examples_by_op[op_char],
                    mode,
                    selected_rule=selected,
                ):
                    lines.append(f"  {trial_line}")
                survivor = _rule_op_phrase(selected) if selected is not None else "unknown"
                lines.append(f"  surviving rule for operator {op_char}: {survivor}.")
            lines.append("  this flow works, but a simpler final rule is preferred later.")

    return lines


def _operator_trial_lines(
    op_char: str,
    examples: list[tuple[str, str, str, str]],
    selected_rule: _Rule,
) -> list[str]:
    return _operator_trial_lines_for_mode(
        examples,
        selected_rule.mode,
        selected_rule=selected_rule,
    )


def _select_target_rule(
    examples_by_op: dict[str, list[tuple[str, str, str, str]]],
    reversal_out_mode: str,
    q_op: str,
) -> tuple[_Rule, str] | None:
    shared_modes = {
        mode
        for mode in _candidate_modes(reversal_out_mode)
        if all(_rules_matching_examples(examples, mode) for examples in examples_by_op.values())
    }
    if not shared_modes:
        return None

    if q_op in examples_by_op:
        candidates: list[_Rule] = []
        for mode in shared_modes:
            selected = _best_rule_for_group(examples_by_op[q_op], allowed_modes={mode})
            if selected is not None:
                candidates.append(selected)
        if candidates:
            rule = min(candidates, key=_rule_key)
            return rule, ""

    sorted_modes = sorted(shared_modes, key=lambda item: (_mode_name(item).count("->"), _mode_name(item)))
    for mode in sorted_modes:
        selected_rules = {
            op_char: _best_rule_for_group(examples, allowed_modes={mode})
            for op_char, examples in examples_by_op.items()
        }
        if any(rule is None for rule in selected_rules.values()):
            continue
        used_families = {
            _PRIORITY_FAMILY[rule.op_name]
            for rule in selected_rules.values()
            if rule is not None and rule.op_name in _PRIORITY_FAMILY
        }
        for op_name, offset in _OP_PRIORITY:
            if _PRIORITY_FAMILY[op_name] not in used_families:
                return _Rule(mode, op_name, offset), ""
    mode = sorted_modes[0]
    op_name, offset = _OP_PRIORITY[0]
    return _Rule(mode, op_name, offset), ""


def _rule_summary(op_char: str, rule: _Rule) -> str:
    return (
        f"operator {op_char}: {_rule_op_phrase(rule)}"
        f"; mode = {_mode_name(rule.mode)}"
    )


def _missing_operator_elimination_lines(
    selected_rules: dict[str, _Rule],
    q_op: str,
    target_rule: _Rule,
) -> list[str]:
    used_families = {
        _PRIORITY_FAMILY[rule.op_name]: op_char
        for op_char, rule in selected_rules.items()
        if op_char != q_op and rule.op_name in _PRIORITY_FAMILY
    }
    lines = [
        "  Step 3. Check whether the question operator appears in the examples.",
        f"    The question operator is {q_op}.",
        f"    Operator {q_op} does not appear in the examples, so infer it by elimination.",
        "    Priority: "
        + ", ".join(_priority_rule_phrase(op_name, offset) for op_name, offset in _OP_PRIORITY)
        + ".",
    ]
    for op_name, offset in _OP_PRIORITY:
        phrase = _priority_rule_phrase(op_name, offset)
        family = _PRIORITY_FAMILY[op_name]
        owner = used_families.get(family)
        if owner is not None:
            lines.append(f"    remove {phrase}: {family} family is already used by operator {owner}.")
            continue
        lines.append(f"    keep {phrase}: first unused operation, so assign it to operator {q_op}.")
        break
    lines.append(f"    {_rule_summary(q_op, target_rule)}.")
    return lines


def reasoning_equation_numeric(problem: Problem) -> str | None:
    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        parsed_lhs = _parse_expr(str(ex.input_value))
        if parsed_lhs is None:
            continue
        a_str, op_char, b_str = parsed_lhs
        parsed.append((a_str, op_char, b_str, str(ex.output_value)))
    if not parsed:
        return None

    question = _parse_expr(str(problem.question))
    if question is None:
        return None
    q_a, q_op, q_b = question

    by_op: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for example in parsed:
        by_op[example[1]].append(example)

    reversal_out_mode, reversal_choice_lines = _reversal_choice(parsed)
    target = _select_target_rule(by_op, reversal_out_mode, q_op)
    if target is None:
        return None
    target_rule, _ = target
    best_mode = target_rule.mode

    selected_rules: dict[str, _Rule] = {}
    for op_char, examples in by_op.items():
        selected = _best_rule_for_group(examples, allowed_modes={best_mode})
        if selected is not None:
            selected_rules[op_char] = selected
    selected_rules[q_op] = target_rule

    lines: list[str] = []
    lines.append("We need to infer a numeric equation rule from the examples and then solve the question.")
    lines.append("The reasoning should include trial and error: try simple shared flows, reject candidates that contradict an example, keep the surviving operator rule, and then apply it to the question.")
    lines.append("")
    lines.append("Examples:")
    for a_str, op_char, b_str, out in parsed:
        lines.append(f"  {a_str}{op_char}{b_str} = {out}")
    lines.append("")
    lines.append(f"Question: {problem.question}")

    lines.append("")
    lines.append("Reasoning flow:")
    lines.append("  Step 1. Inspect the right-hand sides to decide the output reversal style.")
    for choice_line in reversal_choice_lines:
        lines.append(f"    {choice_line}")
    lines.append("  Step 2. Try shared input/output flows in priority order.")
    for trial_line in _flow_trial_lines(
        by_op,
        best_mode,
        selected_rules,
        reversal_out_mode,
        q_op,
    ):
        lines.append(f"    {trial_line}")
    lines.append(f"    Shared flow: {_mode_name(best_mode)}.")
    if q_op not in by_op:
        lines.extend(_missing_operator_elimination_lines(selected_rules, q_op, target_rule))
    else:
        lines.append("  Step 3. Check the question operator and record the selected operator rules.")
        lines.append(f"    The question operator is {q_op}.")
        if q_op in by_op:
            lines.append(f"    Operator {q_op} appears in the examples, so reuse its inferred rule.")
        else:
            lines.append(f"    Operator {q_op} does not appear in the examples, so use the fallback inferred rule.")

        for op_char in sorted(selected_rules):
            rule = selected_rules[op_char]
            lines.append(f"    {_rule_summary(op_char, rule)}.")

    lines.append("  Step 4. Verify that the inferred rules reproduce every example.")
    for idx, (a_str, op_char, b_str, expected) in enumerate(parsed, start=1):
        rule = selected_rules.get(op_char)
        if rule is None:
            return None
        ok, line = _compact_rule_attempt(rule, a_str, op_char, b_str, expected)
        if not ok:
            return None
        lines.append(f"    Example {idx}: {line}")

    lines.append("")
    lines.append(f"  Step 5. Apply the same flow to the question: {problem.question}.")
    final_predicted = _predict_with_rule(q_a, q_b, q_op, target_rule)
    if final_predicted is None:
        return None
    ok, final_line = _compact_rule_attempt(target_rule, q_a, q_op, q_b, final_predicted)
    if not ok:
        return None
    lines.append(f"    {final_line}")
    lines.append(f"    Therefore the result is {final_predicted}.")
    lines.append("")
    lines.append(f"The answer in \\boxed{{-}} is \\boxed{{{final_predicted}}}")
    return "\n".join(lines)
