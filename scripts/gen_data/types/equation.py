from __future__ import annotations

import random
import re

from scripts.cot_prompt.store_types import Example
from scripts.gen_data.types.base import RawProblemRecord, make_generated_record, make_record

_SKIP_HINTS = (
    "Wonderland",
    "determine",
    "examples",
    "rules",
    "result",
    "few",
    "applied",
    "set of",
    "secret",
)


def _example_lines(prompt: str) -> list[str]:
    return [
        line.strip()
        for line in prompt.splitlines()
        if " = " in line and not line.startswith("Now") and "determine" not in line.lower()
    ]


def _target_expr(prompt: str) -> str:
    match = re.search(r"determine the result for:\s*(.+)", prompt)
    return match.group(1).strip() if match else ""


def _parse_examples_and_question(prompt: str) -> tuple[list[Example], str]:
    examples: list[Example] = []
    for line in _example_lines(prompt):
        left, right = line.split(" = ", 1)
        examples.append(Example(left.strip(), right.strip()))
    return examples, _target_expr(prompt)


def _is_equation_prompt(prompt: str) -> bool:
    return "secret set of transformation rules is applied to equations" in prompt


def _is_numeric_equation(prompt: str) -> bool:
    for line in prompt.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        if any(hint in line for hint in _SKIP_HINTS):
            continue
        left = line.split("=", 1)[0]
        if re.search(r"\d", left):
            return True
    return False


def _numeric_operator(expr: str) -> str | None:
    match = re.match(r"^\d+([^\d]+)\d+$", expr.strip())
    return match.group(1) if match else None


def _numeric_split(prompt: str) -> str:
    target_op = _numeric_operator(_target_expr(prompt))
    if target_op is None:
        return "deduce"
    example_ops = {
        op
        for line in _example_lines(prompt)
        if (op := _numeric_operator(line.split(" = ", 1)[0].strip()))
    }
    return "deduce" if target_op in example_ops else "guess"


def _cryptarithm_split(prompt: str) -> str:
    target = _target_expr(prompt)
    example_lefts = [line.split(" = ", 1)[0].strip() for line in _example_lines(prompt)]
    if not target or not example_lefts:
        return "deduce"
    example_chars = set("".join(example_lefts))
    return "deduce" if all(ch in example_chars for ch in target) else "guess"


class NumericEquationBuilder:
    category = "equation_numeric"
    generated_categories = ("equation_numeric_deduce", "equation_numeric_guess")
    description = "equation transformation with numeric operands"

    def matches(self, prompt: str) -> bool:
        return _is_equation_prompt(prompt) and _is_numeric_equation(prompt)

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        examples, question = _parse_examples_and_question(row.get("prompt", ""))
        split = _numeric_split(row.get("prompt", ""))
        return make_record(row, f"equation_numeric_{split}", examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        op_rules = [("+", "add"), ("-", "sub"), ("*", "mul"), ("%", "mod")]
        op_count = rng.randint(1, 3)
        if category == "equation_numeric_guess":
            target_spec = rng.choice(op_rules)
            candidate_specs = [spec for spec in op_rules if spec != target_spec]
            example_specs = rng.sample(candidate_specs, op_count)
        else:
            example_specs = rng.sample(op_rules, op_count)
            target_spec = rng.choice(example_specs)
        example_count = rng.randint(4, 6)

        def value(a: int, b: int, rule: str) -> int:
            if rule == "add":
                return a + b
            if rule == "sub":
                return a - b
            if rule == "mul":
                return a * b
            return a % b

        def make_expr(op: str, rule: str) -> tuple[str, str]:
            if rule == "mul":
                a, b = rng.randint(10, 35), rng.randint(10, 35)
            elif rule == "mod":
                b = rng.randint(10, 35)
                a = rng.randint(b, 99)
            elif rule == "sub":
                a, b = rng.randint(20, 99), rng.randint(10, 90)
            else:
                a, b = rng.randint(10, 99), rng.randint(10, 99)
            return f"{a:02d}{op}{b:02d}", str(value(a, b, rule))

        examples: list[Example] = []
        for op, rule in example_specs:
            left, right = make_expr(op, rule)
            examples.append(Example(left, right))
        for _ in range(example_count - len(examples)):
            op, rule = rng.choice(example_specs)
            left, right = make_expr(op, rule)
            examples.append(Example(left, right))
        rng.shuffle(examples)

        question, answer = make_expr(*target_spec)
        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:",
            *[f"{example.input_value} = {example.output_value}" for example in examples],
            f"Now, determine the result for: {question}",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)


class CryptarithmBuilder:
    category = "cryptarithm"
    generated_categories = ("cryptarithm_deduce", "cryptarithm_guess")
    description = "equation transformation with symbol operands"

    def matches(self, prompt: str) -> bool:
        return _is_equation_prompt(prompt) and not _is_numeric_equation(prompt)

    def build(self, row: dict[str, str]) -> RawProblemRecord:
        examples, question = _parse_examples_and_question(row.get("prompt", ""))
        split = _cryptarithm_split(row.get("prompt", ""))
        return make_record(row, f"cryptarithm_{split}", examples, question)

    def generate(
        self,
        category: str,
        rng: random.Random,
        problem_id: str,
    ) -> RawProblemRecord:
        value_symbols = list("!@#$%&?ABCDEFG")
        rng.shuffle(value_symbols)
        value_symbols = value_symbols[:10]
        op_symbols = list("+-*/|~^")
        rng.shuffle(op_symbols)
        op_count = rng.randint(1, 3)
        if category == "cryptarithm_guess":
            question_op = op_symbols[0]
            example_ops = op_symbols[1 : 1 + op_count]
        else:
            example_ops = op_symbols[:op_count]
            question_op = rng.choice(example_ops)
        example_count = rng.randint(4, 6)
        op_modes = {op: rng.choice(("fwd", "rev")) for op in example_ops}

        def pair() -> str:
            return rng.choice(value_symbols) + rng.choice(value_symbols)

        def output_for(a: str, b: str, mode: str) -> str:
            return a + b if mode == "fwd" else b + a

        def make_expr(op: str, mode: str) -> tuple[str, str]:
            a = pair()
            b = pair()
            return f"{a}{op}{b}", output_for(a, b, mode)

        examples: list[Example] = []
        for op in example_ops:
            left, right = make_expr(op, op_modes[op])
            examples.append(Example(left, right))
        for _ in range(example_count - len(examples)):
            op = rng.choice(example_ops)
            left, right = make_expr(op, op_modes[op])
            examples.append(Example(left, right))
        rng.shuffle(examples)
        question_mode = "fwd" if category == "cryptarithm_guess" else op_modes[question_op]
        question, answer = make_expr(question_op, question_mode)
        lines = [
            "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:",
            *[f"{example.input_value} = {example.output_value}" for example in examples],
            f"Now, determine the result for: {question}",
        ]
        prompt = "\n".join(lines)
        return make_generated_record(problem_id, category, prompt, answer, examples, question)
