"""ソルバー問題で共有する生ペイロードと実行時データ型。"""

import json
from typing import Any, Literal, cast

from scripts.basic.const import PROBLEM_DIR, PROBLEMS_INDEX

ProblemCategory = Literal[
    "bit_manipulation",
    "cipher",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "gravity",
    "numeral",
    "unit_conversion",
]


class Example:
    input_value: str
    output_value: str

    def __init__(self, input_value: str, output_value: str):
        self.input_value = input_value
        self.output_value = output_value

    def to_payload(self) -> dict[str, str]:
        return {
            "input_value": self.input_value,
            "output_value": self.output_value,
        }


class Problem:
    id: str
    category: ProblemCategory
    examples: list[Example]
    question: str
    answer: str
    prompt: str

    def __init__(
        self,
        id: str,
        category: ProblemCategory,
        examples: list[Example],
        question: str,
        answer: str,
        prompt: str = "",
    ):
        self.id = id
        self.category = category
        self.examples = examples
        self.question = question
        self.answer = answer
        self.prompt = prompt

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Problem":
        raw_examples = cast(list[dict[str, Any]], payload["examples"])
        examples = [
            Example(
                str(example["input_value"]),
                str(example["output_value"]),
            )
            for example in raw_examples
        ]
        return cls(
            id=str(payload["id"]),
            category=cast(ProblemCategory, payload["category"]),
            examples=examples,
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            prompt=str(payload.get("prompt", "")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "answer": self.answer,
            "examples": [example.to_payload() for example in self.examples],
            "question": self.question,
        }

    def to_index_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "category": self.category,
        }

    @classmethod
    def load_from_json(cls, id: str) -> "Problem":
        with (PROBLEM_DIR/ f"{id}.jsonl").open() as f:
            payload = cast(dict[str, Any], json.loads(f.readline()))
        return cls.from_payload(payload)

    @classmethod
    def load_all(cls) -> list["Problem"]:
        problems: list[Problem] = []
        with PROBLEMS_INDEX.open() as f:
            line = f.readline().strip()
        if line:
            problems.append(
                cls.from_payload(cast(dict[str, Any], json.loads(line)))
            )
        return problems


def _fmt_int_with_dp(value: int, dp: int) -> str:
    """整数を指定した小数桁数の10進文字列として整形する。"""
    if dp == 0:
        return str(value)
    s = str(value).zfill(dp + 1)
    s = s[: len(s) - dp] + "." + s[len(s) - dp :]
    # 先頭のゼロを削るが、小数点前の1桁は残す
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def truncate_3dp(s: str) -> str:
    """10進文字列を最大で小数第3位まで切り捨てる（丸めなし）。"""
    if "." not in s:
        return s
    integer, frac = s.split(".")
    if len(frac) <= 3:
        return s
    return integer + "." + frac[:3]


def _dp_count(s: str) -> int:
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def pad_dp(s: str, n: int) -> str:
    """10進文字列を小数点以下ちょうど指定桁数に揃える。"""
    if "." not in s:
        s = s + "."
    integer, frac = s.split(".")
    return integer + "." + frac.ljust(n, "0")


def cast_dp_pair(a: str, b: str) -> tuple[str, str, int, int]:
    """2つの値の小数点以下桁数を同じに揃える。

    戻り値は (a_padded, b_padded, a_target_dp, b_target_dp)。
    目標桁数は両者の最大値で、個別の値はそれぞれ何桁へ揃えたかを示す。
    """
    da, db = _dp_count(a), _dp_count(b)
    target = max(da, db)
    return pad_dp(a, target), pad_dp(b, target), target, target


def long_multiplication_lines(a_str: str, b_str: str) -> tuple[list[str], str]:
    """2つの10進数の筆算風の掛け算手順を生成する。

    *b* を位取りごとの成分へ分解し、それぞれに *a* を掛けてから累積和を表示する。

    戻り値は (lines, result_str)。result_str は正確な積。
    """
    a_dp = len(a_str.split(".")[1]) if "." in a_str else 0
    b_dp = len(b_str.split(".")[1]) if "." in b_str else 0
    total_dp = a_dp + b_dp

    a_int = int(a_str.replace(".", ""))
    b_int = int(b_str.replace(".", ""))

    lines: list[str] = []

    # 乗数を位取り成分へ分解する（最下位桁から）
    b_digits_str = str(abs(b_int))
    b_num_digits = len(b_digits_str)

    # (成分表示, スケール済み積の整数値, 積の表示)
    components: list[tuple[str, int, str]] = []
    for i in range(b_num_digits - 1, -1, -1):
        d = int(b_digits_str[i])
        if d == 0:
            continue
        # 乗数の小数桁数に合わせてスケール済みの成分値
        comp_scaled = d * (10 ** (b_num_digits - 1 - i))
        comp_display = _fmt_int_with_dp(comp_scaled, b_dp)
        if b_dp > 0:
            comp_display = pad_dp(comp_display, b_dp)

        product_int = a_int * comp_scaled  # 全体の小数桁数に合わせてスケール済み
        product_display = _fmt_int_with_dp(product_int, total_dp)
        if total_dp > 0:
            product_display = pad_dp(product_display, total_dp)

        components.append((comp_display, product_int, product_display))

    # 掛け算の行: a * 成分 = 積
    for comp_display, _, product_display in components:
        lines.append(f"{a_str} * {comp_display} = {product_display}")

    # 累積和（小さい位から大きい位へ畳み込む）
    if len(components) >= 2:
        running = components[0][1]
        for i in range(1, len(components)):
            running_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                running_display = pad_dp(running_display, total_dp)
            running += components[i][1]
            sum_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                sum_display = pad_dp(sum_display, total_dp)
            lines.append(f"{running_display} + {components[i][2]} = {sum_display}")

    # 最終結果の文字列を計算する
    total = a_int * b_int
    result_str = _fmt_int_with_dp(total, total_dp)
    return lines, result_str


def long_division_lines(
    numerator_str: str, denominator_str: str, max_decimal_digits: int = 3
) -> tuple[list[str], str]:
    """反復減算による筆算風の割り算手順を生成する。

    戻り値は (lines, result_str)。result_str は切り捨て済みの商。
    """
    n_dp: int = len(numerator_str.split(".")[1]) if "." in numerator_str else 0
    d_dp: int = len(denominator_str.split(".")[1]) if "." in denominator_str else 0
    max_dp: int = max(n_dp, d_dp)

    num: int = int(round(float(numerator_str) * 10**max_dp))
    den: int = int(round(float(denominator_str) * 10**max_dp))

    lines: list[str] = []
    acc: int = 0  # 整数としての累積値。実値は累積値を小数桁数で割ったもの
    decimal_digits: int = 0

    def fmt_acc() -> str:
        if decimal_digits == 0:
            return str(acc)
        s = str(acc).zfill(decimal_digits + 1)
        return s[:-decimal_digits] + "." + s[-decimal_digits:]

    def fmt_scale() -> str:
        if decimal_digits == 0:
            return "1"
        return "0." + "0" * (decimal_digits - 1) + "1"

    def fmt_line(n: int) -> str:
        return f"= {fmt_acc()} + {fmt_scale()} * {n} / {den}"

    lines.append(fmt_line(num))

    while decimal_digits <= max_decimal_digits:
        if num >= den:
            num -= den
            acc += 1
            lines.append(fmt_line(num))
        else:
            decimal_digits += 1
            if decimal_digits > max_decimal_digits:
                break
            num *= 10
            acc *= 10
            lines.append(fmt_line(num))

    # ループ終了前に小数桁数が上限を超えて増えていた場合は戻す
    if decimal_digits > max_decimal_digits:
        decimal_digits = max_decimal_digits
    return lines, fmt_acc()
