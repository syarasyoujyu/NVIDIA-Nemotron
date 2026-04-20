"""numeric_equation パターンのバリデーターおよび出力生成。

各問題のプロンプトを解析し、演算子記号ごとに独立したルールを決定する。
ルールが全演算子で確定できれば「マッチ」とみなす。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..base import PatternValidator
from .label import _flow_label
from .matching import _best_mode_for_group
from .parser import _parse_lhs, _parse_op_groups
from .predict import _all_modes_for_group, _predict


class NumericEquationValidator(PatternValidator):
    pattern_name = "numeric_equation"

    def family_names(self) -> list[str]:
        return ["per_operator_any_mode"]

    def _parse_pairs(self, row: dict[str, str]) -> list[tuple[str, str]]:
        """プロンプトから (左辺, 右辺) ペアのリストを構築する。

        例示行と「Now, determine the result for:」行を収集し、
        最後尾に (ターゲット左辺, answer) を追加して返す。
        """
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
        """全演算子グループが何らかのモードでマッチするかを検証する。

        演算子ごとに独立してルールを探索し、全グループで成立すればマッチ。
        """
        if family_name != "per_operator_any_mode":
            return False
        pairs = self._parse_pairs(row)
        if not pairs:
            return False
        op_groups = _parse_op_groups(pairs)
        if op_groups is None:
            return False
        return all(
            _best_mode_for_group(examples) is not None
            for examples in op_groups.values()
        )

    def _extra_unmatched_fields(self, row: dict[str, str]) -> dict[str, str]:
        """アンマッチ行に対して、どの演算子記号がマッチ失敗かを記録する。"""
        pairs = self._parse_pairs(row)
        op_groups = _parse_op_groups(pairs)
        if op_groups is None:
            return {"failed_operators": "parse_error"}
        failed = [op for op, examples in op_groups.items() if _best_mode_for_group(examples) is None]
        return {"failed_operators": "|".join(failed) if failed else ""}

    def _build_matched_entry(self, row: dict[str, str]) -> dict | None:
        """マッチした行のエントリーを構築する。

        - operator_flows: 演算子記号 → 確定フローラベル（answer込み）
        - example_alternatives: 例示のみから導かれる代案が予測を分岐させる場合のみ記録
          - target_operator: ターゲット入力の演算子記号
          - target_answer: 実際の答え
          - answer_flow: answerを含めた場合の確定フロー
        """
        result = self.validate_row(row)
        if not result.matched:
            return None

        pairs = self._parse_pairs(row)
        op_groups = _parse_op_groups(pairs)
        if op_groups is None:
            return None

        # 演算子ごとに最良ルールを決定してフローラベルを生成
        operator_flows = {}
        for op_char, examples in op_groups.items():
            best = _best_mode_for_group(examples)
            if best:
                mode, op_name, offset = best
                rev_in, swap, out_mode = mode
                operator_flows[op_char] = _flow_label(op_name, offset, rev_in, swap, out_mode)

        # 例示のみ（answer除外）でターゲット演算子の別解を計算
        example_pairs = pairs[:-1]
        example_op_groups = _parse_op_groups(example_pairs) or {}
        target_lhs, _ = pairs[-1]
        target_parsed = _parse_lhs(target_lhs)
        target_op = target_parsed[1] if target_parsed else None

        example_alternatives: dict[str, str] = {}
        if target_op and target_parsed and target_op in example_op_groups:
            ta, _, tb = target_parsed
            all_modes = _all_modes_for_group(example_op_groups[target_op])
            predictions: dict[str, str] = {}
            for mode, op_name, offset in all_modes:
                rev_in, swap, out_mode = mode
                label = _flow_label(op_name, offset, rev_in, swap, out_mode)
                pred = _predict(ta, tb, target_op, op_name, offset, rev_in, swap, out_mode)
                predictions[label] = pred if pred is not None else "?"
            # 予測が分岐する場合のみ別解として記録
            if len(set(predictions.values())) > 1:
                example_alternatives = predictions

        entry: dict = {
            "id": row["id"],
            "pattern": self.pattern_name,
            "operator_flows": operator_flows,
        }
        if example_alternatives:
            entry["example_alternatives"] = example_alternatives
            entry["target_operator"] = target_op
            entry["target_answer"] = row["answer"]
            entry["answer_flow"] = operator_flows.get(target_op, "")
        return entry

    def write_outputs(self, output_root: Path, rows: list[dict[str, str]]) -> dict[str, str]:
        result = super().write_outputs(output_root, rows)

        pattern_dir = output_root / self.pattern_name
        matched_entries = [
            entry for row in rows
            if (entry := self._build_matched_entry(row)) is not None
        ]

        # matched.jsonl: 全マッチ行のルール詳細
        matched_path = pattern_dir / "matched.jsonl"
        with matched_path.open("w", encoding="utf-8") as f:
            for entry in matched_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # alternatives.csv: 別解が存在する問題の一覧（1行=1問題）
        alt_rows = []
        for entry in matched_entries:
            if "example_alternatives" not in entry:
                continue
            answer = entry["target_answer"]
            alts = entry["example_alternatives"]
            alt_flows = list(alts.keys())
            alt_predictions = list(alts.values())
            matches = [pred == answer for pred in alt_predictions]
            alt_rows.append({
                "id": entry["id"],
                "target_operator": entry["target_operator"],
                "answer_flow": entry["answer_flow"],
                "answer": answer,
                "alt_flows": json.dumps(alt_flows, ensure_ascii=False),
                "alt_predictions": json.dumps(alt_predictions, ensure_ascii=False),
                "matches_answer": json.dumps(matches),
            })
        alt_path = pattern_dir / "alternatives.csv"
        with alt_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "target_operator", "answer_flow", "answer", "alt_flows", "alt_predictions", "matches_answer"],
            )
            writer.writeheader()
            writer.writerows(alt_rows)

        return result


class NumericEquationDeduceValidator(NumericEquationValidator):
    pattern_name = "numeric_equation/deduce"


class NumericEquationGuessValidator(NumericEquationValidator):
    pattern_name = "numeric_equation/guess"


VALIDATOR = NumericEquationValidator()
DEDUCE_VALIDATOR = NumericEquationDeduceValidator()
GUESS_VALIDATOR = NumericEquationGuessValidator()
