"""学習スクリプト共通のユーティリティ。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import tinker

from scripts.basic.common import load_jsonl
from scripts.basic.const import CORPUS_DIR, CORPUS_INDEX
from scripts.train.loss_config import (
    CrossEntropyLossConfig,
    CrossEntropyWithWeightingLossConfig,
    LossConfig,
)

logger = logging.getLogger(__name__)

CATEGORY_ORDER = (
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
)

TASK_TYPE_ORDER = (
    "gravity",
    "numeral",
    "unit_conversion",
    "cipher",
    "bit_manipulation",
    "equation_transformation",
)

TASK_TYPE_BY_CATEGORY = {
    "gravity": "gravity",
    "numeral": "numeral",
    "unit_conversion": "unit_conversion",
    "cipher": "cipher",
    "bit_manipulation": "bit_manipulation",
    "cryptarithm_deduce": "equation_transformation",
    "cryptarithm_guess": "equation_transformation",
    "equation_numeric_deduce": "equation_transformation",
    "equation_numeric_guess": "equation_transformation",
}

Category = Literal[
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
]




class CorpusEntry(TypedDict):
    """corpus.jsonl の項目。"""

    problem_id: str
    segment: str
    category: Category
    masked_token_count: int
    unmasked_token_count: int
    token_count: int
    answer: str
    included: bool


def load_corpus_entries() -> list[CorpusEntry]:
    """corpus.jsonl を読み込み、型付き項目として返す。"""
    return cast(list[CorpusEntry], load_jsonl(CORPUS_INDEX))


@dataclass
class TrainingExample:
    """事前トークナイズ済みデータを持つ単一の学習サンプル。"""

    problem_id: str
    segment: str
    category: Category
    masked_token_count: int
    unmasked_token_count: int

    @classmethod
    def from_dict(cls, entry: CorpusEntry) -> TrainingExample:
        return cls(
            problem_id=entry["problem_id"],
            segment=entry["segment"],
            category=entry["category"],
            masked_token_count=entry["masked_token_count"],
            unmasked_token_count=entry["unmasked_token_count"],
        )

    def get_segment_path(self) -> Path:
        """コーパスセグメントファイルへのパスを取得する。"""
        return CORPUS_DIR / self.problem_id / self.segment

    def load_tokens(self) -> tuple[list[int], list[int]]:
        """セグメントファイルからトークンとマスクを読み込む。

        戻り値は (tokens, mask)。mask[i]=1 は未マスク（このトークンで学習）を表す。
        """
        segments = load_jsonl(self.get_segment_path())
        tokens: list[int] = []
        mask: list[int] = []
        for seg in segments:
            seg_tokens = seg["tokens"]
            tokens.extend(seg_tokens)
            mask_val = 1 if seg["type"] == "unmasked" else 0
            mask.extend([mask_val] * len(seg_tokens))
        return tokens, mask


def build_datum(
    tokens: list[int],
    advantages: list[int],
    ref_logprobs: list[float] | None,
    prev_logprobs: list[float] | None,
    epoch: int,
    loss: LossConfig,
) -> tinker.Datum:
    """学習用データを構築する。"""
    assert len(tokens) == len(advantages)
    assert ref_logprobs is None or len(ref_logprobs) == len(tokens) - 1
    assert prev_logprobs is None or len(prev_logprobs) == len(tokens) - 1

    model_input = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=tokens[:-1])]
    )
    target_tokens = tokens[1:]

    loss_fn_inputs: dict[str, tinker.TensorData] = {
        "target_tokens": tinker.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
    }

    float_advantages = [float(a) for a in advantages[1:]]

    if isinstance(loss, CrossEntropyLossConfig):
        if isinstance(loss, CrossEntropyWithWeightingLossConfig):
            float_advantages = loss.apply_weights(
                float_advantages, prev_logprobs, ref_logprobs, epoch
            )
        loss_fn_inputs["weights"] = tinker.TensorData(
            data=float_advantages,
            dtype="float32",
            shape=[len(float_advantages)],
        )
    else:
        loss_fn_inputs["advantages"] = tinker.TensorData(
            data=float_advantages,
            dtype="float32",
            shape=[len(float_advantages)],
        )
        if ref_logprobs is not None:
            loss_fn_inputs["logprobs"] = tinker.TensorData(
                data=ref_logprobs,
                dtype="float32",
                shape=[len(ref_logprobs)],
            )

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs=loss_fn_inputs,
    )
