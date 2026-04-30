"""SFT training script helpers that are not part of the training loop itself."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

from scripts.basic.const import PROBLEMS_INDEX
from scripts.train.base import (
    CATEGORY_ORDER,
    TASK_TYPE_BY_CATEGORY,
    TASK_TYPE_ORDER,
    TrainingExample,
)
from scripts.train.loss_config import LossConfig


def parse_category_limit_counts(value: str) -> list[int | None]:
    """カテゴリ順の上限リストをパースする。空欄/null は上限なし。"""
    return _parse_limit_counts(value, CATEGORY_ORDER, "category")


def parse_task_type_limit_counts(value: str) -> list[int | None]:
    """タスクタイプ順の上限リストをパースする。空欄/null は上限なし。"""
    return _parse_limit_counts(value, TASK_TYPE_ORDER, "task type")


def _parse_limit_counts(
    value: str,
    order: tuple[str, ...],
    label: str,
) -> list[int | None]:
    """固定順の上限リストをパースする。空欄/null は上限なし。"""
    raw = value.strip()
    if not raw:
        return [None] * len(order)
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if not raw.endswith("]"):
                raise argparse.ArgumentTypeError(str(exc)) from exc
            parsed = [part.strip() for part in raw[1:-1].split(",")]
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError("category limits must be a list")
        parts = parsed
    else:
        parts = [part.strip() for part in raw.split(",")]

    if len(parts) > len(order):
        raise argparse.ArgumentTypeError(
            f"expected at most {len(order)} {label} limits, got {len(parts)}"
        )

    limits: list[int | None] = []
    for part in parts:
        if part is None:
            limits.append(None)
            continue
        if isinstance(part, str) and part.strip().lower() in {"", "none", "null", "-"}:
            limits.append(None)
            continue
        try:
            limit = int(part)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"invalid category limit value: {part!r}"
            ) from exc
        if limit < 0:
            raise argparse.ArgumentTypeError("category limits must be >= 0")
        limits.append(limit)

    limits.extend([None] * (len(order) - len(limits)))
    return limits


def stratified_batches(
    examples: list[TrainingExample],
    batch_size: int,
    rng: random.Random,
    *,
    stratify_by: str,
) -> list[list[int]]:
    """全体のカテゴリ/タスクタイプ比率をなるべく保つバッチを返す。

    例: データが task_type A:B:C = 3:1:1 なら、各バッチもできるだけ
    3:1:1 に近くなるよう、各 group のサンプルを全バッチへ均等に撒く。
    """
    n = len(examples)
    n_batches = math.ceil(n / batch_size)

    def group_key(example: TrainingExample) -> str:
        if stratify_by == "task_type":
            return TASK_TYPE_BY_CATEGORY[example.category]
        return example.category

    by_group: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_group.setdefault(group_key(ex), []).append(i)
    for idx_list in by_group.values():
        rng.shuffle(idx_list)

    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for group in sorted(by_group.keys()):
        for idx in by_group[group]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    return batches


def compute_epoch_metrics(
    loss_config: LossConfig,
    examples: list[TrainingExample],
    all_ref_logprobs: dict[str, list[float]],
    all_epoch_logprobs: dict[str, list[float]],
    epoch: int,
) -> list[list[dict[str, float]]]:
    """1エポック分の集約指標を全サンプル平均で計算する。"""
    all_metrics: dict[str, list[tuple[float, int]]] = {}
    all_nll_values: list[tuple[float, int]] = []
    all_token_diffs: list[float] = []
    all_token_epoch_lps: list[float] = []
    for example in examples:
        key = example.problem_id
        epoch_logprobs = all_epoch_logprobs[key]
        ref_logprobs = all_ref_logprobs[key]
        _, full_mask = example.load_tokens()
        target_mask = full_mask[1:]
        target_mask = target_mask[: len(epoch_logprobs)]
        weight = example.unmasked_token_count
        metrics = loss_config.compute_metrics(
            epoch_logprobs, ref_logprobs, target_mask, epoch=epoch
        )
        for metric_name, metric_value in metrics.items():
            all_metrics.setdefault(metric_name, []).append((metric_value, weight))
        unmasked_lps = []
        for lp, rp, m in zip(epoch_logprobs, ref_logprobs, target_mask):
            if m == 1:
                unmasked_lps.append(lp)
                all_token_diffs.append(lp - rp)
                all_token_epoch_lps.append(lp)
        if unmasked_lps:
            nll = -sum(unmasked_lps) / len(unmasked_lps)
            all_nll_values.append((nll, weight))

    avg = {}
    for k, metric_entries in all_metrics.items():
        total_weight = sum(w for _, w in metric_entries)
        if total_weight > 0:
            avg[k] = round(sum(v * w for v, w in metric_entries) / total_weight, 6)

    general_metrics: list[list[dict]] = []
    if all_nll_values:
        total_w = sum(w for _, w in all_nll_values)
        avg_nll = sum(v * w for v, w in all_nll_values) / total_w
        general_metrics.append([{"nll_per_token": round(avg_nll, 6)}])
        general_metrics.append([{"perplexity": round(math.exp(min(avg_nll, 20)), 4)}])
    general_metrics.extend(
        loss_config.compute_global_metrics(all_token_diffs, all_token_epoch_lps)
    )

    layout = loss_config.chart_layout()
    metrics_grouped: list[list[dict]] = []
    for group in layout:
        chart = [{k: avg[k]} for k in group if k in avg]
        if chart:
            metrics_grouped.append(chart)
    return metrics_grouped[:2] + general_metrics + metrics_grouped[2:]


def filter_training_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    """必要に応じてカテゴリで絞り込む。"""
    return examples


def _load_cot_prompt_correctness() -> dict[str, bool]:
    """scripts/cot_prompt が生成した答えの正誤を problem_id -> bool で返す。"""
    correctness: dict[str, bool] = {}
    with PROBLEMS_INDEX.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            correctness[str(row["id"])] = row.get("status") == "rule_found"
    if not correctness:
        raise ValueError(f"No problem rows found in {PROBLEMS_INDEX}")
    return correctness


def filter_examples_by_cot_prompt(
    examples: list[TrainingExample],
    *,
    mode: str,
) -> tuple[list[TrainingExample], dict[str, object]]:
    """scripts/cot_prompt で作った答えの正誤に応じて学習サンプルを選ぶ。"""
    if mode == "all":
        return examples, {
            "mode": mode,
            "source": str(PROBLEMS_INDEX),
            "matched_problem_rows": None,
            "missing_problem_rows": None,
            "kept_examples": len(examples),
            "dropped_examples": 0,
        }
    correctness = _load_cot_prompt_correctness()

    kept: list[TrainingExample] = []
    missing = 0
    for example in examples:
        if example.problem_id not in correctness:
            missing += 1
            continue
        is_correct = correctness[example.problem_id]
        if mode == "incorrect" and not is_correct:
            kept.append(example)
        elif mode == "correct" and is_correct:
            kept.append(example)

    if not kept:
        raise ValueError(
            "CoT prompt filtering removed all training examples. "
            f"mode={mode}, source={PROBLEMS_INDEX}"
        )

    return kept, {
        "mode": mode,
        "source": str(PROBLEMS_INDEX),
        "matched_problem_rows": len(correctness),
        "missing_problem_rows": missing,
        "kept_examples": len(kept),
        "dropped_examples": len(examples) - len(kept),
        "correct_cot_prompt_problems": sum(1 for v in correctness.values() if v),
        "incorrect_cot_prompt_problems": sum(1 for v in correctness.values() if not v),
    }


def limit_examples_by_category(
    examples: list[TrainingExample],
    limits: list[int | None] | None,
) -> tuple[list[TrainingExample], dict[str, object]]:
    """カテゴリごとの上限数に従って学習サンプルを切り詰める。"""
    before_counts = Counter(example.category for example in examples)
    if limits is None:
        return examples, {
            "enabled": False,
            "category_order": list(CATEGORY_ORDER),
            "limits": None,
            "before_counts": dict(sorted(before_counts.items())),
            "after_counts": dict(sorted(before_counts.items())),
            "kept_examples": len(examples),
            "dropped_examples": 0,
        }

    limit_by_category = dict(zip(CATEGORY_ORDER, limits))
    seen: Counter[str] = Counter()
    kept: list[TrainingExample] = []
    for example in examples:
        limit = limit_by_category.get(example.category)
        if limit is None:
            kept.append(example)
            continue
        if seen[example.category] < limit:
            kept.append(example)
            seen[example.category] += 1

    if not kept:
        raise ValueError(
            "Category limit filtering removed all training examples. "
            f"limits={limits}"
        )

    after_counts = Counter(example.category for example in kept)
    return kept, {
        "enabled": True,
        "category_order": list(CATEGORY_ORDER),
        "limits": {
            category: limit
            for category, limit in zip(CATEGORY_ORDER, limits)
            if limit is not None
        },
        "before_counts": dict(sorted(before_counts.items())),
        "after_counts": dict(sorted(after_counts.items())),
        "kept_examples": len(kept),
        "dropped_examples": len(examples) - len(kept),
    }


def limit_examples_by_task_type(
    examples: list[TrainingExample],
    limits: list[int | None] | None,
    *,
    strategy: str,
    seed: int,
) -> tuple[list[TrainingExample], dict[str, object]]:
    """タスクタイプごとの上限数に従って学習サンプルを切り詰める。"""
    before_counts = Counter(
        TASK_TYPE_BY_CATEGORY[example.category] for example in examples
    )
    if limits is None:
        return examples, {
            "enabled": False,
            "task_type_order": list(TASK_TYPE_ORDER),
            "limits": None,
            "strategy": strategy,
            "seed": seed if strategy == "random" else None,
            "before_counts": dict(sorted(before_counts.items())),
            "after_counts": dict(sorted(before_counts.items())),
            "kept_examples": len(examples),
            "dropped_examples": 0,
        }

    limit_by_task_type = dict(zip(TASK_TYPE_ORDER, limits))
    if strategy == "head":
        seen: Counter[str] = Counter()
        kept: list[TrainingExample] = []
        for example in examples:
            task_type = TASK_TYPE_BY_CATEGORY[example.category]
            limit = limit_by_task_type.get(task_type)
            if limit is None:
                kept.append(example)
                continue
            if seen[task_type] < limit:
                kept.append(example)
                seen[task_type] += 1
    elif strategy == "random":
        rng = random.Random(seed)
        indices_by_task_type: dict[str, list[int]] = {}
        for i, example in enumerate(examples):
            task_type = TASK_TYPE_BY_CATEGORY[example.category]
            indices_by_task_type.setdefault(task_type, []).append(i)

        selected_indices: set[int] = set()
        for task_type, indices in indices_by_task_type.items():
            limit = limit_by_task_type.get(task_type)
            if limit is None or limit >= len(indices):
                selected_indices.update(indices)
            elif limit > 0:
                selected_indices.update(rng.sample(indices, limit))

        kept = [example for i, example in enumerate(examples) if i in selected_indices]
    else:
        raise ValueError(f"Unknown task_type_limit_strategy: {strategy}")

    if not kept:
        raise ValueError(
            "Task type limit filtering removed all training examples. "
            f"limits={limits}"
        )

    after_counts = Counter(TASK_TYPE_BY_CATEGORY[example.category] for example in kept)
    return kept, {
        "enabled": True,
        "task_type_order": list(TASK_TYPE_ORDER),
        "limits": {
            task_type: limit
            for task_type, limit in zip(TASK_TYPE_ORDER, limits)
            if limit is not None
        },
        "strategy": strategy,
        "seed": seed if strategy == "random" else None,
        "before_counts": dict(sorted(before_counts.items())),
        "after_counts": dict(sorted(after_counts.items())),
        "kept_examples": len(kept),
        "dropped_examples": len(examples) - len(kept),
    }
