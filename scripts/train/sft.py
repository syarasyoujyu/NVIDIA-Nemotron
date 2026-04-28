"""nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 の SFT 学習を行い、エポックごとに logprob を保存する。

対応する loss 関数: cross_entropy, importance_sampling, ppo, cispo, dro
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import random
import time
from collections import Counter
from datetime import datetime

import tinker
from dotenv import load_dotenv

from scripts.basic.const import PROBLEMS_INDEX, SFT_DIR
from scripts.train.base import (
    CATEGORY_ORDER,
    TASK_TYPE_BY_CATEGORY,
    TASK_TYPE_ORDER,
    TrainingExample,
    build_datum,
    load_corpus_entries,
)
from scripts.train.config import Cfg, IndexRecord, LogprobRecord
from scripts.train.loss_config import (
    LossConfig,
)
from scripts.trainer.client import ServiceClient

load_dotenv()
logger = logging.getLogger(__name__)


def _parse_category_limit_counts(value: str) -> list[int | None]:
    """カテゴリ順の上限リストをパースする。空欄/null は上限なし。"""
    return _parse_limit_counts(value, CATEGORY_ORDER, "category")


def _parse_task_type_limit_counts(value: str) -> list[int | None]:
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


def _build_cfg() -> Cfg:
    p = argparse.ArgumentParser(description="SFT training script")
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lora_rank", type=int)
    p.add_argument("--max_length", type=int)
    p.add_argument("--log_path", type=str)
    p.add_argument("--backend", choices=["tinker", "modal"])
    p.add_argument("--micro_batch_size", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--train_mlp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--train_attn", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--train_unembed", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--grad_clip_norm", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument(
        "--cot_prompt_filter_mode",
        choices=["all", "incorrect", "correct"],
        default="all",
        help=(
            "all: use all corpus rows; incorrect: train only problems whose "
            "scripts/cot_prompt answer did not match the actual answer; "
            "correct: train only matched problems."
        ),
    )
    p.add_argument(
        "--batch_stratify_by",
        choices=["category", "task_type"],
        default="task_type",
        help=(
            "Distribute samples across batches while preserving the selected "
            "group ratio as much as possible."
        ),
    )
    p.add_argument(
        "--category_limit_counts",
        type=_parse_category_limit_counts,
        help=(
            "Per-category maximum counts in fixed category order. "
            "Accepts comma-separated values with blanks, or a JSON list. "
            f"Order: {', '.join(CATEGORY_ORDER)}"
        ),
    )
    p.add_argument(
        "--task_type_limit_counts",
        type=_parse_task_type_limit_counts,
        help=(
            "Per-task-type maximum counts in fixed task type order. "
            "Accepts comma-separated values with blanks, or a JSON list. "
            f"Order: {', '.join(TASK_TYPE_ORDER)}"
        ),
    )
    args = p.parse_args()

    cfg = Cfg()
    for field in [
        "num_epochs", "batch_size", "lora_rank", "max_length",
        "log_path", "backend", "micro_batch_size",
        "train_mlp", "train_attn", "train_unembed",
        "cot_prompt_filter_mode", "batch_stratify_by",
        "category_limit_counts", "task_type_limit_counts",
    ]:
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.learning_rate is not None:
        cfg.lr_schedule.learning_rate = args.learning_rate
    if args.grad_clip_norm is not None:
        cfg.adam_config.grad_clip_norm = args.grad_clip_norm
    if args.weight_decay is not None:
        cfg.adam_config.weight_decay = args.weight_decay
    return cfg


def _stratified_batches(
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

    # group ごとにまとめ、group 内でシャッフルする
    by_group: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_group.setdefault(group_key(ex), []).append(i)
    for idx_list in by_group.values():
        rng.shuffle(idx_list)

    # 各 group を全バッチへ均等に撒くことで、全体比率を各バッチへ近似する。
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

    # 各指標のトークン重み付き平均
    avg = {}
    for k, metric_entries in all_metrics.items():
        total_weight = sum(w for _, w in metric_entries)
        if total_weight > 0:
            avg[k] = round(sum(v * w for v, w in metric_entries) / total_weight, 6)

    # 汎用指標: 負の対数尤度、パープレキシティ、logprob 差分のパーセンタイル
    general_metrics: list[list[dict]] = []
    if all_nll_values:
        total_w = sum(w for _, w in all_nll_values)
        avg_nll = sum(v * w for v, w in all_nll_values) / total_w
        general_metrics.append([{"nll_per_token": round(avg_nll, 6)}])
        general_metrics.append([{"perplexity": round(math.exp(min(avg_nll, 20)), 4)}])
    general_metrics.extend(
        loss_config.compute_global_metrics(all_token_diffs, all_token_epoch_lps)
    )

    # チャート配置の定義に従ってチャートグループへ整形する
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
            "before_counts": dict(sorted(before_counts.items())),
            "after_counts": dict(sorted(before_counts.items())),
            "kept_examples": len(examples),
            "dropped_examples": 0,
        }

    limit_by_task_type = dict(zip(TASK_TYPE_ORDER, limits))
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
        "before_counts": dict(sorted(before_counts.items())),
        "after_counts": dict(sorted(after_counts.items())),
        "kept_examples": len(kept),
        "dropped_examples": len(examples) - len(kept),
    }


async def main():
    cfg = _build_cfg()

    log_path = SFT_DIR / cfg.log_path
    logprob_dir = log_path / "logprobs"

    # サンプルを読み込む
    entries = load_corpus_entries()
    entries = [e for e in entries if e["included"]]
    examples = [TrainingExample.from_dict(e) for e in entries]
    examples = filter_training_examples(examples)
    examples, cot_prompt_filter_stats = filter_examples_by_cot_prompt(
        examples,
        mode=cfg.cot_prompt_filter_mode,
    )
    examples, task_type_limit_stats = limit_examples_by_task_type(
        examples,
        limits=cfg.task_type_limit_counts,
    )
    examples, category_limit_stats = limit_examples_by_category(
        examples,
        limits=cfg.category_limit_counts,
    )
    total_masked = sum(e.masked_token_count for e in examples)
    total_unmasked = sum(e.unmasked_token_count for e in examples)
    logger.info(
        f"Loaded {len(examples)} examples, "
        f"{total_masked + total_unmasked:,} tokens "
        f"(unmasked={total_unmasked:,}, masked={total_masked:,})"
    )
    logger.info(f"CoT prompt filter: {cot_prompt_filter_stats}")
    logger.info(f"Task type limit filter: {task_type_limit_stats}")
    logger.info(f"Category limit filter: {category_limit_stats}")
    logger.info(f"Batch stratification: {cfg.batch_stratify_by}")

    n_batches = math.ceil(len(examples) / cfg.batch_size)
    total_steps = n_batches * cfg.num_epochs
    logger.info(
        f"Training for {n_batches} batches x {cfg.num_epochs} epochs"
        f" = {total_steps} steps"
    )
    lr_params = dataclasses.asdict(cfg.lr_schedule)
    logger.info(f"LR schedule: {type(cfg.lr_schedule).__name__} {lr_params}")

    # 学習クライアントを作成する
    service_client = ServiceClient(backend=cfg.backend)
    training_client = await service_client.create_lora_training_client_async(
        base_model=cfg.model_name,
        rank=cfg.lora_rank,
        train_mlp=cfg.train_mlp,
        train_attn=cfg.train_attn,
        train_unembed=cfg.train_unembed,
    )

    logprob_dir.mkdir(parents=True, exist_ok=True)

    # 設定を保存する
    config = dataclasses.asdict(cfg)
    config["time"] = datetime.now().strftime("%m-%d-%H-%M")
    config["stats"] = {
        "num_examples": len(examples),
        "total_masked_tokens": total_masked,
        "total_unmasked_tokens": total_unmasked,
        "total_steps": total_steps,
        "batch_stratify_by": cfg.batch_stratify_by,
        "cot_prompt_filter": cot_prompt_filter_stats,
        "task_type_limit_filter": task_type_limit_stats,
        "category_limit_filter": category_limit_stats,
    }
    with open(log_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # GitHub Pages 用のログパス一覧に追記する
    logpaths_file = SFT_DIR / "logpaths.txt"
    existing = set(logpaths_file.read_text().splitlines()) if logpaths_file.exists() else set()
    if cfg.log_path not in existing:
        with open(logpaths_file, "a") as f:
            f.write(cfg.log_path + "\n")

    step = 0
    # エポック0で収集し、エポック1以降の参照 logprobs として使う
    all_ref_logprobs: dict[str, list[float]] = {}
    # 直近エポックの logprobs
    all_prev_logprobs: dict[str, list[float]] = {}

    metrics_path = log_path / "metrics.jsonl"
    loss_path = log_path / "loss.jsonl"
    with (
        open(logprob_dir / "index.jsonl", "w") as index_file,
        open(metrics_path, "w") as metrics_file,
        open(loss_path, "w") as loss_file,
    ):
        for epoch in range(cfg.num_epochs):
            loss_fn_config = cfg.loss_config.config(epoch)
            logger.info(
                f"Starting epoch {epoch} "
                f"(loss={cfg.loss_config.name} config={loss_fn_config})"
            )
            epoch_start = time.time()

            rng = random.Random(epoch)
            batches = _stratified_batches(
                examples,
                cfg.batch_size,
                rng,
                stratify_by=cfg.batch_stratify_by,
            )

            print(f"Batch sizes: {Counter(len(batch) for batch in batches)}")

            epoch_dir = logprob_dir / str(epoch)

            for batch_indices in batches:
                batch_examples = [examples[i] for i in batch_indices]

                data: list[tinker.Datum] = []
                valid_examples: list[TrainingExample] = []
                target_masks: list[list[int]] = []
                batch_tokens: list[list[int]] = []
                batch_masks: list[list[int]] = []
                for example in batch_examples:
                    tokens, advantages = example.load_tokens()
                    if len(tokens) > cfg.max_length:
                        tokens = tokens[: cfg.max_length]
                        advantages = advantages[: cfg.max_length]
                    key = example.problem_id
                    if epoch == 0:
                        ref_logprobs = None
                        prev_logprobs = None
                    else:
                        ref_logprobs = all_ref_logprobs[key][: len(tokens) - 1]
                        prev_logprobs = all_prev_logprobs[key][: len(tokens) - 1]
                    datum = build_datum(
                        tokens,
                        advantages,
                        ref_logprobs,
                        prev_logprobs,
                        epoch,
                        cfg.loss_config,
                    )
                    data.append(datum)
                    valid_examples.append(example)
                    target_masks.append(advantages[1:])
                    batch_tokens.append(tokens)
                    batch_masks.append(advantages)

                if not data:
                    continue

                batch_time = time.time()

                lr = cfg.lr_schedule.get_lr(step, total_steps, epoch, cfg.num_epochs)

                # 順伝播・逆伝播と最適化ステップ
                fwd_bwd_future = await training_client.forward_backward_async(
                    data,
                    loss_fn=cfg.loss_config.name,
                    loss_fn_config=loss_fn_config,
                    micro_batch_size=cfg.micro_batch_size,
                )
                optim_future = await training_client.optim_step_async(
                    cfg.adam_config.to_adam_params(lr)
                )

                fwd_bwd_result = await fwd_bwd_future.result_async()
                optim_result = await optim_future.result_async()

                # logprobs を取り出す
                logprobs_list = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]

                elapsed = time.time() - batch_time
                logger.info(
                    f"epoch={epoch} step={step}/{total_steps} "
                    f"lr={lr:.2e} n={len(data)} "
                    f"loss={cfg.loss_config.name} t={elapsed:.1f}s"
                )

                # 指標を記録する
                metrics_record: dict = {
                    "epoch": epoch,
                    "step": step,
                    "lr": lr,
                    "n": len(data),
                    "elapsed": round(elapsed, 2),
                    "time": datetime.now().strftime("%m-%d-%H-%M"),
                }
                metrics_record.update(
                    {f"fwd/{k}": v for k, v in fwd_bwd_result.metrics.items()}
                )
                if optim_result.metrics:
                    metrics_record.update(
                        {f"optim/{k}": v for k, v in optim_result.metrics.items()}
                    )
                # このステップのカテゴリ別トークンあたり損失を計算する
                cat_loss: dict[str, float] = {}
                cat_tokens: dict[str, int] = {}
                for i, example in enumerate(valid_examples):
                    cat = example.category
                    lp_data = logprobs_list[i].data
                    loss_val = sum(-v for v, m in zip(lp_data, target_masks[i]) if m)
                    cat_loss[cat] = cat_loss.get(cat, 0.0) + loss_val
                    cat_tokens[cat] = (
                        cat_tokens.get(cat, 0) + example.unmasked_token_count
                    )
                for cat in sorted(cat_loss):
                    if cat_tokens[cat] > 0:
                        metrics_record[f"_loss_per_token/{cat}"] = (
                            cat_loss[cat] / cat_tokens[cat]
                        )
                total_tokens = sum(cat_tokens.values())
                if total_tokens > 0:
                    metrics_record["_loss_per_token"] = (
                        sum(cat_loss.values()) / total_tokens
                    )

                # このステップのカテゴリ別最小 logprob を計算する（未マスクトークンのみ）
                cat_min_lp: dict[str, float] = {}
                for i, example in enumerate(valid_examples):
                    cat = example.category
                    lp_data = logprobs_list[i].data
                    unmasked_lps = [v for v, m in zip(lp_data, target_masks[i]) if m]
                    cat_min = min(unmasked_lps) if unmasked_lps else 0.0
                    if cat not in cat_min_lp or cat_min < cat_min_lp[cat]:
                        cat_min_lp[cat] = cat_min
                for cat in sorted(cat_min_lp):
                    metrics_record[f"_min_logprob/{cat}"] = round(cat_min_lp[cat], 4)

                metrics_file.write(json.dumps(metrics_record) + "\n")
                metrics_file.flush()

                # サンプルごとの logprobs を保存する
                for i, example in enumerate(valid_examples):
                    lp_data = logprobs_list[i].data

                    LogprobRecord(
                        problem_id=example.problem_id,
                        segment=example.segment,
                        logprobs=[round(v, 4) for v in lp_data],
                    ).save(epoch_dir)

                    # エポック0で参照 logprobs を収集する
                    key = example.problem_id
                    if epoch == 0:
                        all_ref_logprobs[key] = list(lp_data)
                        # トークン ID は一度だけ保存する
                        tokens_dir = log_path / "tokens"
                        stem = example.segment.removesuffix(".jsonl")
                        tokens_path = tokens_dir / example.problem_id / f"{stem}.json"
                        tokens_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(tokens_path, "w") as tf:
                            json.dump(
                                {"tokens": batch_tokens[i], "mask": batch_masks[i]}, tf
                            )
                    all_prev_logprobs[key] = list(lp_data)

                    unmasked_lps_i = [v for v, m in zip(lp_data, target_masks[i]) if m]
                    record = IndexRecord(
                        epoch=epoch,
                        step=step,
                        problem_id=example.problem_id,
                        segment=example.segment,
                        category=example.category,
                        num_loss_tokens=example.unmasked_token_count,
                        total_loss=round(sum(-v for v in unmasked_lps_i), 4),
                        min_logprob=round(min(unmasked_lps_i), 4)
                        if unmasked_lps_i
                        else 0.0,
                    )
                    index_file.write(json.dumps(dataclasses.asdict(record)) + "\n")

                step += 1

            epoch_elapsed = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_elapsed:.1f}s")

            # エポックごとの損失指標を計算して保存する
            epoch_metrics = compute_epoch_metrics(
                loss_config=cfg.loss_config,
                examples=examples,
                all_ref_logprobs=all_ref_logprobs,
                all_epoch_logprobs=all_prev_logprobs,
                epoch=epoch,
            )
            loss_record = {"epoch": epoch, "metrics": epoch_metrics}
            loss_file.write(json.dumps(loss_record) + "\n")
            loss_file.flush()

    # 最終チェックポイントを保存する
    await training_client.save_checkpoint_async(
        name="final",
        log_path=str(log_path),
    )

    logger.info(f"Training completed. Logprobs saved to {logprob_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
