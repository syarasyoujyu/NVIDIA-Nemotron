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

from scripts.basic.const import SFT_DIR
from scripts.train.base import (
    CATEGORY_ORDER,
    TASK_TYPE_ORDER,
    TrainingExample,
    build_datum,
    load_corpus_entries,
)
from scripts.train.config import Cfg, IndexRecord, LogprobRecord
from scripts.train.utils import (
    compute_epoch_metrics,
    filter_examples_by_cot_prompt,
    filter_training_examples,
    limit_examples_by_category,
    limit_examples_by_task_type,
    parse_category_limit_counts,
    parse_task_type_limit_counts,
    stratified_batches,
)
from scripts.trainer.client import ServiceClient

load_dotenv()
logger = logging.getLogger(__name__)


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
    p.add_argument(
        "--from_pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load a saved Tinker training state before starting SFT.",
    )
    p.add_argument(
        "--pretrained_path",
        type=str,
        help="Tinker state checkpoint path used when --from_pretrained is enabled.",
    )
    p.add_argument(
        "--pretrained_load_optimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Restore optimizer state with the weights. "
            "Use --no-pretrained_load_optimizer to load weights only."
        ),
    )
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
        type=parse_category_limit_counts,
        help=(
            "Per-category maximum counts in fixed category order. "
            "Accepts comma-separated values with blanks, or a JSON list. "
            f"Order: {', '.join(CATEGORY_ORDER)}"
        ),
    )
    p.add_argument(
        "--task_type_limit_counts",
        type=parse_task_type_limit_counts,
        help=(
            "Per-task-type maximum counts in fixed task type order. "
            "Accepts comma-separated values with blanks, or a JSON list. "
            f"Order: {', '.join(TASK_TYPE_ORDER)}"
        ),
    )
    p.add_argument(
        "--task_type_limit_strategy",
        choices=["head", "random"],
        help=(
            "How to choose examples when --task_type_limit_counts is set. "
            "head keeps the first 0,1,2,... examples in corpus order; "
            "random samples examples per task type."
        ),
    )
    p.add_argument(
        "--task_type_limit_seed",
        type=int,
        help="Random seed used when --task_type_limit_strategy=random.",
    )
    args = p.parse_args()

    cfg = Cfg()
    for field in [
        "num_epochs", "batch_size", "lora_rank", "max_length",
        "log_path", "backend", "micro_batch_size",
        "from_pretrained", "pretrained_path", "pretrained_load_optimizer",
        "train_mlp", "train_attn", "train_unembed",
        "cot_prompt_filter_mode", "batch_stratify_by",
        "category_limit_counts", "task_type_limit_counts",
        "task_type_limit_strategy", "task_type_limit_seed",
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
    if cfg.from_pretrained and not cfg.pretrained_path:
        p.error("--pretrained_path is required when --from_pretrained is enabled")
    if not cfg.from_pretrained and cfg.pretrained_path:
        p.error("--pretrained_path can only be used with --from_pretrained")
    return cfg


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
        strategy=cfg.task_type_limit_strategy,
        seed=cfg.task_type_limit_seed,
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
    if cfg.from_pretrained:
        assert cfg.pretrained_path is not None
        optimizer_msg = (
            "with optimizer state" if cfg.pretrained_load_optimizer else "weights only"
        )
        logger.info(
            f"Loading pretrained training state from {cfg.pretrained_path} "
            f"({optimizer_msg})"
        )
        await training_client.load_state_async(
            cfg.pretrained_path,
            with_optimizer=cfg.pretrained_load_optimizer,
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
            batches = stratified_batches(
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
