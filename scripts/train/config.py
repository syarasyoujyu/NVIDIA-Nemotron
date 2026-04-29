from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from tinker import types

from scripts.train.loss_config import (
    CrossEntropyLossConfig,
    LossConfig,
)
from scripts.train.lr_schedule import LRSchedule, StepLinearDecayLRSchedule


@dataclasses.dataclass
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1e9

    def to_adam_params(self, learning_rate: float) -> types.AdamParams:
        return types.AdamParams(
            learning_rate=learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )


@dataclasses.dataclass
class Cfg:
    loss_config: LossConfig = dataclasses.field(default_factory=CrossEntropyLossConfig)
    lr_schedule: LRSchedule = dataclasses.field(
        default_factory=lambda: StepLinearDecayLRSchedule(learning_rate=2e-4)
    )
    log_path: str = dataclasses.field(
        default_factory=lambda: datetime.now().strftime("%m-%d-%H-%M")
    )
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    batch_size: int = 64
    num_epochs: int = 1
    lora_rank: int = 32 # 32
    max_length: int = 8192
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    adam_config: AdamConfig = dataclasses.field(default_factory=AdamConfig)
    backend: Literal["tinker", "modal"] = "tinker"
    micro_batch_size: int | None = 16  # tinker では None（サーバー側で決定）、modal では整数
    from_pretrained: bool = False
    pretrained_path: str | None = None
    pretrained_load_optimizer: bool = True
    cot_prompt_filter_mode: Literal["all", "incorrect", "correct"] = "all"
    batch_stratify_by: Literal["category", "task_type"] = "task_type"
    category_limit_counts: list[int | None] | None = None
    task_type_limit_counts: list[int | None] | None = None
    task_type_limit_strategy: Literal["head", "random"] = "head"
    task_type_limit_seed: int = 0


@dataclasses.dataclass
class LogprobRecord:
    problem_id: str
    segment: str
    logprobs: list[float]

    def save(self, epoch_dir: Path) -> None:
        stem = self.segment.removesuffix(".jsonl")
        path = epoch_dir / self.problem_id / f"{stem}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"logprobs": self.logprobs}) + "\n")


@dataclasses.dataclass
class IndexRecord:
    epoch: int
    step: int
    problem_id: str
    segment: str
    category: str
    num_loss_tokens: int
    total_loss: float
    min_logprob: float
