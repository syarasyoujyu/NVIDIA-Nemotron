"""tinker と modal の両バックエンドに対応するサービス/学習クライアント。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import modal
from tinker.types import AdamParams

T = TypeVar("T")


@dataclass
class LogprobsData:
    """tinker の logprobs と同じ形にする。"""

    data: list[float]


class Future(Generic[T]):
    """tinker の非同期実行結果インターフェースに合わせた、解決済みの簡易結果。"""

    def __init__(self, value: T) -> None:
        self._value = value

    async def result_async(self) -> T:
        return self._value


class TrainingClient:
    """tinker または modal バックエンドへ処理を委譲する学習クライアント。"""

    def __init__(
        self,
        backend: Literal["tinker", "modal"],
        tinker_client: object | None = None,
        modal_remote: modal.cls.Obj | None = None,
    ) -> None:
        self._backend = backend
        self._tinker_client = tinker_client
        self._modal_remote = modal_remote

    async def forward_backward_async(
        self,
        data: list,
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, float] | None = None,
        micro_batch_size: int | None = None,
    ) -> Future:
        if loss_fn_config is None:
            loss_fn_config = {}

        if self._backend == "tinker":
            assert self._tinker_client is not None
            return await self._tinker_client.forward_backward_async(  # type: ignore[attr-defined]
                data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            )

        # modal: リモート呼び出し用に学習用データを通常のリストへシリアライズする
        batch_tokens: list[list[int]] = []
        batch_target_tokens: list[list[int]] = []
        batch_weights: list[list[float]] = []

        for datum in data:
            batch_tokens.append(datum.model_input.chunks[0].tokens)
            batch_target_tokens.append(datum.loss_fn_inputs["target_tokens"].data)
            if "weights" in datum.loss_fn_inputs:
                batch_weights.append(datum.loss_fn_inputs["weights"].data)
            elif "advantages" in datum.loss_fn_inputs:
                batch_weights.append(datum.loss_fn_inputs["advantages"].data)
            else:
                batch_weights.append([1.0] * len(batch_target_tokens[-1]))

        assert self._modal_remote is not None
        raw = await self._modal_remote.forward_backward.remote.aio(
            batch_tokens=batch_tokens,
            batch_target_tokens=batch_target_tokens,
            batch_weights=batch_weights,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            micro_batch_size=micro_batch_size,
        )

        # tinker の結果インターフェースに合わせて生の辞書をラップする
        loss_fn_outputs = [
            {"logprobs": LogprobsData(data=entry["logprobs"])}
            for entry in raw["loss_fn_outputs"]
        ]
        result = _ForwardBackwardResult(
            loss_fn_outputs=loss_fn_outputs,
            metrics=raw.get("metrics", {}),
        )
        return Future(result)

    async def optim_step_async(self, adam_params: AdamParams) -> Future:
        if self._backend == "tinker":
            assert self._tinker_client is not None
            return await self._tinker_client.optim_step_async(adam_params)  # type: ignore[attr-defined]

        assert self._modal_remote is not None
        raw = await self._modal_remote.optim_step.remote.aio(
            learning_rate=adam_params.learning_rate,
            beta1=adam_params.beta1,
            beta2=adam_params.beta2,
            eps=adam_params.eps,
            weight_decay=adam_params.weight_decay,
            grad_clip_norm=adam_params.grad_clip_norm,
        )
        return Future(_OptimResult(metrics=raw))

    async def load_state_async(
        self,
        path: str,
        *,
        with_optimizer: bool = True,
    ) -> None:
        if self._backend != "tinker":
            raise NotImplementedError(
                "Loading pretrained state is only supported by tinker"
            )

        assert self._tinker_client is not None
        if with_optimizer:
            future = await self._tinker_client.load_state_with_optimizer_async(  # type: ignore[attr-defined]
                path
            )
        else:
            future = await self._tinker_client.load_state_async(path)  # type: ignore[attr-defined]
        await future.result_async()

    async def save_checkpoint_async(self, name: str, log_path: str) -> None:
        if self._backend == "tinker":
            from tinker_cookbook import checkpoint_utils

            loop_state = {"name": name}
            await checkpoint_utils.save_checkpoint_async(
                training_client=self._tinker_client,
                name=name,
                log_path=log_path,
                kind="both",
                loop_state=loop_state,
            )
        else:
            assert self._modal_remote is not None
            await self._modal_remote.save_checkpoint.remote.aio(path="/adapter/weights")



class ServiceClient:
    """tinker と modal の両バックエンドに対応するサービスクライアント。"""

    def __init__(self, backend: Literal["tinker", "modal"] = "tinker") -> None:
        self.backend = backend

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ) -> TrainingClient:
        if self.backend == "modal":
            remote = modal.Cls.from_name("trainer-gpu", "Trainer")()
            # モデル初期化はコンテナ起動時に @modal.enter() で行われる。
            # 最初のメソッド呼び出しは、初期化完了まで待機する。
            return TrainingClient(backend="modal", modal_remote=remote)

        import tinker

        tinker_sc = tinker.ServiceClient()
        tinker_tc = await tinker_sc.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata={"save_every":"100"},
        )
        return TrainingClient(backend="tinker", tinker_client=tinker_tc)


@dataclass
class _ForwardBackwardResult:
    loss_fn_outputs: list[dict[str, LogprobsData]]
    metrics: dict[str, float]


@dataclass
class _OptimResult:
    metrics: dict[str, float]
