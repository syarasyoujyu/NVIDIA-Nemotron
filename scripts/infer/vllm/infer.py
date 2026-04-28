"""Run vLLM inference for Wonderland problems.

The script loads tokenized prompts from ``data/corpus_infer`` and writes raw
generations to JSONL. The evaluator runs by default after inference and writes
scored JSONL plus a summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
DEFAULT_TRITON_PTXAS_PATH = Path("/tmp/triton/backends/nvidia/bin/ptxas")
if DEFAULT_TRITON_PTXAS_PATH.exists():
    os.environ.setdefault("TRITON_PTXAS_PATH", str(DEFAULT_TRITON_PTXAS_PATH))

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from scripts.basic.common import load_jsonl as load_project_jsonl
from scripts.basic.const import CORPUS_INFER_DIR, CORPUS_INFER_INDEX, PROBLEMS_INDEX
from scripts.infer import evaluator

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
INFER_ROOT = ROOT / "data" / "inference" / "vllm"
DEFAULT_MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@dataclasses.dataclass
class ProblemRecord:
    problem_id: str
    category: str
    answer: str
    prompt_tokens: list[int]
    segment: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def default_output_dir(run: str | None, model_path: str, lora_path: str | None) -> Path:
    infer_time = datetime.now().strftime("%m-%d-%H-%M-%S")
    if run:
        model_name = run
    elif lora_path:
        model_name = Path(lora_path).name or str(Path(lora_path).parent.name)
    else:
        model_name = model_path
    model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return INFER_ROOT / (model_name[:96] or "model") / infer_time


def load_problem_records(
    *,
    categories: set[str] | None,
    statuses: set[str] | None,
    ids: set[str] | None,
    limit: int | None,
) -> list[ProblemRecord]:
    corpus_by_problem = {
        entry["problem_id"]: entry
        for entry in load_project_jsonl(CORPUS_INFER_INDEX)
        if entry["included"]
    }
    index_rows = load_jsonl(PROBLEMS_INDEX)
    selected = []
    for row in index_rows:
        pid = str(row["id"])
        category = str(row["category"])
        status = str(row.get("status", ""))
        if ids is not None and pid not in ids:
            continue
        if categories is not None and category not in categories:
            continue
        if statuses is not None and status not in statuses:
            continue
        selected.append((pid, category))
        if limit is not None and len(selected) >= limit:
            break

    records = []
    for pid, category in selected:
        corpus_entry = corpus_by_problem.get(pid)
        if corpus_entry is None:
            raise FileNotFoundError(
                f"No included tokenized inference corpus entry for {pid}. "
                "Run scripts/gen_data/gen_corpus.py first."
            )
        prompt_tokens = load_prompt_tokens_from_corpus_infer(corpus_entry)
        records.append(
            ProblemRecord(
                problem_id=pid,
                category=category,
                answer=str(corpus_entry["answer"]),
                prompt_tokens=prompt_tokens,
                segment=str(corpus_entry["segment"]),
            )
        )
    return records


def load_prompt_tokens_from_corpus_infer(corpus_entry: dict[str, Any]) -> list[int]:
    segment_path = CORPUS_INFER_DIR / corpus_entry["problem_id"] / corpus_entry["segment"]
    segments = load_project_jsonl(segment_path)
    prompt_tokens = []
    for segment in segments:
        prompt_tokens.extend(segment["tokens"])
    if not prompt_tokens:
        raise ValueError(f"No prompt tokens found for {corpus_entry['problem_id']}")
    return prompt_tokens


def cache_model(
    path: str | Path,
    *,
    num_workers: int | None = None,
    chunk_mb: int = 256,
    exts: tuple[str, ...] = (".bin", ".pt", ".safetensors"),
) -> int:
    model_path = Path(path)
    if model_path.is_dir():
        files = sorted(
            p for p in model_path.rglob("*") if p.is_file() and str(p).endswith(exts)
        )
    else:
        files = [model_path] if model_path.exists() else []
    if not files:
        print(f"No local model weight files found to cache at: {model_path}")
        return 0

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    chunk_size = chunk_mb * 1024 * 1024

    def warmup_file(path_to_read: Path) -> int:
        total = 0
        with path_to_read.open("rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return total

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    started = time.perf_counter()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, path): path for path in files}
        for i, future in enumerate(as_completed(futures), 1):
            bytes_read = future.result()
            total_bytes += bytes_read
            print(f"[{i}/{len(files)}] cached {futures[future].name}")
    elapsed = time.perf_counter() - started
    total_gb = total_bytes / 1024**3
    speed = total_gb / elapsed if elapsed else 0.0
    print(f"[cache_model] total read ~= {total_gb:.2f} GB")
    print(f"[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s")
    return total_bytes


def serialize_logprobs(
    logprobs: list[dict[int, Any]] | None,
) -> list[dict[str, Any] | None] | None:
    if logprobs is None:
        return None

    serialized: list[dict[str, Any] | None] = []
    for token_logprobs in logprobs:
        if token_logprobs is None:
            serialized.append(None)
            continue
        serialized.append(
            {
                str(token_id): {
                    "logprob": getattr(value, "logprob", None),
                    "rank": getattr(value, "rank", None),
                    "decoded_token": getattr(value, "decoded_token", None),
                }
                for token_id, value in token_logprobs.items()
            }
        )
    return serialized


def min_logprob(logprobs: list[dict[int, Any]] | None) -> float | None:
    if not logprobs:
        return None
    values = [
        float(value.logprob)
        for token_logprobs in logprobs
        if token_logprobs
        for value in token_logprobs.values()
        if getattr(value, "logprob", None) is not None
    ]
    return min(values) if values else None


def completion_to_record(
    *,
    problem: ProblemRecord,
    run: str | None,
    model_path: str,
    lora_path: str | None,
    problem_index: int,
    sample_index: int,
    completion,
    elapsed: float,
) -> dict[str, Any]:
    tokens = list(completion.token_ids)
    text = str(completion.text)
    predicted_answer = evaluator.extract_final_answer(text)
    raw_logprobs = getattr(completion, "logprobs", None)
    cumulative_logprob = getattr(completion, "cumulative_logprob", None)
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = getattr(completion, "stop_reason", None)
    return {
        "problem_id": problem.problem_id,
        "category": problem.category,
        "answer": problem.answer,
        "run": run,
        "model_path": model_path,
        "lora_path": lora_path,
        "problem_index": problem_index,
        "sample_index": sample_index,
        "segment": problem.segment,
        "prompt_token_count": len(problem.prompt_tokens),
        "generated_token_count": len(tokens),
        "generated_tokens": tokens,
        "logprobs": serialize_logprobs(raw_logprobs),
        "min_logprob": min_logprob(raw_logprobs),
        "cumulative_logprob": cumulative_logprob,
        "finish_reason": str(finish_reason) if finish_reason is not None else "",
        "stop_reason": str(stop_reason) if stop_reason is not None else "",
        "elapsed": round(elapsed, 3),
        "output_text": text,
        "predicted_answer": predicted_answer,
    }


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    logprobs = args.logprobs if args.logprobs and args.logprobs > 0 else None
    return SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=logprobs,
    )


def build_llm(args: argparse.Namespace) -> LLM:
    if args.cache_model:
        cache_model(
            args.model_path,
            num_workers=args.cache_workers,
            chunk_mb=args.cache_chunk_mb,
        )

    if args.transformers_offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.triton_ptxas_path:
        os.environ["TRITON_PTXAS_PATH"] = args.triton_ptxas_path

    return LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=not args.no_trust_remote_code,
        enable_lora=bool(args.lora_path),
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )


def build_lora_request(args: argparse.Namespace) -> LoRARequest | None:
    if not args.lora_path:
        return None
    return LoRARequest(args.adapter_name, args.lora_id, args.lora_path)


def run_inference(args: argparse.Namespace) -> Path:
    records = load_problem_records(
        categories=set(args.category) if args.category else None,
        statuses=set(args.status) if args.status else None,
        ids=set(args.id) if args.id else None,
        limit=args.limit,
    )
    if not records:
        raise ValueError("No problems selected.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_output_dir(args.run, args.model_path, args.lora_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or args.max_num_seqs
    if batch_size < 1:
        raise ValueError("Batch size must be >= 1.")

    config = {
        "backend": "vllm",
        "run": args.run,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "adapter_name": args.adapter_name if args.lora_path else None,
        "lora_id": args.lora_id if args.lora_path else None,
        "num_problems": len(records),
        "num_samples": args.num_samples,
        "prompt_source": "data/corpus_infer tokenized prompt_token_ids",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": args.stop,
        "logprobs": args.logprobs,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "batch_size": batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "categories": args.category,
        "statuses": args.status,
        "ids": args.id,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sampling_params = build_sampling_params(args)
    llm = build_llm(args)
    lora_request = build_lora_request(args)
    out_path = output_dir / "generations.jsonl"

    completed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompt_token_ids = [problem.prompt_tokens for problem in batch]
            started = time.perf_counter()
            outputs = llm.generate(
                prompts=None,
                sampling_params=sampling_params,
                prompt_token_ids=prompt_token_ids,
                lora_request=lora_request,
            )
            elapsed = time.perf_counter() - started

            for batch_offset, (problem, request_output) in enumerate(
                zip(batch, outputs)
            ):
                for sample_index, completion in enumerate(request_output.outputs):
                    row = completion_to_record(
                        problem=problem,
                        run=args.run,
                        model_path=args.model_path,
                        lora_path=args.lora_path,
                        problem_index=start + batch_offset,
                        sample_index=sample_index,
                        completion=completion,
                        elapsed=elapsed,
                    )
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            completed += len(batch)
            print(f"{completed}/{len(records)} sampled")

    if not args.no_eval:
        evaluator.evaluate_file(out_path)

    print(f"Wrote generations to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM inference.")
    parser.add_argument(
        "--run",
        help="Optional train-run label used for output directory grouping.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Base model path or HF repo name passed to vLLM.",
    )
    parser.add_argument(
        "--lora-path",
        help="PEFT LoRA adapter directory. If omitted, the base model is used.",
    )
    parser.add_argument("--adapter-name", default="adapter")
    parser.add_argument("--lora-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category", action="append")
    parser.add_argument("--status", action="append")
    parser.add_argument("--id", action="append")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=["<|im_end|>"])
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--transformers-offline", action="store_true")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--triton-ptxas-path")
    parser.add_argument("--cache-model", action="store_true")
    parser.add_argument("--cache-workers", type=int)
    parser.add_argument("--cache-chunk-mb", type=int, default=1024)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Only write generations.jsonl; skip evaluator.py.",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1.")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be >= 1.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.lora_path and not Path(args.lora_path).exists():
        parser.error(f"--lora-path does not exist: {args.lora_path}")
    return args


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
