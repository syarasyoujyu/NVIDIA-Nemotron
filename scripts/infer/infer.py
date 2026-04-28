"""Run Tinker inference for Wonderland problems.

The script samples from either a direct Tinker model path or a saved training
run under ``data/training/sft/<run>``. Raw generations are written to JSONL.
The evaluator runs by default after inference and writes scored JSONL plus a
summary.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import tinker
from dotenv import load_dotenv
from tinker import types
from transformers import AutoTokenizer

from scripts.basic.common import load_jsonl as load_project_jsonl
from scripts.basic.const import CORPUS_INFER_DIR, CORPUS_INFER_INDEX, PROBLEMS_INDEX
from scripts.infer import evaluator

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "data" / "training" / "sft"
INFER_ROOT = ROOT / "data" / "inference"
DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


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


def resolve_model_path(run: str | None, model_path: str | None) -> str:
    if model_path:
        return model_path
    if not run:
        raise ValueError("Pass --run or --model-path.")

    checkpoints_path = SFT_ROOT / run / "checkpoints.jsonl"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {checkpoints_path}")

    checkpoints = load_jsonl(checkpoints_path)
    if not checkpoints:
        raise ValueError(f"No checkpoint records in {checkpoints_path}")
    latest = checkpoints[-1]
    path = latest.get("sampler_path") or latest.get("state_path")
    if not path:
        raise ValueError(f"No sampler_path/state_path in {checkpoints_path}")
    return str(path)


def default_output_dir(run: str | None, model_path: str) -> Path:
    infer_time = datetime.now().strftime("%m-%d-%H-%M-%S")
    if run:
        model_name = run
    else:
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path).strip("_")
        model_name = model_name[:96] or "model-path"
    return INFER_ROOT / model_name / infer_time


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


def result_sequences(result) -> list:
    samples = getattr(result, "samples", None)
    if samples is not None:
        return list(samples)
    return list(result.sequences)


def sequence_to_record(
    *,
    problem: ProblemRecord,
    run: str | None,
    model_path: str,
    problem_index: int,
    sample_index: int,
    prompt_token_count: int,
    sequence,
    tokenizer,
    elapsed: float,
) -> dict[str, Any]:
    tokens = list(sequence.tokens)
    text = tokenizer.decode(tokens)
    predicted_answer = evaluator.extract_final_answer(text)
    logprobs = getattr(sequence, "logprobs", None)
    stop_reason = getattr(sequence, "stop_reason", None)
    return {
        "problem_id": problem.problem_id,
        "category": problem.category,
        "answer": problem.answer,
        "run": run,
        "model_path": model_path,
        "problem_index": problem_index,
        "sample_index": sample_index,
        "segment": problem.segment,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": len(tokens),
        "generated_tokens": tokens,
        "logprobs": list(logprobs) if logprobs is not None else None,
        "stop_reason": str(stop_reason) if stop_reason is not None else "",
        "elapsed": round(elapsed, 3),
        "output_text": text,
        "predicted_answer": predicted_answer,
    }


async def sample_problem(
    *,
    sampling_client,
    tokenizer,
    problem: ProblemRecord,
    problem_index: int,
    run: str | None,
    model_path: str,
    sampling_params,
    num_samples: int,
) -> list[dict[str, Any]]:
    prompt = types.ModelInput.from_ints(problem.prompt_tokens)
    start = time.perf_counter()
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=sampling_params,
    )
    elapsed = time.perf_counter() - start
    return [
        sequence_to_record(
            problem=problem,
            run=run,
            model_path=model_path,
            problem_index=problem_index,
            sample_index=i,
            prompt_token_count=len(problem.prompt_tokens),
            sequence=sequence,
            tokenizer=tokenizer,
            elapsed=elapsed,
        )
        for i, sequence in enumerate(result_sequences(result))
    ]


async def run_inference(args: argparse.Namespace) -> Path:
    model_path = resolve_model_path(args.run, args.model_path)
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
        output_dir = default_output_dir(args.run, model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "backend": "tinker",
        "run": args.run,
        "model_path": model_path,
        "base_model": args.base_model,
        "num_problems": len(records),
        "num_samples": args.num_samples,
        "prompt_source": "data/corpus_infer tokenized prompts",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": args.stop,
        "concurrency": args.concurrency,
        "categories": args.category,
        "statuses": args.status,
        "ids": args.id,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        model_path=model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
    )

    out_path = output_dir / "generations.jsonl"

    completed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), args.concurrency):
            batch = records[start : start + args.concurrency]
            batch_results = await asyncio.gather(
                *[
                    sample_problem(
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        problem=problem,
                        problem_index=start + i,
                        run=args.run,
                        model_path=model_path,
                        sampling_params=sampling_params,
                        num_samples=args.num_samples,
                    )
                    for i, problem in enumerate(batch)
                ]
            )
            for rows in batch_results:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            completed += len(batch)
            print(f"{completed}/{len(records)} sampled")

    if not args.no_eval:
        evaluator.evaluate_file(out_path)

    print(f"Wrote generations to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tinker inference.")
    parser.add_argument("--run", help="SFT run name under data/training/sft.")
    parser.add_argument("--model-path", help="Direct tinker:// sampler model path.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
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
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Only write generations.jsonl; skip evaluator.py.",
    )
    args = parser.parse_args()
    if not args.run and not args.model_path:
        parser.error("Pass --run or --model-path.")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1.")
    return args


def main() -> None:
    asyncio.run(run_inference(parse_args()))


if __name__ == "__main__":
    main()
