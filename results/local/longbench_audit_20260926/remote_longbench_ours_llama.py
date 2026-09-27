#!/usr/bin/env python3
"""LongBench original English tasks with the existing Llama Ours runner."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "gov_report", "qmsum", "trec", "repobench-p"]
MAX_NEW = {
    "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "gov_report": 512, "qmsum": 512,
    "trec": 64, "repobench-p": 64,
}


def middle_truncate(tokenizer, prompt: str, max_tokens: int) -> str:
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() <= max_tokens:
        return prompt
    half = max_tokens // 2
    return tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
        ids[-(max_tokens - half):], skip_special_tokens=True
    )


def make_prompt(tokenizer, row: dict, template: str, max_context: int, chat: bool) -> str:
    prompt = template.format(**row)
    if row.get("input", "") and "{input}" not in template:
        prompt += row["input"]
    if chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return middle_truncate(tokenizer, prompt, max_context)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prompt-config", required=True)
    ap.add_argument("--runner-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=80)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    sys.path.insert(0, args.runner_dir)
    import benchmark_autoregressive_lowmem_llama as bench
    from llama_8b_ruler_lowmem import generate_ours
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    prompts = json.loads(Path(args.prompt_config).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    model_max = max(args.max_context + max(MAX_NEW[t] for t in tasks) + 64, 1024)
    runner = bench.load_runner(args.model_path, "ours_direct_v2", args.m, args.device,
                               max_tokens=model_max, attn_implementation="sdpa",
                               logits_to_keep=1)
    print(json.dumps({"device": args.device, "gpu": torch.cuda.get_device_name(),
                      "m": args.m, "max_context": args.max_context, "tasks": tasks}), flush=True)
    for task in tasks:
        source = Path(args.data_dir) / f"{task}.jsonl"
        rows = [json.loads(line) for line in source.open()]
        if args.limit:
            rows = rows[:args.limit]
        out_path = out_dir / f"{task}.jsonl"
        existing = 0
        if out_path.exists():
            existing = sum(1 for _ in out_path.open())
        if existing >= len(rows):
            print(f"[skip] {task} {existing}/{len(rows)}", flush=True)
            continue
        mode = "a" if existing else "w"
        with out_path.open(mode, encoding="utf-8") as out:
            for i, row in enumerate(rows[existing:], existing):
                chat = task not in {"trec", "repobench-p"}
                prompt = make_prompt(tokenizer, row, prompts[task], args.max_context, chat)
                max_new = MAX_NEW[task]
                start = time.perf_counter()
                pred, setup_s, decode_s, prompt_tokens = generate_ours(
                    runner, tokenizer, prompt, max_new, args.device
                )
                result = {"pred": pred, "answers": row["answers"],
                          "all_classes": row.get("all_classes", []),
                          "length": row.get("length", 0),
                          "index": i, "prompt_tokens": prompt_tokens,
                          "setup_s": setup_s, "decode_s": decode_s,
                          "elapsed_s": time.perf_counter() - start}
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{task} {i+1}/{len(rows)} prompt={prompt_tokens} "
                      f"setup={setup_s:.2f}s decode={decode_s:.2f}s", flush=True)
                if args.smoke:
                    return


if __name__ == "__main__":
    main()
