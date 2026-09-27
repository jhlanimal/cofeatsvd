#!/usr/bin/env python3
"""Run the English LongBench tasks with Qwen3-32B on a six-GPU shard."""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "gov_report", "qmsum", "trec", "repobench-p"]
CHAT_TASKS = set(TASKS) - {"trec", "repobench-p"}


def key(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def middle_truncate(tok, text: str, limit: int) -> str:
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() <= limit:
        return text
    left = limit // 2
    return tok.decode(ids[:left], skip_special_tokens=True) + tok.decode(
        ids[-(limit - left):], skip_special_tokens=True)


def install_contexts(model):
    for module in [*model.model.layers, model.model.norm, model.lm_head]:
        original = module.forward
        device = next(module.parameters()).device

        @functools.wraps(original)
        def forward(*args, _original=original, _device=device, **kwargs):
            with torch.cuda.device(_device):
                return _original(*args, **kwargs)

        module.forward = forward


def install_chunked_mlp(model, chunk=512):
    for layer in model.model.layers:
        original = layer.mlp.forward

        def forward(x, _original=original):
            if x.shape[1] <= chunk:
                return _original(x)
            out = torch.empty_like(x)
            for start in range(0, x.shape[1], chunk):
                out[:, start:start + chunk] = _original(x[:, start:start + chunk])
            return out

        layer.mlp.forward = forward


def sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def make_mapping(cfg, ndev):
    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0,
               "model.norm": ndev - 1, "lm_head": ndev - 1}
    mapping.update({f"model.layers.{i}": min(i * ndev // cfg.num_hidden_layers, ndev - 1)
                    for i in range(cfg.num_hidden_layers)})
    return mapping


@torch.inference_mode()
def generate_full(model, tok, prompt, max_new, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    start = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    token = out.logits[:, -1].argmax(-1)
    sync_all()
    setup = time.perf_counter() - start
    generated = []
    dec = time.perf_counter()
    for _ in range(max_new):
        generated.append(int(token.item()))
        out = model(input_ids=token.view(1, 1).to(device), past_key_values=past,
                    use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        token = out.logits[:, -1].argmax(-1)
    sync_all()
    return tok.decode(generated, skip_special_tokens=True), setup, time.perf_counter() - dec, ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prompt-config", required=True)
    ap.add_argument("--maxlen-config", required=True)
    ap.add_argument("--runner-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["full", "ours"], required=True)
    ap.add_argument("--m", type=int, default=80)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--mlp-chunk", type=int, default=512)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    ndev = torch.cuda.device_count()
    if ndev != 6:
        raise RuntimeError(f"Expected six visible GPUs, got {ndev}")
    sys.path.insert(0, args.runner_dir)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__
    import qwen3_8b_ruler_lowmem as q
    import benchmark_autoregressive_lowmem as bench
    from run_qwen32b_ruler_multigpu import runner_class, install_gqa_adapter

    install_gqa_adapter(bench)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    prompts = json.loads(Path(args.prompt_config).read_text())
    maxlens = json.loads(Path(args.maxlen_config).read_text())
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    mapping = make_mapping(cfg, ndev)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), tasks=args.tasks.split(","), device_map=mapping,
                  physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  model=args.model_path, transformers=__version__, dtype="bfloat16", seed=42)
    cp = out / "config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise RuntimeError("Output configuration mismatch")
    cp.write_text(json.dumps(config, indent=2))

    model = None
    runner = None
    if args.method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
            device_map=mapping).eval()
        install_chunked_mlp(model, args.mlp_chunk)
        install_contexts(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa",
            device_map=mapping).eval()
        bench.install_rms(model)
        runner = runner_class(bench)(model, "ours_direct_v2", args.m,
                                     max_tokens=args.max_context + 640,
                                     logits_to_keep=1)
        install_contexts(model)
        install_chunked_mlp(model, args.mlp_chunk)

    summary = {}
    for task in [x for x in args.tasks.split(",") if x]:
        raw = Path(args.data_dir) / f"{task}.jsonl"
        rows = [json.loads(x) for x in raw.open()]
        if args.limit:
            rows = rows[:args.limit]
        prompt_template = prompts[task]
        dest = out / f"{task}.jsonl"
        saved = [json.loads(x) for x in dest.open()] if dest.exists() else []
        if len(saved) > len(rows):
            raise RuntimeError(f"Too many rows in {dest}")
        for i in range(len(saved), len(rows)):
            row = rows[i]
            prompt = prompt_template.format(**row)
            prompt = middle_truncate(tok, prompt, args.max_context)
            if task in CHAT_TASKS:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
            n = len(tok(prompt, add_special_tokens=False).input_ids)
            max_new = min(int(maxlens[task]), args.max_context - n)
            if max_new <= 0:
                raise RuntimeError(f"{task}/{i}: no generation room after truncation")
            sync_all()
            start = time.perf_counter()
            if args.method == "full":
                pred, setup, decode, prompt_tokens = generate_full(model, tok, prompt, max_new, "cuda:0")
            else:
                runner.reset_runtime()
                runner.workspaces = [(None, None) for _ in runner.states]
                runner.compact_stores = [None for _ in runner.states]
                runner.active_buffers = [(None, None) for _ in runner.states]
                runner.capacity = 0
                pred, setup, decode, prompt_tokens = q.generate_ours(
                    runner, tok, prompt, max_new, "cuda:0")
            result = dict(index=i, sample_id=row.get("_id", i), answers=row["answers"],
                          all_classes=row.get("all_classes", []), length=row.get("length", 0),
                          input_hash=key(row), pred=pred, prompt_tokens=prompt_tokens,
                          new_tokens=max_new, prompt_setup_s=setup, decode_s=decode,
                          elapsed_s=time.perf_counter() - start)
            with dest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary[task] = {"samples": i + 1, "target": len(rows)}
            (out / "summary_local.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps({"method": args.method, "task": task, "done": i + 1,
                              "target": len(rows), "prompt_tokens": prompt_tokens,
                              "setup_s": round(setup, 2), "decode_s": round(decode, 2)},
                             ensure_ascii=False), flush=True)
        summary[task] = {"samples": len(rows), "target": len(rows)}
        (out / "summary_local.json").write_text(json.dumps(summary, indent=2))
        print(f"DONE {task} {summary[task]}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
