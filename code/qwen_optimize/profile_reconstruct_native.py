#!/usr/bin/env python3
from __future__ import annotations
import sys, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'optimize'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from analyze_post_rope_k_pages import DEFAULT_DATA, load_tokens
from benchmark_autoregressive_full_vs_ours import load_runner

def main():
    device='cuda'; model_path='/data/share/Qwen3-4B-Instruct-2507'; init=128; decode=40
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    ids=load_tokens(tok,DEFAULT_DATA,init+decode).to(device)
    r=load_runner(model_path,'ours_reconstruct_native_bf16',80,device)
    out=r.prefill(ids[:init],torch.arange(init,device=device)); r.seal_prefill()
    for pos in range(init, init+8): r.run_token(ids[pos],pos)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as p:
        for pos in range(init+8,init+40): r.run_token(ids[pos],pos)
        torch.cuda.synchronize()
    text=p.key_averages().table(sort_by='self_cuda_time_total',row_limit=80)
    outp=ROOT/'optimize/results/reconstruct_optimization/profiler_v2.txt'; outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(text)
    print(text)

if __name__=='__main__': main()
