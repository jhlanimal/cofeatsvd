#!/usr/bin/env python3
"""Decode-only CUDA stage breakdown; diagnostic, not official latency."""
import csv, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'optimize'))
from analyze_post_rope_k_pages import DEFAULT_DATA, load_tokens
from benchmark_autoregressive_full_vs_ours import load_runner

MODEL='/data/share/Qwen3-4B-Instruct-2507'
OUT=ROOT/'optimize/results/direct_attention_v2/current_breakdown.csv'
INIT=128; WARMUP=28; PROFILE_TOKENS=8

def wrap_attention(model, name):
    original=[]
    for layer in model.model.layers:
        attn=layer.self_attn; fn=attn.forward
        def wrapped(*args, _fn=fn, **kwargs):
            with torch.autograd.profiler.record_function(name):
                return _fn(*args, **kwargs)
        attn.forward=wrapped; original.append((attn,fn))
    return original

def restore(original):
    for attn,fn in original: attn.forward=fn

def profile_full(ids):
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,attn_implementation='eager').eval().cuda()
    past=model(input_ids=ids[:INIT].view(1,-1),use_cache=True).past_key_values
    pos_cache=torch.arange(ids.numel(),device=ids.device,dtype=torch.long)
    for pos in range(INIT,INIT+WARMUP):
        past=model(input_ids=ids[pos].view(1,1),past_key_values=past,use_cache=True,cache_position=pos_cache[pos:pos+1]).past_key_values
    original=wrap_attention(model,'full_attention')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as p:
        for pos in range(INIT+WARMUP,INIT+WARMUP+PROFILE_TOKENS):
            with torch.autograd.profiler.record_function('decode_step'):
                past=model(input_ids=ids[pos].view(1,1),past_key_values=past,use_cache=True,cache_position=pos_cache[pos:pos+1]).past_key_values
        torch.cuda.synchronize()
    restore(original); rows=p.key_averages()
    out=extract(rows,'Full-native', 'full_attention')
    del model; torch.cuda.empty_cache(); return out

def profile_ours(ids):
    runner=load_runner(MODEL,'ours_reconstruct_native_bf16',80,'cuda:0')
    runner.prefill(ids[:INIT],torch.arange(INIT,device=ids.device)); runner.seal_prefill()
    for pos in range(INIT,INIT+WARMUP): runner.run_token(ids[pos],pos)
    original=wrap_attention(runner.model,'ours_attention')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as p:
        for pos in range(INIT+WARMUP,INIT+WARMUP+PROFILE_TOKENS):
            with torch.autograd.profiler.record_function('decode_step'):
                runner.run_token(ids[pos],pos)
        torch.cuda.synchronize()
    restore(original); rows=p.key_averages()
    out=extract(rows,'Ours-v2-BF16','ours_attention')
    del runner.model; torch.cuda.empty_cache(); return out

def extract(events,path,attention_name):
    vals={e.key:float(e.cuda_time_total)/1000.0/PROFILE_TOKENS for e in events}
    total=vals.get('decode_step',0.0)
    attention=vals.get(attention_name,0.0)
    row={'path':path,'profile_tokens':PROFILE_TOKENS,'decode_cuda_ms_per_token':total,'attention_cuda_ms_per_token':attention,'non_attention_cuda_ms_per_token':max(total-attention,0.0)}
    for name in ('ours_qkv_norm_rope','ours_reconstruct_k','ours_native_attention','ours_o_proj','ours_page_seal'):
        row[name+'_ms_per_token']=vals.get(name,0.0)
    for key in ('aten::mm','aten::copy_','aten::cat','aten::arange','aten::linalg_eigh','aten::_linalg_eigh','cudaLaunchKernel','Memcpy DtoD (Device -> Device)'):
        row[key]=vals.get(key,0.0)
    return row

def main():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
    ids=load_tokens(tok,DEFAULT_DATA,INIT+WARMUP+PROFILE_TOKENS).to('cuda:0')
    rows=[profile_full(ids),profile_ours(ids)]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
