#!/usr/bin/env python3
"""Profiler for the isolated grouped-compression implementations."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path
import torch
from benchmark_compression import reference
from optimized_grouped_compression import compress_optimized_exact

OUT = Path(__file__).parent / "results"

def timed(fn, warmup=100, iters=1000):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); xs=[]
    for _ in range(iters):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); xs.append(a.elapsed_time(b))
    xs.sort(); return {'mean_ms':statistics.mean(xs),'median_ms':statistics.median(xs),'p50_ms':xs[len(xs)//2],'p95_ms':xs[int(.95*(len(xs)-1))],'min_ms':min(xs)}

def prof(fn, path):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False) as p:
        for _ in range(20): fn()
    path.write_text(p.key_averages().table(sort_by='cuda_time_total', row_limit=100))
    return p.key_averages()

def main():
    dev='cuda:0'; torch.manual_seed(0); k=torch.randn(8,32,128,device=dev); q=torch.randn(8,4,32,128,device=dev)
    rows=[]
    for m in (64,80,96):
        for name, fn in [('reference',lambda m=m:[reference(k[i],q[i],m) for i in range(8)]),('optimized_exact',lambda m=m:[compress_optimized_exact(k[i],q[i],m,True) for i in range(8)])]:
            torch.cuda.reset_peak_memory_stats(); s=timed(fn); s.update(method=name,m=m,peak_memory=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()); rows.append(s)
            prof(fn, OUT/f'profiler_{name}_m{m}.txt')
    OUT.mkdir(exist_ok=True)
    with (OUT/'profiler_latency.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (OUT/'sync_audit.json').write_text(json.dumps({'optimized_dynamic_gpu_to_cpu':['torch.unique(counts).tolist()'], 'gpu_sync_risk':['Python loop over group sizes','torch.unique(...).tolist()'], 'source_note':'optimized_grouped_compression.py'} ,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
