#!/usr/bin/env python3
import csv, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from triton_page_parallel_attention_v2 import compact_page_attention_v2
from triton_compact_decode_attention import reconstruct_k_triton

OUT = Path(__file__).resolve().parent / 'results/direct_attention_v2/page_parallel_v2.csv'

def timed(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(iters):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b)*1000)
    return statistics.mean(vals), statistics.median(vals)

def main():
    rows=[]; dev='cuda:0'; torch.manual_seed(2)
    for n in (4,8,12,16,32,64):
        q=torch.randn((32,128),device=dev); c=torch.randn((n,8,32,80),device=dev)
        coeff=torch.randn((n,8,128),device=dev); owner=torch.randint(0,80,(n,8,128),device=dev)
        hv=torch.randn((n,8,32,128),device=dev); ak=torch.randn((8,17,128),device=dev); av=torch.randn((8,17,128),device=dev)
        scale=128**-0.5; wk=torch.empty((8,n*32,128),device=dev)
        def recon(): reconstruct_k_triton(c,coeff,owner,wk)
        for warps in (2,4,8):
            fn=lambda warps=warps: compact_page_attention_v2(q,c,coeff,owner,hv,ak,av,scale,num_warps=warps)
            mean,med=timed(fn)
            rows.append({'history_tokens':n*32,'num_warps':warps,'kernel':'page_parallel_A+B','mean_us':mean,'median_us':med,'programs':(n+1)*8*4,'launches':2})
        mean,med=timed(recon)
        rows.append({'history_tokens':n*32,'num_warps':'na','kernel':'reconstruct_only','mean_us':mean,'median_us':med,'programs':n*8,'launches':1})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows: print(r,flush=True)
if __name__=='__main__': main()
