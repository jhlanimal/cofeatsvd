#!/usr/bin/env python3
"""Reference/optimized exact compression benchmark and correctness checks."""
from __future__ import annotations
import argparse, csv, json, statistics, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_weighted_vs_puzzlekv import weighted_pair_cost
from optimized_grouped_compression import compress_optimized_exact, assign_groups_vectorized
from eval_weighted_vs_puzzlekv import puzzle_scores

def reference(k, q, m):
    k=k.float(); q=q.float(); p,d=k.shape
    e=q.square().mean(dim=(0,1)) if q.ndim==3 else q.square().mean(0)
    imp=k.square().sum(0)*e; order=torch.argsort(imp,descending=True); hi,lo=order[:m],order[m:]
    owner_lo=torch.argmin(weighted_pair_cost(k[:,hi],k[:,lo],e[hi],e[lo]),0) if lo.numel() else torch.empty(0,dtype=torch.long,device=k.device)
    owner=torch.empty(d,dtype=torch.long,device=k.device); owner[hi]=torch.arange(m,device=k.device); owner[lo]=owner_lo
    mask=torch.nn.functional.one_hot(owner,num_classes=m).T.to(k.dtype)
    x=k*torch.sqrt(e.clamp_min(1e-8))[None,:]
    cov=torch.einsum('pd,gd,qd->gpq',x,mask,x); _,v=torch.linalg.eigh(cov); crows=v[:,:,-1]
    ca=crows@k; coeff=ca[owner,torch.arange(d,device=k.device)]; c=crows.T
    return c,coeff,owner

def puzzle(k, rank):
    k = k.float()
    gram = k @ k.T
    _, vectors = torch.linalg.eigh(gram)
    left = vectors[:, -rank:].contiguous()
    right = left.T @ k
    return left, right

def event_ms(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e))
    vals.sort(); return dict(mean=statistics.mean(vals),median=statistics.median(vals),p50=vals[len(vals)//2],p95=vals[int(.95*(len(vals)-1))],min=min(vals))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--warmup',type=int,default=100); ap.add_argument('--iters',type=int,default=1000); ap.add_argument('--seed',type=int,default=0); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.manual_seed(args.seed); dev=torch.device(args.device); p,d,h=32,128,8
    rows=[]; dist=[]
    for m in (64,80,96):
      k=torch.randn(h,p,d,device=dev); q=torch.randn(h,4,p,d,device=dev)
      for name,fn in [('reference',lambda: [reference(k[i],q[i],m) for i in range(h)]),('optimized_exact',lambda: [compress_optimized_exact(k[i],q[i],m,True) for i in range(h)])]:
        torch.cuda.reset_peak_memory_stats(dev); stats=event_ms(fn,args.warmup,args.iters); stats.update(method=name,m=m,scope='page_all_kv_heads',peak_memory=torch.cuda.max_memory_allocated(dev)); rows.append(stats)
      rank = {64:14, 80:17, 96:20}[m]
      fn = lambda rank=rank: [puzzle(k[i], rank) for i in range(h)]
      torch.cuda.reset_peak_memory_stats(dev); stats=event_ms(fn,args.warmup,args.iters); stats.update(method='puzzle',m=m,rank=rank,scope='page_all_kv_heads',peak_memory=torch.cuda.max_memory_allocated(dev)); rows.append(stats)
      for i in range(h):
        _,_,o=assign_groups_vectorized(k[i].float(),q[i].float().square().mean((0,1)),m); counts=torch.bincount(o,minlength=m); dist.extend(counts.tolist())
      c1,co1,o1=reference(k[0],q[0],m); c2,co2,o2=compress_optimized_exact(k[0],q[0],m,True); r1=c1[:,o1]*co1.unsqueeze(0); r2=c2[:,o2]*co2.unsqueeze(0); query=torch.randn(16,d,device=dev); s1=query@r1.T; s2=query@r2.T
      print(json.dumps({'m':m,'reconstruction_rel':float(torch.linalg.norm(r1-r2)/torch.linalg.norm(r1)),'score_max_abs':float((s1-s2).abs().max()),'score_rel':float(torch.linalg.norm(s1-s2)/torch.linalg.norm(s1))}))
    out=Path(__file__).parent/'results'; out.mkdir(exist_ok=True)
    with (out/'reference_latency.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out/'group_size_distribution.csv').open('w',newline='') as f:
      w=csv.writer(f); w.writerow(['group_size','count']);
      for n in range(1,6): w.writerow([f'{n}' if n<5 else '>4',sum(x==n if n<5 else x>4 for x in dist)])
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
