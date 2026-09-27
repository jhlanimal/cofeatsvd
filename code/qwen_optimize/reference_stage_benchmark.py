#!/usr/bin/env python3
from __future__ import annotations
import csv, re, statistics
from pathlib import Path
import torch
from benchmark_compression import reference
from eval_weighted_vs_puzzlekv import weighted_pair_cost

OUT=Path(__file__).parent/'results'; OUT.mkdir(exist_ok=True)
STAGES=['q_energy','k_energy_importance','top_m','gather_high_low','pair_dot','pair_cost','argmin_owner','group_mask','covariance','eigh','principal_direction','coefficient']

def ref_stages(k,q,m):
    t={s:[] for s in STAGES}; ev=[]
    def mark(name):
        e=torch.cuda.Event(True); e.record(); ev.append((name,e))
    mark('q_energy'); e=q.float().square().mean((0,1)); mark('k_energy_importance'); imp=k.float().square().sum(0)*e
    mark('top_m'); order=torch.argsort(imp,descending=True); high,low=order[:m],order[m:]; mark('gather_high_low')
    kh,kl=k.float()[:,high],k.float()[:,low]; wh,wl=e[high],e[low]; mark('pair_dot')
    dot=kh.T@kl; mark('pair_cost'); nh=kh.square().sum(0); nl=kl.square().sum(0); a=wh*nh; d=wl*nl; b2=wh[:,None]*wl[None,:]*dot.square(); cost=.5*(a[:,None]+d[None,:]-torch.sqrt((a[:,None]-d[None,:]).square()+4*b2+1e-8)); mark('argmin_owner'); owner_low=torch.argmin(cost,0) if low.numel() else torch.empty(0,dtype=torch.long,device=k.device)
    owner=torch.empty(k.shape[1],dtype=torch.long,device=k.device); owner[high]=torch.arange(m,device=k.device); owner[low]=owner_low; mark('group_mask'); mask=torch.nn.functional.one_hot(owner,num_classes=m).T.to(k.dtype); mark('covariance'); x=k.float()*torch.sqrt(e.clamp_min(1e-8))[None,:]; cov=torch.einsum('pd,gd,qd->gpq',x,mask,x); mark('eigh'); vals,vec=torch.linalg.eigh(cov); mark('principal_direction'); crows=vec[:,:,-1]; mark('coefficient'); ca=crows@k.float(); coeff=ca[owner,torch.arange(k.shape[1],device=k.device)]; mark('end');
    return crows,coeff,owner,ev

def stats(xs):
    xs=sorted(xs); return statistics.mean(xs),statistics.median(xs),xs[int(.95*(len(xs)-1))]

def main():
    dev='cuda:0'; torch.manual_seed(0); rows=[]
    for m in (64,80,96):
        k=torch.randn(8,32,128,device=dev); q=torch.randn(8,4,32,128,device=dev)
        run=lambda: [ref_stages(k[i],q[i],m) for i in range(8)]
        for _ in range(100): run()
        allvals={s:[] for s in STAGES}; total=[]
        for _ in range(1000):
            start=torch.cuda.Event(True); end=torch.cuda.Event(True); start.record(); result=run(); end.record(); end.synchronize(); total.append(start.elapsed_time(end))
            v={s:sum(result[i][3][j][1].elapsed_time(result[i][3][j+1][1]) for i in range(8) for j in range(len(result[i][3])-1) if result[i][3][j][0]==s) for s in STAGES}
            for s in STAGES: allvals[s].append(v[s])
        tm=statistics.mean(total)
        for s in STAGES:
            mean,med,p95=stats(allvals[s]); rows.append({'m':m,'stage':s,'mean_ms':mean,'median_ms':med,'p95_ms':p95,'percent_total':100*mean/tm})
        print('m',m,'clean_total_ms',tm)
    with (OUT/'reference_stage_latency.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    clean=[]
    for m in (64,80,96):
        k=torch.randn(8,32,128,device=dev); q=torch.randn(8,4,32,128,device=dev); fn=lambda: [reference(k[i],q[i],m) for i in range(8)]
        for _ in range(100): fn()
        xs=[]
        for _ in range(1000):
            a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); xs.append(a.elapsed_time(b))
        mean,med,p95=stats(xs); clean.append({'m':m,'mean_ms':mean,'median_ms':med,'p95_ms':p95})
    with (OUT/'reference_clean_total_latency.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=clean[0].keys()); w.writeheader(); w.writerows(clean)
if __name__=='__main__': main()
