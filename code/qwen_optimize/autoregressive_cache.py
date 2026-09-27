"""Token-streaming KV cache for exact Reference grouped compression."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page
@dataclass
class LayerCache:
    page_size:int=32; method:str='full'; m:int|None=None
    sealed_k:list=field(default_factory=list); sealed_v:list=field(default_factory=list)
    active_k:list=field(default_factory=list); active_v:list=field(default_factory=list); active_q:list=field(default_factory=list)
    sealing_event_count:int=0; compressed_history_calls:int=0; full_history_fallback:bool=False
    def append_and_attend(self,q,k,v,scale):
        self.active_k.append(k.reshape(-1)); self.active_v.append(v.reshape(-1)); self.active_q.append(q.reshape(-1))
        ak=torch.stack(self.active_k); av=torch.stack(self.active_v); ks=[]; vs=[]
        for pk,pv in zip(self.sealed_k,self.sealed_v):
            hk=pk if self.method=='full' else pk[0]@pk[1].T
            if self.method!='full': self.compressed_history_calls+=1
            ks.append(hk); vs.append(pv)
        ks.append(ak); vs.append(av); scores=q@(torch.cat(ks) .T)*scale
        out=(torch.softmax(scores.float(),-1)@torch.cat(vs).float()).to(q.dtype)
        if len(self.active_k)==self.page_size:
            pk=torch.stack(self.active_k); pv=torch.stack(self.active_v); pq=torch.stack(self.active_q)
            self.sealed_k.append(pk if self.method=='full' else weighted_coalescing_page(pk, pq, int(self.m))); self.sealed_v.append(pv)
            self.active_k.clear(); self.active_v.clear(); self.active_q.clear(); self.sealing_event_count+=1
        return out
    @property
    def active_len(self): return len(self.active_k)
    @property
    def sealed_pages(self): return len(self.sealed_k)
