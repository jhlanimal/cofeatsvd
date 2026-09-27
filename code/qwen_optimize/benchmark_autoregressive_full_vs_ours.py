#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gc,json,math,statistics,time,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page, weighted_pair_cost
from analyze_post_rope_k_pages import DEFAULT_DATA,load_tokens
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
from triton_compact_decode_attention import compact_attention, reconstruct_k_triton
from triton_page_parallel_attention_v2 import compact_page_attention_v2
from fused_decode_ops import install_rms, rope as fused_rope


def weighted_coalescing_pages_batched(k_pages, q_pages, m):
 """Batch the unchanged weighted coalescing builder over page dimension."""
 k = k_pages.float()
 q = q_pages.float()
 b, p, d = k.shape
 q_energy = q.square().mean(dim=(1, 2))
 importance = k.square().sum(dim=1) * q_energy
 order = torch.argsort(importance, dim=1, descending=True)
 high, low = order[:, :m], order[:, m:]
 kh = torch.gather(k, 2, high[:, None, :].expand(-1, p, -1))
 kl = torch.gather(k, 2, low[:, None, :].expand(-1, p, -1))
 wh = torch.gather(q_energy, 1, high)
 wl = torch.gather(q_energy, 1, low)
 nh = kh.square().sum(dim=1)
 nl = kl.square().sum(dim=1)
 dot = torch.einsum('bpm,bpl->bml', kh, kl)
 aa = wh * nh
 dd = wl * nl
 b2 = wh[:, :, None] * wl[:, None, :] * dot.square()
 cost = 0.5 * (aa[:, :, None] + dd[:, None, :] -
               torch.sqrt((aa[:, :, None] - dd[:, None, :]).square() + 4 * b2 + 1e-8))
 owners_low = torch.argmin(cost, dim=1)
 owners = torch.empty((b, d), device=k.device, dtype=torch.long)
 owners.scatter_(1, high, torch.arange(m, device=k.device).expand(b, -1))
 owners.scatter_(1, low, owners_low)
 group_mask = torch.nn.functional.one_hot(owners, num_classes=m).transpose(1, 2).to(k.dtype)
 weighted_k = k * torch.sqrt(q_energy.clamp_min(1e-8))[:, None, :]
 covariance = torch.einsum('bpd,bmd,bqd->bmpq', weighted_k, group_mask, weighted_k)
 _, vectors = torch.linalg.eigh(covariance)
 c_rows = vectors[:, :, :, -1].contiguous()
 coeff_all = torch.einsum('bmp,bpd->bmd', c_rows, k)
 coeff = torch.gather(coeff_all, 1, owners[:, None, :]).squeeze(1)
 return c_rows.transpose(1, 2).contiguous(), coeff.contiguous(), owners


class Runner:
 def __init__(self,model,method,m=64,page=32,max_tokens=2176,logits_to_keep=None):
  self.model=model; self.method=method; self.m=m; self.page=page; self.max_tokens=max_tokens; self.logits_to_keep=logits_to_keep; self.states=[]; self.workspaces=[]; self.compact_stores=[]; self.active_buffers=[]; self.active_lens=[]; self.dense_lengths=[]; self.pending_seals=[]; self.reconstructed_pages=[]; self.phase='decode'; self.hook_install_count=0; self.reconstruct_native_calls=0; self.direct_attention_calls=0; self.triton_attention_calls=0; self.kv_reconstruct_calls=0; self.kv_reconstruct_pages=0; self.layer_page_seals=0; self.kv_head_page_compressions=0; self.compressed_attention_layer_calls=0; self.compressed_history_page_score_evals=0; self.compressed_kv_head_page_score_evals=0; self.comp_ms=0; self.prefill_forward_ms=0.0; self.prompt_compression_ms=0.0
  root=model.model; self.root=root; device=next(model.parameters()).device; self.position_cache=torch.arange(self.max_tokens,device=device,dtype=torch.long)
  for layer in root.layers:
   a=layer.self_attn; nh=int(getattr(a,'num_heads',model.config.num_attention_heads)); nk=int(getattr(a,'num_key_value_heads',model.config.num_key_value_heads)); gqa=nh//nk; hd=int(a.head_dim); recon=method in ('ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref','full_wrapper'); direct=method in ('ours_direct_v2','ours_direct_ref'); work_dtype=next(model.parameters()).dtype; self.workspaces.append((None,torch.empty((nk,self.max_tokens,hd),device=device,dtype=work_dtype)) if direct else ((torch.empty((nk,self.max_tokens,hd),device=device,dtype=work_dtype),torch.empty((nk,self.max_tokens,hd),device=device,dtype=work_dtype)) if recon else (None,None))); self.states.append([{'sk':[],'sv':[],'ak':[],'av':[],'aq':[]} for _ in range(nk)]); self.compact_stores.append({'c':torch.empty((self.max_tokens//page,nk,page,m),device=device,dtype=work_dtype),'coeff':torch.empty((self.max_tokens//page,nk,hd),device=device,dtype=work_dtype),'owner':torch.empty((self.max_tokens//page,nk,hd),device=device,dtype=torch.uint8),'pages':0} if recon else None); self.active_buffers.append((torch.empty((nk,page,hd),device=device,dtype=work_dtype),torch.empty((nk,page,gqa,hd),device=device,dtype=work_dtype)) if direct else ((None,torch.empty((nk,page,gqa,hd),device=device,dtype=work_dtype)) if recon else (None,None))); self.active_lens.append(0); self.dense_lengths.append(0); self.reconstructed_pages.append(0)
   self.pending_seals.append(False); self._patch(layer,len(self.states)-1)
 def _patch(self,layer,li):
  attn=layer.self_attn; runner=self; original_forward=attn.forward; nh=int(getattr(attn,'num_heads',self.model.config.num_attention_heads)); nk=int(getattr(attn,'num_key_value_heads',self.model.config.num_key_value_heads)); hd=int(attn.head_dim); gqa=nh//nk; scale=float(getattr(attn,'scaling',hd**-0.5))
  # Capture tensors produced by the native prefill path.  The previous
  # implementation recomputed all Q/K/V projections inside this wrapper,
  # doubling the most expensive part of prefill.  Hooks are inactive during
  # decode and therefore do not add work to the steady-state path.
  def capture_q_norm(_module, _inputs, output):
   if runner.phase == 'prefill':
    attn._runner_q_norm = output
  def capture_k_norm(_module, _inputs, output):
   if runner.phase == 'prefill':
    attn._runner_k_norm = output
  def capture_v_proj(_module, _inputs, output):
   if runner.phase == 'prefill':
    attn._runner_v_proj = output
  capture_handles=[attn.q_norm.register_forward_hook(capture_q_norm),
                   attn.k_norm.register_forward_hook(capture_k_norm),
                   attn.v_proj.register_forward_hook(capture_v_proj)]
  if runner.method == 'ours_direct_v2':
   attn.register_buffer('_decode_qkv_weight',torch.cat((attn.q_proj.weight,attn.k_proj.weight,attn.v_proj.weight),dim=0),persistent=False)
   assert attn.q_proj.bias is None and attn.k_proj.bias is None and attn.v_proj.bias is None
  def forward(self,hidden_states,position_embeddings=None,attention_mask=None,past_key_values=None,cache_position=None,**kwargs):
   b,seq,_=hidden_states.shape
   if b!=1: raise RuntimeError('runner requires batch=1')
   # Prefill must remain a single native model forward.  We duplicate only
   # Q/K/V projection work to capture post-RoPE tensors; the original Qwen
   # attention implementation still produces the prefill output.
   if seq > 1:
    if runner.phase != 'prefill':
     raise RuntimeError('multi-token forward is only valid during prefill')
    # original_forward has already computed these tensors.  Reuse its
    # projection/norm results and only perform the same post-RoPE transform
    # needed by the cache collector.
    original_output=original_forward(hidden_states, position_embeddings=position_embeddings,
                                     attention_mask=attention_mask, past_key_values=past_key_values,
                                     cache_position=cache_position, **kwargs)
    q=attn._runner_q_norm.transpose(1,2)
    k=attn._runner_k_norm.transpose(1,2)
    v=attn._runner_v_proj.view(1,seq,nk,hd).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    states=runner.states[li]
    for kh in range(nk):
     states[kh]['prefill_k']=k[0,kh].float().detach()
     states[kh]['prefill_v']=v[0,kh].detach()
     states[kh]['prefill_q']=q[0,kh*gqa:(kh+1)*gqa].float().detach()
    # Keep the capture hooks installed: RULER reuses one patched runner across
    # independent samples.  They are inactive outside prefill and reset each
    # sample's captured tensors on the next prompt.
    return original_output
   if seq!=1: raise RuntimeError('runner requires one token during decode')
   if runner.method == 'ours_direct_v2':
    projected=torch.nn.functional.linear(hidden_states,self._decode_qkv_weight)
    qp,kp,vp=projected.split((nh*hd,nk*hd,nk*hd),dim=-1)
    q=qp.view(1,nh,1,hd); k=kp.view(1,nk,1,hd); v=vp.view(1,nk,1,hd)
   else:
    q=self.q_proj(hidden_states).view(1,1,nh,hd).transpose(1,2); k=self.k_proj(hidden_states).view(1,1,nk,hd).transpose(1,2); v=self.v_proj(hidden_states).view(1,1,nk,hd).transpose(1,2)
   q=self.q_norm(q); k=self.k_norm(k)
   if runner.method == 'ours_direct_v2':
    q,k=fused_rope(q,k,*position_embeddings)
   else:
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
   out=torch.empty((1,nh,1,hd),device=hidden_states.device,dtype=hidden_states.dtype); states=runner.states[li]
   if runner.method in ('ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref','full_wrapper'):
    # The head dimension is already contiguous: four batched writes replace
    # 32 per-head writes without changing any cached values or their dtype.
    store=runner.compact_stores[li]; active_pos=runner.active_lens[li]
    hist_len=runner.dense_lengths[li] if runner.method=='full_wrapper' else store['pages']*runner.page
    k_workspace,v_workspace=runner.workspaces[li]
    if runner.method in ('ours_direct_v2','ours_direct_ref'):
     abk,_=runner.active_buffers[li]; abk[:,active_pos].copy_(k[0,:,0])
    else:
     k_workspace[:,hist_len+active_pos].copy_(k[0,:,0])
    v_workspace[:,hist_len+active_pos].copy_(v[0,:,0])
    _,abq_comp=runner.active_buffers[li]
    abq_comp[:,active_pos].copy_(q[0,:,0].reshape(nk,gqa,hd))
    runner.active_lens[li] += 1
   else:
    for kh in range(nk):
     st=states[kh]
     st['ak'].append(k[0,kh,0].float());st['av'].append(v[0,kh,0]);st['aq'].append(q[0,kh*gqa:(kh+1)*gqa,0])
   if runner.method not in ('full','ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref','full_wrapper') and states[0]['sk']:
    runner.compressed_attention_layer_calls += 1
    runner.compressed_history_page_score_evals += len(states[0]['sk'])
    runner.compressed_kv_head_page_score_evals += len(states[0]['sk'])*nk
   if runner.method == 'ours_triton_compact':
    if states[0]['sk']:
     cs=torch.stack([torch.stack([x[0] for x in st['sk']],dim=0) for st in states],dim=1).contiguous()
     co=torch.stack([torch.stack([x[1] for x in st['sk']],dim=0) for st in states],dim=1).contiguous()
     ow=torch.stack([torch.stack([x[2] for x in st['sk']],dim=0) for st in states],dim=1).contiguous()
     hv=torch.stack([torch.stack(st['sv'],dim=0) for st in states],dim=1).contiguous()
    else:
     cs=torch.empty((0,nk,runner.page,runner.m),device=q.device,dtype=torch.float32)
     co=torch.empty((0,nk,hd),device=q.device,dtype=torch.float32)
     ow=torch.empty((0,nk,hd),device=q.device,dtype=torch.uint8)
     hv=torch.empty((0,nk,runner.page,hd),device=q.device,dtype=hidden_states.dtype)
    ak=torch.stack([torch.stack(st['ak']) for st in states],dim=0).contiguous()
    av=torch.stack([torch.stack(st['av']) for st in states],dim=0).contiguous()
    fused=compact_attention(q[0,:,0].float().contiguous(),cs,co,ow,hv,ak,av,scale)
    result=self.o_proj(fused.reshape(1,1,nh*hd).to(hidden_states.dtype))
    runner.triton_attention_calls += 1
    for kh in range(nk):
     st=states[kh]
     if len(st['ak'])==runner.page:
      pk=torch.stack(st['ak']); pv=torch.stack(st['av']); pq=torch.stack(st['aq'])
      t=time.perf_counter(); rep=weighted_coalescing_page(pk,pq,runner.m,return_compact=True); runner.comp_ms+=(time.perf_counter()-t)*1000
      st['sk'].append(rep); st['sv'].append(pv); st['ak'].clear(); st['av'].clear(); st['aq'].clear(); runner.kv_head_page_compressions += 1
      if kh==nk-1: runner.layer_page_seals += 1
    return result,None
   if runner.method == 'full_wrapper':
    k_workspace,v_workspace=runner.workspaces[li]
    active_len=runner.active_lens[li]
    hist_len=runner.dense_lengths[li]
    current_len=hist_len+active_len
    key_states=repeat_kv(k_workspace[:,:current_len].unsqueeze(0),gqa)
    value_states=repeat_kv(v_workspace[:,:current_len].unsqueeze(0),gqa)
    query_states=q[0].to(k_workspace.dtype).unsqueeze(0)
    attn_weights=torch.matmul(query_states,key_states.transpose(2,3))*scale
    attn_weights=torch.softmax(attn_weights.float(),dim=-1).to(query_states.dtype)
    native_out=torch.matmul(attn_weights,value_states)
    result=self.o_proj(native_out.transpose(1,2).reshape(1,1,nh*hd).to(hidden_states.dtype))
    runner.reconstruct_native_calls += 1
    return result,None
   if runner.method in ('ours_direct_v2','ours_direct_ref'):
    store=runner.compact_stores[li]; page_count=store['pages']; active_len=runner.active_lens[li]
    hist_len=page_count*runner.page; current_len=hist_len+active_len
    k_workspace,v_workspace=runner.workspaces[li]
    active_k=(runner.active_buffers[li][0][:,:active_len] if runner.method in ('ours_direct_v2','ours_direct_ref') else k_workspace[:,hist_len:hist_len+active_len])
    if runner.method == 'ours_direct_ref':
     # Sparse q @ A, independently for every page and KV head.
     qh=q[0,:,0].float().reshape(nk,gqa,hd)
     owner=store['owner'][:page_count].long()
     projected=torch.zeros((page_count,nk,gqa,runner.m),device=q.device)
     projected.scatter_add_(3,owner[:,:,None,:].expand(-1,-1,gqa,-1),
                            qh[None]*store['coeff'][:page_count,:,None,:])
     hs=torch.einsum('nhgm,nhpm->hgnp',projected,store['c'][:page_count].float()).reshape(nk,gqa,hist_len)
     active_scores=torch.einsum('hgd,htd->hgt',qh,active_k.float())
     prob=torch.cat((hs,active_scores),dim=-1).mul(scale).softmax(dim=-1)
     compact_out=torch.einsum('hgt,htd->hgd',prob,v_workspace[:,:current_len].float()).reshape(nh,hd)
    else:
     compact_out=compact_page_attention_v2(
         q[0,:,0].contiguous(),store['c'][:page_count],store['coeff'][:page_count],
         store['owner'][:page_count],
         v_workspace[:,:hist_len].view(nk,page_count,runner.page,hd).permute(1,0,2,3),
         active_k,v_workspace[:,hist_len:current_len],scale,num_warps=2)
    result=self.o_proj(compact_out.reshape(1,1,nh*hd).to(hidden_states.dtype))
    runner.direct_attention_calls += 1
    if active_len == runner.page:
     runner.pending_seals[li]=True
    return result,None
   if runner.method in ('ours_reconstruct_native','ours_reconstruct_native_bf16'):
    # Diagnostic path only.  Reconstruct effective dense K_hat from the exact
    # compressed representation, then use one native SDPA call.  This does
    # not claim compressed-memory savings; it isolates the custom attention
    # execution cost.
    store=runner.compact_stores[li]; page_count=store['pages']; active_len=runner.active_lens[li]; hist_len=page_count*runner.page; current_len=hist_len+active_len
    k_workspace,v_workspace=runner.workspaces[li]
    # Sealed pages are immutable. Reconstruct only pages sealed since the
    # previous decode step; historical K stays in the reusable workspace.
    if page_count > runner.reconstructed_pages[li]:
     cs=store['c'][:page_count]
     coeff=store['coeff'][:page_count]
     owner=store['owner'][:page_count]
     new_start=runner.reconstructed_pages[li]
     reconstruct_k_triton(cs[new_start:page_count], coeff[new_start:page_count], owner[new_start:page_count], k_workspace[:,new_start*runner.page:hist_len])
     runner.reconstructed_pages[li]=page_count; runner.kv_reconstruct_calls += 1; runner.kv_reconstruct_pages += page_count-new_start
    key_k=k_workspace[:,:current_len].unsqueeze(0)
    key_v=v_workspace[:,:current_len].unsqueeze(0)
    q_native=q[0].to(k_workspace.dtype).unsqueeze(0)
    # PyTorch 2.3 has no enable_gqa keyword. Keep the fallback out of the
    # hot path: exception handling here would happen once per layer/token.
    native_out=torch.nn.functional.scaled_dot_product_attention(
        q_native, repeat_kv(key_k, gqa), repeat_kv(key_v, gqa),
        dropout_p=0.0, is_causal=False, scale=scale)
    result=self.o_proj(native_out.transpose(1,2).reshape(1,1,nh*hd).to(hidden_states.dtype))
    runner.reconstruct_native_calls += 1
    if runner.active_lens[li] == runner.page:
     # The current token has already consumed this page exactly.  Defer all
     # compression until the complete transformer token has finished.
     runner.pending_seals[li] = True
    return result,None
   # Batch all sealed pages for one KV head.  The previous path rebuilt C@A.T
   # and concatenated Python lists independently for every Q head and token.
   for kh in range(nk):
    st=states[kh]
    if st['sk']:
     qg=q[0,kh*gqa:(kh+1)*gqa,0].float()
     if runner.method=='full':
      hk=torch.cat(st['sk'],dim=0)
      hv=torch.cat(st['sv'],dim=0)
     elif runner.method in ('ours_compact','ours_triton_compact'):
      cs=torch.stack([x[0] for x in st['sk']],dim=0)
      co=torch.stack([x[1] for x in st['sk']],dim=0)
      ow=torch.stack([x[2].long() for x in st['sk']],dim=0)
      gathered=torch.gather(cs,2,ow[:,None,:].expand(-1,cs.shape[1],-1))
      hs=torch.einsum('gd,npd,nd->ngp',qg,gathered,co)
      st['_hist_scores']=hs.permute(1,0,2).reshape(gqa,-1)
      hv=torch.cat(st['sv'],dim=0)
     elif runner.method=='ours_denseA_batched':
      # Exact dense-A path, batched over all sealed pages and GQA heads.
      # This removes the historical page loop and Q-head loop without changing
      # the Reference representation or its mathematical result.
      cs=torch.stack([x[0] for x in st['sk']],dim=0)
      aa=torch.stack([x[1] for x in st['sk']],dim=0)
      projected=torch.einsum('gd,ndm->ngm',qg,aa)
      hs=torch.einsum('ngm,npm->ngp',projected,cs)
      st['_hist_scores']=hs.permute(1,0,2).reshape(gqa,-1)
      hv=torch.cat(st['sv'],dim=0)
     else:
      cs=torch.stack([x[0] for x in st['sk']],dim=0)
      aa=torch.stack([x[1] for x in st['sk']],dim=0)
      projected=torch.einsum('gd,ndm->ngm',qg,aa)
      hs=torch.einsum('ngm,npm->ngp',projected,cs)
      # Keep the historical scores in page/token order, one tensor per GQA group.
      st['_hist_scores']=hs.permute(1,0,2).reshape(gqa,-1)
      hv=torch.cat(st['sv'],dim=0)
     st['_hist_v']=hv
   for qh in range(nh):
    kh=qh//gqa; st=states[kh]; ks=[];vs=[]
    if st['sk']:
     score=st['_hist_scores'][qh-kh*gqa]
     vals=st['_hist_v']
    else:
     score=torch.empty(0,device=q.device); vals=torch.empty((0,hd),device=q.device)
    active_k=torch.stack(st['ak']); active_v=torch.stack(st['av'])
    score=torch.cat((score,(q[0,qh,0].float()@active_k.T))) * scale
    prob=torch.softmax(score.float(),-1); out[0,qh,0]=(prob@torch.cat((vals.float(),active_v.float()))).to(out.dtype)
   result=self.o_proj(out.transpose(1,2).reshape(1,1,nh*hd))
   for kh in range(nk):
    st=states[kh]
    if len(st['ak'])==runner.page:
     pk=torch.stack(st['ak']);pv=torch.stack(st['av']);pq=torch.stack(st['aq'])
     if runner.method=='full': rep=pk
     elif runner.method in ('ours_compact','ours_triton_compact'):
      t=time.perf_counter(); rep=weighted_coalescing_page(pk,pq,runner.m,return_compact=True); runner.comp_ms+=(time.perf_counter()-t)*1000
     elif runner.method=='ours_denseA_batched':
      t=time.perf_counter(); rep=weighted_coalescing_page(pk,pq,runner.m); runner.comp_ms+=(time.perf_counter()-t)*1000
     else:
      t=time.perf_counter(); rep=weighted_coalescing_page(pk,pq,runner.m); runner.comp_ms+=(time.perf_counter()-t)*1000
     st['sk'].append(rep);st['sv'].append(pv);st['ak'].clear();st['av'].clear();st['aq'].clear();runner.kv_head_page_compressions+=1
     if kh==nk-1: runner.layer_page_seals+=1
   return result,None
  import types; attn.forward=types.MethodType(forward,attn)
  runner.hook_install_count += 1
 def reset_runtime(self):
  """Reuse the patched model for an independent prompt/sample."""
  self.states = [[{'sk': [], 'sv': [], 'ak': [], 'av': [], 'aq': []}
                  for _ in range(len(layer_states))]
                 for layer_states in self.states]
  for store in self.compact_stores:
   if store is not None:
    store['pages'] = 0
  self.active_lens = [0 for _ in self.active_lens]
  self.dense_lengths = [0 for _ in self.dense_lengths]
  self.pending_seals = [False for _ in self.pending_seals]
  self.reconstructed_pages = [0 for _ in self.reconstructed_pages]
  self.phase = 'decode'
 def seal_prefill(self):
  """Seal all complete prompt pages after native prefill has finished."""
  if self.phase != 'prefill':
   raise RuntimeError('seal_prefill called outside prefill')
  for li, states in enumerate(self.states):
   store = self.compact_stores[li]
   page_count = None
   for kh, st in enumerate(states):
    pk = st.pop('prefill_k')
    pv = st.pop('prefill_v')
    pq = st.pop('prefill_q')
    page_count = pk.shape[0] // self.page
    if pk.shape[0] % self.page:
     raise RuntimeError(f'prefill length {pk.shape[0]} is not page aligned')
    for start in range(0, pk.shape[0], self.page):
     kp=pk[start:start+self.page]
     vp=pv[start:start+self.page]
     qp=pq[:,start:start+self.page]
     if self.method == 'full_wrapper':
      k_workspace,v_workspace=self.workspaces[li]
      k_workspace[kh,start:start+kp.shape[0]].copy_(kp)
      v_workspace[kh,start:start+vp.shape[0]].copy_(vp)
      continue
     if self.method == 'full': rep=kp
     elif self.method in ('ours_compact','ours_triton_compact'):
      t=time.perf_counter(); rep=weighted_coalescing_page(kp,qp,self.m,return_compact=True); self.comp_ms += (time.perf_counter()-t)*1000
     elif self.method in ('ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref'):
      t=time.perf_counter(); rep=weighted_coalescing_page(kp,qp,self.m,return_compact=True); self.comp_ms += (time.perf_counter()-t)*1000
      page_index = start // self.page
      _,v_workspace=self.workspaces[li]
      v_workspace[kh,start:start+vp.shape[0]].copy_(vp)
      store['c'][page_index,kh].copy_(rep[0])
      store['coeff'][page_index,kh].copy_(rep[1])
      store['owner'][page_index,kh].copy_(rep[2])
      self.kv_head_page_compressions += 1
      continue
     else:
      t=time.perf_counter(); rep=weighted_coalescing_page(kp,qp,self.m); self.comp_ms += (time.perf_counter()-t)*1000
     st['sk'].append(rep); st['sv'].append(vp)
     self.kv_head_page_compressions += 1
   if self.method in ('ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref'):
    store['pages'] = pk.shape[0] // self.page
   if self.method == 'full_wrapper':
    self.dense_lengths[li] = int((page_count or 0) * self.page)
   self.layer_page_seals += int(page_count or 0)
  self.phase='decode'
 def prefill(self, ids, positions):
  self.phase='prefill'
  kwargs = {'logits_to_keep': self.logits_to_keep} if self.logits_to_keep is not None else {}
  return self.model(input_ids=ids.view(1,-1), use_cache=False, cache_position=positions, **kwargs)
 def seal_pending(self):
  """Seal pages only after the complete decode token has finished."""
  if self.method not in ('ours_reconstruct_native','ours_reconstruct_native_bf16','ours_direct_v2','ours_direct_ref'):
   return
  for li, pending in enumerate(self.pending_seals):
   if not pending:
    continue
   states=self.states[li]; store=self.compact_stores[li]
   k_workspace,v_workspace=self.workspaces[li]
   hist_len=store['pages']*self.page
   _,pq_buf=self.active_buffers[li]
   active_k,_=self.active_buffers[li]
   rep=weighted_coalescing_pages_batched(active_k[:,:self.page],pq_buf,self.m)
   page_index=store['pages']
   store['c'][page_index].copy_(rep[0])
   store['coeff'][page_index].copy_(rep[1])
   store['owner'][page_index].copy_(rep[2])
   self.kv_head_page_compressions += len(states)
   store['pages'] += 1
   self.layer_page_seals += 1
   self.active_lens[li]=0
   self.pending_seals[li]=False
 def run_token(self,token,pos):
  x=token.reshape(1,1).to(next(self.model.parameters()).device); cp=self.position_cache[pos:pos+1]
  kwargs = {'logits_to_keep': self.logits_to_keep} if self.logits_to_keep is not None else {}
  logits=self.model(input_ids=x,use_cache=False,cache_position=cp,**kwargs).logits
  self.seal_pending()
  return logits

def load_runner(path,method,m,device,max_tokens=2176,attn_implementation='eager',logits_to_keep=None):
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,attn_implementation=attn_implementation).eval().to(device)
 if method == 'ours_direct_v2':
  install_rms(model)
 return Runner(model,method,m,max_tokens=max_tokens,logits_to_keep=logits_to_keep)

@torch.inference_mode()
def run_native_full(model, ids, init, decode, quality=True, greedy=False):
 device=next(model.parameters()).device; ids=ids.to(device); pos_cache=torch.arange(ids.numel(),device=device,dtype=torch.long)
 torch.cuda.synchronize(); prefill_start=torch.cuda.Event(True); prefill_end=torch.cuda.Event(True); decode_start=torch.cuda.Event(True); decode_end=torch.cuda.Event(True)
 prefill_start.record(); out=model(input_ids=ids[:init].view(1,-1), use_cache=True); nll=None
 if quality:
  end=init if decode else init-1
  nll=torch.nn.functional.cross_entropy(out.logits[:,:end].float().reshape(-1,out.logits.shape[-1]),ids[1:end+1].reshape(-1),reduction='sum')
 token=out.logits[:,-1].argmax(-1) if greedy else None
 prefill_end.record(); prefill_end.synchronize(); prefill_ms=prefill_start.elapsed_time(prefill_end); past=out.past_key_values
 torch.cuda.synchronize(); decode_start.record()
 for pos in range(init,init+decode):
  out=model(input_ids=(token if greedy else ids[pos]).view(1,1), past_key_values=past, use_cache=True, cache_position=pos_cache[pos:pos+1]); past=out.past_key_values
  if greedy: token=out.logits[:,0].argmax(-1)
  if quality and pos+1 < ids.numel(): nll=nll+torch.nn.functional.cross_entropy(out.logits[:,0].float(),ids[pos+1].view(1),reduction='sum')
 decode_end.record(); decode_end.synchronize(); return prefill_ms,decode_start.elapsed_time(decode_end),None if nll is None else float(nll.item())

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model-path',default='/data/share/Qwen3-4B-Instruct-2507')
 ap.add_argument('--data-path',default=DEFAULT_DATA)
 ap.add_argument('--device',default='cuda:0')
 ap.add_argument('--init',type=int,default=128)
 ap.add_argument('--decode',type=int,default=2048)
 ap.add_argument('--mode',choices=('quality','benchmark'),default='quality')
 ap.add_argument('--warmup',type=int,default=2)
 ap.add_argument('--repeats',type=int,default=5)
 ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--skip-full',action='store_true')
 ap.add_argument('--only-ours',action='store_true',help='Filter the selected experiment to non-Full methods')
 ap.add_argument('--direct-m',type=int,default=80,help='m for --direct-test')
 ap.add_argument('--compact-test',action='store_true')
 ap.add_argument('--dense-test',action='store_true')
 ap.add_argument('--reconstruct-test',action='store_true')
 ap.add_argument('--clean-v2-test',action='store_true')
 ap.add_argument('--triton-test',action='store_true')
 ap.add_argument('--wrapper-test',action='store_true')
 ap.add_argument('--direct-test',action='store_true')
 ap.add_argument('--direct-ref-test',action='store_true')
 ap.add_argument('--full-fused-only',action='store_true')
 ap.add_argument('--greedy',action='store_true',help='Benchmark autoregressive token feedback instead of fixed evaluation inputs')
 ap.add_argument('--compile-ours',action='store_true')
 ap.add_argument('--output-dir',default='qwen/optimize/results/end_to_end')
 a=ap.parse_args()
 if a.greedy and a.mode != 'benchmark':
  ap.error('PPL requires teacher-forced inputs; --greedy is only for benchmark mode')
 if str(a.device).startswith('cuda'):
  # Triton 3.x requires the launch device to match the current CUDA device.
  # Keep multi-GPU runs deterministic when a non-default device is selected.
  torch.cuda.set_device(torch.device(a.device))
 tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,use_fast=True)
 total=a.init+a.decode; ids=load_tokens(tok,a.data_path,total).to(a.device)
 if a.full_fused_only: methods=[('full',None)]
 elif a.clean_v2_test: methods=[('full',None),('ours_reconstruct_native_bf16',80)]
 elif a.wrapper_test: methods=[('full',None),('full_wrapper',80)]
 elif a.direct_test: methods=[('full',None),('ours_direct_v2',a.direct_m)]
 elif a.direct_ref_test: methods=[('full',None),('ours_direct_ref',80)]
 elif a.triton_test: methods=[('full',None),('ours',80),('ours_triton_compact',80)]
 elif a.reconstruct_test: methods=[('full',None),('ours_reconstruct_native',80),('ours_reconstruct_native_bf16',80)]
 elif a.dense_test: methods=[('full',None),('ours',80),('ours_denseA_batched',80)]
 elif a.compact_test: methods=[('full',None),('ours',80),('ours_compact',80)]
 elif a.skip_full: methods=[('ours',64),('ours',80),('ours',96)]
 else: methods=[('full',None),('ours',64),('ours',80),('ours',96)]
 if a.only_ours: methods=[(method,m) for method,m in methods if method != 'full']
 position_cache=torch.arange(total,device=ids.device,dtype=torch.long)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 (out/'config.json').write_text(json.dumps({**vars(a),'token_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
     'ppl_prediction_tokens':total-1,'quality_mode':'teacher-forced token-by-token',
     'decode_input':'greedy autoregressive' if a.greedy else 'fixed evaluation tokens',
     'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('benchmark_autoregressive_full_vs_ours.py','triton_page_parallel_attention_v2.py','fused_decode_ops.py')},
     'torch':torch.__version__,'gpu':torch.cuda.get_device_name(a.device)},indent=2))
 quality=(a.mode=='quality')
 def run_one(method,m):
  if method=='full':
   model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,attn_implementation='eager').eval().to(a.device)
   if a.full_fused_only:
    install_rms(model)
    import transformers.models.qwen3.modeling_qwen3 as qwen_modeling
    def native_fused_rope(q,k,cos,sin,*args,**kwargs):
     if q.shape[0] == 1 and q.shape[2] == 1:
      return fused_rope(q,k,cos,sin)
     return apply_rotary_pos_emb(q,k,cos,sin,*args,**kwargs)
    qwen_modeling.apply_rotary_pos_emb=native_fused_rope
   torch.cuda.reset_peak_memory_stats()
   prefill_ms,decode_ms,nll=run_native_full(model,ids,a.init,a.decode,quality=quality,greedy=a.greedy)
   row={'prefill_total_ms':prefill_ms,'decode_total_ms':decode_ms,'nll':nll,'ppl':math.exp(nll/max(total-1,1)) if nll is not None else None,'prompt_compression_ms':0.0,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
   del model
   torch.cuda.empty_cache()
   gc.collect()
   return row
  r=load_runner(a.model_path,method,m,a.device)
  torch.cuda.reset_peak_memory_stats()
  if a.compile_ours and method=='ours_reconstruct_native_bf16':
   r.model=torch.compile(r.model,mode='reduce-overhead',fullgraph=False,dynamic=False)
  r.position_cache=position_cache
  positions=torch.arange(a.init,device=ids.device,dtype=torch.long)
  torch.cuda.synchronize(); prefill_start=torch.cuda.Event(True); prefill_end=torch.cuda.Event(True); decode_start=torch.cuda.Event(True); decode_end=torch.cuda.Event(True)
  prefill_start.record(); prefill_out=r.prefill(ids[:a.init],positions)
  nll=None
  if quality:
   end=a.init if a.decode else a.init-1
   nll=torch.nn.functional.cross_entropy(prefill_out.logits[:,:end].float().reshape(-1,prefill_out.logits.shape[-1]),ids[1:end+1].reshape(-1),reduction='sum')
  token=prefill_out.logits[:,-1].argmax(-1) if a.greedy else None
  r.seal_prefill(); prefill_end.record(); prefill_end.synchronize(); prefill_ms=prefill_start.elapsed_time(prefill_end)
  torch.cuda.synchronize(); decode_start.record()
  for i in range(a.init,total):
   logits=r.run_token(token if a.greedy else ids[i],i)
   if a.greedy: token=logits[:,0].argmax(-1)
   if quality and i+1<total:
    nll=nll+torch.nn.functional.cross_entropy(logits[:,0].float(),ids[i+1].view(1),reduction='sum')
  decode_end.record(); decode_end.synchronize(); decode_ms=decode_start.elapsed_time(decode_end)
  final_nll=None if nll is None else float(nll.item())
  row={'prefill_total_ms':prefill_ms,'decode_total_ms':decode_ms,'nll':final_nll,'ppl':math.exp(final_nll/max(total-1,1)) if final_nll is not None else None,'prompt_compression_ms':None,'layer_page_seals':r.layer_page_seals,'kv_head_page_compressions':r.kv_head_page_compressions,'reconstruct_native_calls':r.reconstruct_native_calls,'kv_reconstruct_calls':r.kv_reconstruct_calls,'kv_reconstruct_pages':r.kv_reconstruct_pages,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
  del r.model
  del r
  torch.cuda.empty_cache()
  gc.collect()
  return row
 def label(method,m):
  if a.full_fused_only: return 'Full-native-fused'
  if method == 'ours_direct_v2': return f'Ours-Direct-Fused m{m}'
  return ('Full' if method=='full' else 'Full-through-Ours-wrapper' if method=='full_wrapper' else 'Ours-Reconstruct-Native' if method=='ours_reconstruct_native' else 'Ours-Reconstruct-Native-v2-BF16' if method=='ours_reconstruct_native_bf16' else 'Ours-Triton-Compact' if method=='ours_triton_compact' else 'Ours direct ref' if method=='ours_direct_ref' else f'Ours m{m}')
 runs=[]
 for method,m in methods:
  if not quality:
   for _ in range(max(a.warmup,0)):
    run_one(method,m)
  reps=1 if quality else max(a.repeats,1)
  for rep in range(reps):
   row=run_one(method,m); row.update({'method':label(method,m),'repeat':rep,'decode_ms_per_token':row['decode_total_ms']/a.decode,'tokens_per_sec':1000*a.decode/row['decode_total_ms'],'tokens':total,'num_prediction_tokens':total-1 if quality else 0,'mode':a.mode}); runs.append(row)
   print(row,flush=True)
   (out/'progress.json').write_text(json.dumps(runs,indent=2))
 if quality:
  (out/('smoke.json' if a.smoke else 'quality.json')).write_text(json.dumps(runs,indent=2))
 else:
  (out/('smoke.json' if a.smoke else 'benchmark_runs.json')).write_text(json.dumps(runs,indent=2))
  summary=[]
  for method in dict.fromkeys(x['method'] for x in runs):
   xs=[x['decode_ms_per_token'] for x in runs if x['method']==method]
   summary.append({'method':method,'mean_ms_per_token':statistics.mean(xs),'median_ms_per_token':statistics.median(xs),'p95_ms_per_token':statistics.quantiles(xs,n=20,method='inclusive')[18] if len(xs)>=2 else xs[0],'std_ms_per_token':statistics.stdev(xs) if len(xs)>=2 else 0.0})
  with (out/'benchmark_summary.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
if __name__=='__main__':main()
