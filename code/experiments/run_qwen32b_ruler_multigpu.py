#!/usr/bin/env python3
"""Layer-sharded CoFeatKV RULER evaluation, preserving the existing builder."""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

import torch


TASKS = ('niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,'
         'niah_multikey_2,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1')


def install_gqa_adapter(bench):
    original = bench.compact_page_attention_v2

    def attention(q, c, coeff, owner, hist_v, active_k, active_v, scale, **kwargs):
        if q.shape[0] == 32:
            return original(q, c, coeff, owner, hist_v, active_k, active_v, scale, **kwargs)
        if q.shape != (64, 128) or c.shape[1] != 8:
            raise ValueError(f'Unsupported attention geometry: {q.shape}, {c.shape}')
        grouped = q.reshape(8, 8, 128)
        parts = [original(grouped[:, start:start + 4].contiguous().reshape(32, 128),
                          c, coeff, owner, hist_v, active_k, active_v, scale, **kwargs)
                 .reshape(8, 4, 128) for start in (0, 4)]
        return torch.cat(parts, dim=1).reshape(64, 128)

    bench.compact_page_attention_v2 = attention


def runner_class(bench):
    class ShardedRunner(bench.Runner):
        def prepare_capacity(self, seq):
            if self.capacity >= seq:
                return
            if seq % self.page:
                raise ValueError('Capacity must be page aligned')
            pages = seq // self.page
            for li, layer in enumerate(self.root.layers):
                weight = layer.self_attn.k_proj.weight
                device, dtype = weight.device, weight.dtype
                nk = self.model.config.num_key_value_heads
                hd = layer.self_attn.head_dim
                gqa = self.model.config.num_attention_heads // nk
                self.workspaces[li] = (None, None)
                self.compact_stores[li] = {
                    'c': torch.empty((pages, nk, self.page, self.m), device=device, dtype=dtype),
                    'coeff': torch.empty((pages, nk, hd), device=device, dtype=dtype),
                    'owner': torch.empty((pages, nk, hd), device=device, dtype=torch.uint8),
                    'pages': 0,
                }
                self.active_buffers[li] = (
                    torch.empty((nk, self.page, hd), device=device, dtype=dtype),
                    torch.empty((nk, self.page, gqa, hd), device=device, dtype=dtype),
                )
            self.capacity = seq

        def seal_prefill(self):
            if self.phase != 'prefill' or not self.capacity:
                raise RuntimeError('Expected prepared native prefill')
            for li, layer in enumerate(self.root.layers):
                device = layer.self_attn.k_proj.weight.device
                k, v = self.workspaces[li]
                with torch.cuda.device(device):
                    self.workspaces[li] = (k, v.to(device))
            self.phase = 'decode'

        def seal_pending(self):
            for li, pending in enumerate(self.pending_seals):
                if not pending:
                    continue
                store = self.compact_stores[li]
                active_k, pq = self.active_buffers[li]
                with torch.cuda.device(active_k.device):
                    rep = bench.weighted_coalescing_pages_batched(active_k[:, :self.page], pq, self.m)
                    page = store['pages']
                    store['c'][page].copy_(rep[0])
                    store['coeff'][page].copy_(rep[1])
                    store['owner'][page].copy_(rep[2])
                    self.kv_head_page_compressions += len(self.states[li])
                    store['pages'] += 1
                    self.layer_page_seals += 1
                    self.active_lens[li] = 0
                    self.pending_seals[li] = False

    return ShardedRunner


def install_device_contexts(model):
    # Triton kernels must launch with the owning layer's CUDA device current.
    for module in [*model.model.layers, model.model.norm, model.lm_head]:
        original = module.forward
        device = next(module.parameters()).device

        @functools.wraps(original)
        def forward(*args, _original=original, _device=device, **kwargs):
            with torch.cuda.device(_device):
                return _original(*args, **kwargs)

        module.forward = forward


def install_chunked_mlp(model, chunk):
    for layer in model.model.layers:
        original = layer.mlp.forward

        def forward(x, _original=original):
            if x.shape[1] <= chunk:
                return _original(x)
            result = torch.empty_like(x)
            for start in range(0, x.shape[1], chunk):
                result[:, start:start + chunk] = _original(x[:, start:start + chunk])
            return result

        layer.mlp.forward = forward


def sync_all():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runner-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--tasks', default=TASKS)
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--m', type=int, default=64)
    p.add_argument('--source-samples', type=int, default=500)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--max-new', type=int, default=128)
    p.add_argument('--mlp-chunk', type=int, default=512)
    p.add_argument('--stop-after', type=int, default=0,
                   help='Exit after this many new predictions; resume config is unchanged')
    a = p.parse_args()
    if not 0 < a.samples <= a.source_samples:
        raise ValueError('Invalid sample count')
    sys.path.insert(0, a.runner_dir)
    # The shared weighted-coalescing helper imports this legacy project module.
    # Keep the runner independent while resolving the existing implementation.
    sys.path.insert(0, str(Path(a.runner_dir).parent / 'pukkz_svd'))
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    import benchmark_autoregressive_lowmem as bench
    import qwen3_8b_ruler_lowmem as q
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__

    torch.manual_seed(42)
    install_gqa_adapter(bench)
    torch.cuda.set_device(0)
    ndev = torch.cuda.device_count()
    cfg = AutoConfig.from_pretrained(a.model_path, local_files_only=True)
    mapping = {'model.embed_tokens': 0, 'model.rotary_emb': 0,
               'model.norm': ndev - 1, 'lm_head': ndev - 1}
    mapping.update({f'model.layers.{i}': min(i * ndev // cfg.num_hidden_layers, ndev - 1)
                    for i in range(cfg.num_hidden_layers)})
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k != 'stop_after'}
    config.update(device_map=mapping, physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  method='ours_direct_v2', dtype='bfloat16', page=32, seed=42,
                  template='base', transformers=__version__, batch_size=1)
    cp = out / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise RuntimeError('Output config mismatch; use a new directory')
    cp.write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    runner = None
    summary = {}
    completed = 0
    for task in a.tasks.split(','):
        q.prepare(Path(a.data_dir), a.model_path, a.length, [task], a.source_samples)
        rows = q.load_samples(Path(a.data_dir), task, a.source_samples)
        if len(rows) != a.source_samples:
            raise RuntimeError(f'{task}: expected {a.source_samples} source rows')
        rows = rows[:a.samples]
        dest = out / f'{task}.jsonl'
        saved = [json.loads(s) for s in dest.open()] if dest.exists() else []
        if len(saved) > len(rows):
            raise RuntimeError('Too many saved rows')
        for i, row in enumerate(saved):
            if row['ordinal'] != i or row['sample_sha256'] != q.sample_key(rows[i]):
                raise RuntimeError(f'{task}: resume data mismatch')
        for i in range(len(saved), len(rows)):
            if runner is None:
                model = AutoModelForCausalLM.from_pretrained(
                    a.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True, use_safetensors=True,
                    attn_implementation='sdpa', device_map=mapping).eval()
                bench.install_rms(model)
                runner = runner_class(bench)(model, 'ours_direct_v2', a.m,
                                            max_tokens=a.length + a.max_new + 64,
                                            logits_to_keep=1)
                install_device_contexts(model)
                install_chunked_mlp(model, a.mlp_chunk)
                print('MODEL_READY', flush=True)
            row = rows[i]
            prompt = row['input'] + row.get('answer_prefix', '')
            prompt_tokens = len(tok(prompt).input_ids)
            max_new = min(a.max_new, a.length - prompt_tokens)
            if max_new <= 0:
                raise RuntimeError(f'{task}/{i}: {prompt_tokens} tokens leave no decode room')
            # Release old capacities before reallocation across the six devices.
            runner.reset_runtime()
            runner.workspaces = [(None, None) for _ in runner.states]
            runner.compact_stores = [None for _ in runner.states]
            runner.active_buffers = [(None, None) for _ in runner.states]
            runner.capacity = 0
            for device in range(ndev):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            sync_all()
            start = time.perf_counter()
            pred, setup, decode, n = q.generate_ours(runner, tok, prompt, max_new, 'cuda:0')
            sync_all()
            result = dict(index=row['index'], ordinal=i, outputs=row['outputs'], pred=pred,
                          sample_sha256=q.sample_key(row), prompt_tokens=n, new_tokens=max_new,
                          prompt_setup_s=setup, decode_s=decode,
                          elapsed_s=time.perf_counter() - start,
                          peak_allocated_gib=[torch.cuda.max_memory_allocated(d) / 2**30
                                              for d in range(ndev)])
            with dest.open('a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            completed += 1
            summary[task] = dict(samples=i + 1, target=a.samples, score=q.local_score(dest, task))
            (out / 'summary_local.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(dict(task=task, done=i + 1, target=a.samples,
                                  elapsed_s=result['elapsed_s'],
                                  peak_gib=result['peak_allocated_gib'])), flush=True)
            if a.stop_after and completed >= a.stop_after:
                print('STOP_AFTER_REACHED', flush=True)
                return
        summary[task] = dict(samples=len(rows), target=a.samples, score=q.local_score(dest, task))
        (out / 'summary_local.json').write_text(json.dumps(summary, indent=2))
        print(f'DONE {task} {summary[task]}', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
