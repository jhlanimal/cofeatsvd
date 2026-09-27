"""Compare local attention outputs using one unmodified Full forward's Q/K/V."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]/'pukkz_svd'))
from benchmark_autoregressive_lowmem import weighted_coalescing_pages_batched
from eval_qwen3_pre_post_svd_ppl import load_tokens, rotate, pre_scores


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    cosine = F.cosine_similarity(actual, reference, dim=-1, eps=1e-12)
    error = (actual-reference).square().sum(-1)
    energy = reference.square().sum(-1)
    return dict(cosine_mean=float(cosine.mean()), cosine_median=float(cosine.median()),
                cosine_p05=float(torch.quantile(cosine.flatten(), .05)),
                relative_squared_error=float(error.sum()/energy.sum()),
                relative_l2_error=float((error.sum()/energy.sum()).sqrt()),
                reference_zero_vectors=int((energy < 1e-20).sum())), dict(
                    cosine=cosine.cpu().numpy(), squared_error=error.cpu().numpy(),
                    reference_energy=energy.cpu().numpy())


def build_ours(k, q, m, page=32):
    nk, n, d = k.shape
    gqa = q.shape[0]//nk
    pages = n//page
    kp = k.reshape(nk*pages, page, d)
    qp = q.reshape(nk, gqa, pages, page, d).permute(0, 2, 1, 3, 4).reshape(nk*pages, gqa, page, d)
    parts = [[], [], []]
    for start in range(0, len(kp), 16):
        values = weighted_coalescing_pages_batched(kp[start:start+16], qp[start:start+16], m)
        for dst, value, dtype in zip(parts, values, (torch.bfloat16, torch.bfloat16, torch.uint8)):
            dst.append(value.to(dtype))
    c, coeff, owner = (torch.cat(p) for p in parts)
    return c.reshape(nk, pages, page, m), coeff.reshape(nk, pages, d), owner.reshape(nk, pages, d)


def ours_scores(query, c, coeff, owner):
    pages, page, m = c.shape
    projected = torch.zeros((query.shape[0], pages, m), device=query.device)
    projected.scatter_add_(2, owner.long()[None].expand(query.shape[0], -1, -1),
                           query[:, None, :] * coeff.float()[None])
    return torch.einsum('qpr,ptr->qpt', projected, c.float()).reshape(query.shape[0], pages*page)


def attention(q, k, v, scale, mode, rank, pre=None, cos=None, sin=None, page=32):
    nh, n, d = q.shape
    nk = k.shape[0]
    output = torch.empty((nh, n, d), device=q.device, dtype=torch.float32)
    factors = build_ours(k, q, rank, page) if mode == 'ours' else None
    key_index = torch.arange(n, device=q.device)[None]
    for kh in range(nk):
        if mode in ('pre', 'post'):
            source = pre[kh].float() if mode == 'pre' else k[kh].float()
            u, s, vh = torch.linalg.svd(source, full_matrices=False)
            left, right = (u[:, :rank]*s[:rank]).contiguous(), vh[:rank].contiguous()
            del u, s, vh, source
        for qh in range(kh*(nh//nk), (kh+1)*(nh//nk)):
            chunk = 16 if mode == 'pre' else 128
            for start in range(0, n, chunk):
                end = min(start+chunk, n)
                query = q[qh, start:end].float()
                if mode == 'full':
                    score = query @ k[kh].float().T
                elif mode == 'post':
                    score = (query @ right.T) @ left.T
                elif mode == 'pre':
                    score = pre_scores(query, left, right, cos.float(), sin.float())
                else:
                    score = ours_scores(query, *(f[kh] for f in factors))
                    # Existing method keeps the active page dense until it seals.
                    for pos in range(start, end, page):
                        stop = min(pos+page, end)
                        score[pos-start:stop-start, pos:stop] = q[qh, pos:stop].float() @ k[kh, pos:stop].float().T
                score *= scale
                score.masked_fill_(key_index > torch.arange(start, end, device=q.device)[:, None], -torch.inf)
                output[qh, start:end] = score.softmax(-1) @ v[kh].float()
    return output


def score_checks(device):
    torch.manual_seed(42)
    q = torch.randn(7, 128, device=device)
    k = torch.randn(64, 128, device=device)
    u, s, vh = torch.linalg.svd(k, full_matrices=False)
    left, right = u*s, vh
    phase = torch.randn(64, 64, device=device).repeat(1, 2)
    cos, sin = phase.cos(), phase.sin()
    direct = pre_scores(q, left, right, cos, sin)
    reference = q @ rotate(left@right, cos, sin).T
    torch.testing.assert_close(direct, reference, atol=1e-4, rtol=1e-4)
    c, coeff, owner = build_ours(k[None].bfloat16(), torch.randn(4, 64, 128, device=device).bfloat16(), 64)
    dense = c[0].float().gather(-1, owner[0].long()[:, None, :].expand(-1, 32, -1))*coeff[0].float()[:, None]
    grouped = ours_scores(q, c[0], coeff[0], owner[0])
    torch.testing.assert_close(grouped, q@dense.reshape(64, 128).T, atol=1e-4, rtol=1e-4)
    return dict(pre_score_max_abs=float((direct-reference).abs().max()),
                ours_score_max_abs=float((grouped-q@dense.reshape(64, 128).T).abs().max()))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='/mnt/data_sda1/share/Qwen3-8B')
    p.add_argument('--data-path', default='/mnt/data1/jhl1/llama_data/llm_data/wikitext-2-raw-v1')
    p.add_argument('--output-dir', default='results/rope_svd_qwen8b_wikitext/attention_o_one_sequence')
    a = p.parse_args()
    from transformers import AutoModel, AutoTokenizer, __version__
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = score_checks('cuda:0')
    print('SCORE_CHECKS', checks, flush=True)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    ids = load_tokens(tok, a.data_path, 4096)
    config = dict(vars(a), selected_layers=[9, 18, 27], zero_based=True, seq_len=4096,
                  num_sequences=1, ranks=[64, 80, 96], ours_page=32, seed=42,
                  transformers=__version__, physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                  input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                  comparison='same Full Q/K/V, pre-o_proj O; FP32 scores and reductions',
                  pre='whole-sequence SVD + exact original per-token RoPE, factor-only scores',
                  post='whole-sequence post-RoPE SVD, factor-only scores',
                  ours='real weighted_coalescing_pages_batched; BF16 c/coeff, uint8 owners; dense active page',
                  caveats=['one-sequence descriptive diagnostic, no confidence intervals',
                           'global SVD uses future-token statistics',
                           'rank vs m is NOT storage-matched',
                           'Ours differs in paging and active-page handling, not only grouping'],
                  score_checks=checks)
    model = AutoModel.from_pretrained(a.model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda:0')
    config.update(num_layers=model.config.num_hidden_layers, num_kv_heads=model.config.num_key_value_heads,
                  num_query_heads=model.config.num_attention_heads, head_dim=model.layers[9].self_attn.head_dim)
    nh, nk = config['num_query_heads'], config['num_kv_heads']
    d = config['head_dim']
    captures = {li: {} for li in config['selected_layers']}
    handles = []
    for li in captures:
        attn = model.layers[li].self_attn
        capture = captures[li]
        for name in ('q_norm', 'k_norm', 'v_proj'):
            def save(module, inputs, output, name=name, capture=capture):
                capture[name] = output.detach().cpu()
            handles.append(getattr(attn, name).register_forward_hook(save))
        def save_position(module, args, kwargs, capture=capture):
            capture['cos'], capture['sin'] = (x.detach().cpu() for x in kwargs['position_embeddings'])
        handles.append(attn.register_forward_pre_hook(save_position, with_kwargs=True))
        def save_o(module, args, capture=capture):
            capture['native_o'] = args[0].detach().cpu()
        handles.append(attn.o_proj.register_forward_pre_hook(save_o))
        capture['scale'] = float(attn.scaling)
    result = model(input_ids=ids[None].to('cuda:0'), use_cache=False)
    del result
    for h in handles:
        h.remove()
    del model, attn
    gc.collect()
    torch.cuda.empty_cache()
    print('CAPTURE_DONE model released', flush=True)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    torch.save(dict(ids=ids, layers=captures), out/'full_captures.pt')
    rows = []
    for li, cap in captures.items():
        q = cap['q_norm'].to('cuda:0').transpose(1, 2)
        pre = cap['k_norm'].to('cuda:0').transpose(1, 2)
        v = cap['v_proj'].to('cuda:0').reshape(1, 4096, nk, d).transpose(1, 2)[0]
        cos, sin = cap['cos'].to('cuda:0'), cap['sin'].to('cuda:0')
        q = rotate(q, cos[:, None], sin[:, None])[0]
        k = rotate(pre, cos[:, None], sin[:, None])[0]
        pre = pre[0]
        assert q.shape == (nh, 4096, d) and k.shape == (nk, 4096, d)
        full = attention(q, k, v, cap['scale'], 'full', d)
        native = cap['native_o'].to('cuda:0').reshape(4096, nh, d).transpose(0, 1)
        check, _ = metrics(full, native)
        print('NATIVE_O_CHECK', li, check, flush=True)
        if check['relative_squared_error'] > 1e-3:
            raise RuntimeError(f'Full O reference mismatch in layer {li}: {check}')
        (out/f'layer{li}_native_check.json').write_text(json.dumps(check, indent=2))
        del native
        for mode, rank in [('post', 128), ('pre', 128)]+[(mode, rank) for rank in (64, 80, 96) for mode in ('pre', 'post', 'ours')]:
            start = time.perf_counter()
            actual = attention(q, k, v, cap['scale'], mode, rank, pre, cos[0], sin[0])
            summary, arrays = metrics(actual, full)
            row = dict(layer=li, method=mode, rank_or_m=rank, **summary, seconds=time.perf_counter()-start)
            rows.append(row)
            np.savez_compressed(out/f'layer{li}_{mode}{rank}_per_head_token.npz', **arrays)
            (out/'results.json').write_text(json.dumps(dict(config=config, rows=rows), indent=2))
            print(json.dumps(row), flush=True)
            if rank == 128 and summary['relative_squared_error'] > 1e-3:
                raise RuntimeError('Full-rank reconstruction sanity check failed')
            del actual
        del full, q, k, v, pre, cos, sin
        torch.cuda.empty_cache()
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
