"""Offline whole-sequence SVD diagnostic; the fitted basis sees future tokens."""
import argparse
import hashlib
import json
import math
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from eval_qwen3_pre_post_svd_ppl import load_tokens, rotate, pre_scores


def patch(model, mode, rank):
    stats = []
    for li, layer in enumerate(model.layers):
        attn = layer.self_attn
        nh, nk = model.config.num_attention_heads, model.config.num_key_value_heads
        d = attn.head_dim
        original = attn.forward

        def forward(self, hidden_states, position_embeddings=None, _li=li, **kwargs):
            n = hidden_states.shape[1]
            q = self.q_norm(self.q_proj(hidden_states).view(1, n, nh, d)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(1, n, nk, d)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(1, n, nk, d).transpose(1, 2)
            cos, sin = position_embeddings
            qr = rotate(q, cos[:, None], sin[:, None]).float()[0]
            kr = rotate(k, cos[:, None], sin[:, None]).float()[0]
            output = torch.empty((nh, n, d), device=q.device, dtype=hidden_states.dtype)
            for h in range(nk):
                if mode != 'full':
                    source = k[0, h].float() if mode.startswith('pre') else kr[h]
                    u, s, vh = torch.linalg.svd(source, full_matrices=False)
                    left = (u[:, :rank] * s[:rank]).contiguous()
                    right = vh[:rank].contiguous()
                    stats.append(dict(layer=_li, head=h, shape=list(source.shape), rank=rank,
                                      discarded_energy=float(s[rank:].square().sum()/s.square().sum())))
                    if mode == 'pre_latent':
                        # Explicit diagnostic convention: standard RoPE in r dimensions.
                        inv = model.config.rope_theta ** (-torch.arange(0, rank, 2, device=q.device).float()/rank)
                        freq = torch.arange(n, device=q.device).float()[:, None] * inv[None]
                        phase = torch.cat((freq, freq), -1)
                        lc, ls = phase.cos(), phase.sin()
                        low_k = rotate(left, lc, ls)
                for j in range(h*(nh//nk), (h+1)*(nh//nk)):
                    if mode == 'pre_latent':
                        low_q = rotate(q[0, j].float() @ right.T, lc, ls)
                    chunk = 16 if mode == 'pre_exact' else 256
                    for start in range(0, n, chunk):
                        end = min(start+chunk, n)
                        if mode == 'full':
                            score = qr[j, start:end] @ kr[h].T
                        elif mode == 'post':
                            score = (qr[j, start:end] @ right.T) @ left.T
                        elif mode == 'pre_exact':
                            score = pre_scores(qr[j, start:end], left, right, cos[0].float(), sin[0].float())
                        else:
                            score = low_q[start:end] @ low_k.T
                        score *= self.scaling
                        mask = torch.arange(n, device=q.device)[None] > torch.arange(start, end, device=q.device)[:, None]
                        score.masked_fill_(mask, -torch.inf)
                        output[j, start:end] = (score.softmax(-1) @ v[0, h].float()).to(output.dtype)
            return self.o_proj(output.transpose(0, 1).reshape(1, n, nh*d)), None

        attn._diagnostic_original = original
        attn.forward = types.MethodType(forward, attn)
    return stats


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-path', default='/mnt/data_sda1/share/Qwen3-8B')
    p.add_argument('--data-path', default='/mnt/data1/jhl1/llama_data/llm_data/wikitext-2-raw-v1')
    p.add_argument('--methods', default='native,full,post128,pre_exact128,pre_latent128,post64,post80,post96,pre_latent64,pre_latent80,pre_latent96,pre_exact64,pre_exact80,pre_exact96')
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    ids = load_tokens(tok, a.data_path, 4096).to('cuda:0')
    lm = AutoModelForCausalLM.from_pretrained(a.model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda:0')
    results = {}
    config = dict(vars(a), transformers=__version__, seq_len=4096, num_sequences=1,
                  input_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                  warning='Offline whole-sequence basis uses future tokens; NOT strictly causal online PPL.',
                  lowdim_rope='Qpre @ V_r, Kpre @ V_r; RoPE theta^(-2j/r); original attention scale d^-0.5',
                  exact_rope='original d-dimensional per-token RoPE moved onto query; no dense K reconstruction')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    native_hidden = None
    for method in a.methods.split(','):
        if method in ('native', 'full'):
            mode, rank = method, 128
        else:
            import re
            match = re.fullmatch(r'(post|pre_exact|pre_latent)(\d+)', method)
            mode, rank = match[1], int(match[2])
        stats = [] if mode == 'native' else patch(lm.model, mode, rank)
        start = time.perf_counter()
        print('START', method, flush=True)
        hidden = lm.model(input_ids=ids[None], use_cache=False).last_hidden_state
        nll = 0.
        for i in range(0, 4095, 128):
            end = min(i+128, 4095)
            logits = lm.lm_head(hidden[:, i:end]).float()
            nll += float(F.cross_entropy(logits[0], ids[i+1:end+1], reduction='sum'))
        torch.cuda.synchronize()
        results[method] = dict(ppl=math.exp(nll/4095), nll=nll, tokens=4095,
                               seconds=time.perf_counter()-start, factorization_count=len(stats))
        if mode == 'native':
            native_hidden = hidden.cpu()
        elif mode == 'full' and native_hidden is not None:
            results[method]['native_hidden_max_error'] = float((hidden.cpu().float()-native_hidden.float()).abs().max())
        if stats:
            results[method]['mean_discarded_energy'] = sum(x['discarded_energy'] for x in stats)/len(stats)
            (out/f'{method}_factors.json').write_text(json.dumps(stats, indent=2))
        print(method, results[method], flush=True)
        (out/'results.json').write_text(json.dumps(dict(config=config, results=results), indent=2))
        if mode != 'native':
            for layer in lm.model.layers:
                layer.self_attn.forward = layer.self_attn._diagnostic_original
                del layer.self_attn._diagnostic_original
        del hidden, logits
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
