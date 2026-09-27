#!/usr/bin/env python
"""Qwen3 PPL comparison for direct pre/post-RoPE K SVD without K reconstruction."""
import argparse
import json
import math
import time
import types
import sys
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def svd_factor(k, rank):
    if not 1 <= rank <= min(k.shape):
        raise ValueError(f"Invalid rank {rank} for {tuple(k.shape)}")
    # Factors are consumed directly in score computation; K is never rebuilt.
    gram = k.float() @ k.float().T
    _, vec = torch.linalg.eigh(gram)
    left = vec[:, -rank:].contiguous()
    right = left.T @ k.float()
    return left, right


def parse_method(method):
    if method == "full":
        return "full", 1
    match = re.fullmatch(r"(pre|post|ours)(\d+)", method)
    if match is None:
        raise ValueError(f"Invalid method: {method}")
    mode, rank = match[1], int(match[2])
    if rank <= 0:
        raise ValueError("Rank must be positive")
    return mode, rank


def pre_scores(q_rot, left, right, cos, sin):
    # Move each key token's rotation onto the query. The intermediate is
    # query x token x rank, never a reconstructed token x feature K matrix.
    inverse_queries = rotate(q_rot[:, None, :], cos[None], -sin[None])
    projected = inverse_queries @ right.T
    return (projected * left[None]).sum(-1)


def patch_model(model, mode, rank, page_size):
    root = model.model
    if mode == "ours":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ruler_resume_20260919_1652" / "code"))
        from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page
    else:
        weighted_coalescing_page = None
    states = []
    for layer_id, layer in enumerate(root.layers):
        attn = layer.self_attn
        nh = int(getattr(attn, "num_heads", model.config.num_attention_heads))
        nk = int(getattr(attn, "num_key_value_heads", model.config.num_key_value_heads))
        hd = int(getattr(attn, "head_dim", attn.k_proj.out_features // nk))
        gqa = nh // nk; scale = float(getattr(attn, "scaling", hd ** -0.5))
        state = {"sealed": [[] for _ in range(nk)], "values": [[] for _ in range(nk)],
                 "starts": [[] for _ in range(nk)]}
        states.append(state)

        def forward(self, hidden_states, position_embeddings=None, _nh=nh, _nk=nk, _hd=hd,
                    _gqa=gqa, _scale=scale, _state=state, _mode=mode, _rank=rank,
                    _page_size=page_size, **kwargs):
            if hidden_states.shape[0] != 1 or hidden_states.shape[1] > _page_size:
                raise RuntimeError("Expected batch=1 and one page per forward")
            bsz, seq, _ = hidden_states.shape
            q = self.q_norm(self.q_proj(hidden_states).view(1, seq, _nh, _hd).transpose(1, 2))
            k = self.k_norm(self.k_proj(hidden_states).view(1, seq, _nk, _hd).transpose(1, 2))
            v = self.v_proj(hidden_states).view(1, seq, _nk, _hd).transpose(1, 2)
            cos, sin = position_embeddings
            q_rot = rotate(q, cos[:, None], sin[:, None])
            k_rot = rotate(k, cos[:, None], sin[:, None])
            source_k = k if _mode == "pre" else k_rot
            if _mode == "full":
                current = [(source_k[0, kh].float(), v[0, kh].float()) for kh in range(_nk)]
            elif _mode == "ours":
                current = [weighted_coalescing_page(k_rot[0, kh].float(),
                            q_rot[0, kh * _gqa:(kh + 1) * _gqa].float(), _rank,
                            return_compact=True) for kh in range(_nk)]
            else:
                current = [svd_factor(source_k[0, kh].float(), _rank) for kh in range(_nk)]
            out_heads = []
            for qh in range(_nh):
                kh = qh // _gqa
                q_now = q_rot[0, qh].float()
                score_parts, value_parts = [], []
                for bi, rep in enumerate(_state["sealed"][kh]):
                    if _mode == "full":
                        old_k, old_v = rep
                        score_parts.append((q_now @ old_k.T) * _scale)
                        value_parts.append(old_v)
                        continue
                    if _mode == "ours":
                        c, coeff, owner = rep
                        grouped_q = torch.zeros((seq, c.shape[1]), device=q_now.device)
                        grouped_q.scatter_add_(1, owner[None].expand(seq, -1), q_now * coeff[None])
                        score_parts.append((grouped_q @ c.T) * _scale)
                        value_parts.append(_state["values"][kh][bi])
                        continue
                    left, right = rep
                    if _mode == "pre":
                        start = _state["starts"][kh][bi]
                        stop = start + left.shape[0]
                        scores_old = pre_scores(q_now, left, right,
                                                self._rope_cos[start:stop], self._rope_sin[start:stop])
                    else:
                        scores_old = (q_now @ right.T) @ left.T
                    score_parts.append(scores_old * _scale)
                    value_parts.append(_state["values"][kh][bi])
                # Current page stays dense for causal masking, as in the existing PPL evaluator.
                current_scores = q_now @ k_rot[0, kh].float().T * _scale
                row = torch.arange(seq, device=hidden_states.device)[:, None]
                col = torch.arange(seq, device=hidden_states.device)[None, :]
                current_scores[col > row] = -torch.inf
                score_parts.append(current_scores)
                value_parts.append(v[0, kh].float())
                scores = torch.cat(score_parts, dim=-1)
                vals = torch.cat(value_parts, dim=0)
                probs = torch.softmax(scores, dim=-1)
                out_heads.append(probs @ vals)
            out = torch.stack(out_heads, dim=0).transpose(0, 1).unsqueeze(0).to(hidden_states.dtype)
            # Seal the current page only after it has served as the dense current page.
            for kh in range(_nk):
                _state["sealed"][kh].append(current[kh])
                _state["values"][kh].append(v[0, kh].float())
                _state["starts"][kh].append(int(kwargs.get("cache_position", 0)[0].item()
                                                if torch.is_tensor(kwargs.get("cache_position", None))
                                                else 0))
            return self.o_proj(out.reshape(bsz, seq, -1)), None

        attn._rope_cos = {}
        attn._rope_sin = {}
        original_forward = attn.forward
        attn.forward = types.MethodType(forward, attn)
        attn._svd_original_forward = original_forward
    model._svd_states = states
    model._svd_mode = mode
    model._svd_rank = rank


def set_position_tables(model, max_len, device):
    root = model.model
    pos = torch.arange(max_len, device=device)[None, :]
    hidden = torch.zeros((1, max_len, model.config.hidden_size), device=device, dtype=model.dtype)
    cos, sin = root.rotary_emb(hidden, pos)
    for layer in root.layers:
        layer.self_attn._rope_cos = cos[0].float()
        layer.self_attn._rope_sin = sin[0].float()


def reset_states(model):
    for state in model._svd_states:
        state["sealed"] = [[] for _ in state["sealed"]]
        state["values"] = [[] for _ in state["values"]]
        state["starts"] = [[] for _ in state["starts"]]


@torch.inference_mode()
def eval_sequence(model, ids, device, page_size):
    total_nll = 0.0
    for start in range(0, ids.numel(), page_size):
        x = ids[start:start + page_size].unsqueeze(0).to(device)
        pos = torch.arange(start, start + x.shape[1], device=device)
        logits = model(input_ids=x, use_cache=False, cache_position=pos).logits.float()
        count = min(x.shape[1], ids.numel() - start - 1)
        target = ids[start + 1:start + count + 1].to(device)
        total_nll += float(F.cross_entropy(logits[0, :count], target, reduction="sum"))
        del logits
    return total_nll, int(ids.numel() - 1)


def load_tokens(tokenizer, data_path, needed):
    import pyarrow.parquet as pq
    files = sorted(Path(data_path).glob("validation*.parquet"))
    texts = []
    chars = 0
    target_chars = max(needed * 5, 1024)
    for file in files:
        for x in pq.read_table(file, columns=["text"]).column("text").to_pylist():
            if x:
                text = str(x).strip()
                texts.append(text); chars += len(text)
                if chars >= target_chars:
                    break
        if chars >= target_chars:
            break
    ids = tokenizer("\n\n".join(texts), add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    if ids.numel() < needed:
        raise RuntimeError(f"WikiText supplied {ids.numel()} tokens, need {needed}")
    return ids[:needed]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/mnt/data_sda1/share/Qwen3-8B")
    p.add_argument("--data-path", default="/mnt/data1/jhl1/llama_data/llm_data/wikitext-2-raw-v1")
    p.add_argument("--output-dir", default="results/rope_svd_qwen8b_wikitext/pre_post_ppl")
    p.add_argument("--num-blocks", type=int, default=13)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--methods", default="full,pre64,pre80,pre96,post64,post80,post96")
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as tf_version
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    ids = load_tokens(tok, a.data_path, a.num_blocks * a.seq_len)
    results = {}
    for method in [x.strip() for x in a.methods.split(",") if x.strip()]:
        model = AutoModelForCausalLM.from_pretrained(a.model_path, local_files_only=True,
            dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager").eval().to(a.device)
        mode, rank = parse_method(method)
        print(f"CONFIG method={method} mode={mode} actual_rank={rank}", flush=True)
        patch_model(model, mode, rank, a.page_size)
        set_position_tables(model, a.seq_len, a.device)
        rows = []; total_nll = total_tokens = 0
        for block in range(a.num_blocks):
            reset_states(model)
            t0 = time.perf_counter()
            nll, ntok = eval_sequence(model, ids[block*a.seq_len:(block+1)*a.seq_len], a.device, a.page_size)
            total_nll += nll; total_tokens += ntok
            row = {"method": method, "block_id": block, "ppl": math.exp(nll/ntok), "nll": nll, "tokens": ntok, "seconds": time.perf_counter()-t0}
            rows.append(row); print(row, flush=True)
            results[method] = {"rows": rows, "aggregate_ppl": math.exp(total_nll/total_tokens),
                               "mode": mode, "actual_rank": rank}
            (out / "pre_post_ppl.json").write_text(json.dumps({"implementation": "v2_exact_token_phase",
                "model_path": a.model_path, "num_blocks": a.num_blocks, "seq_len": a.seq_len,
                "page_size": a.page_size, "results": results}, indent=2))
        results[method] = {"rows": rows, "aggregate_ppl": math.exp(total_nll/total_tokens),
                           "mode": mode, "actual_rank": rank}
        del model; torch.cuda.empty_cache()
    (out / "pre_post_ppl.json").write_text(json.dumps({"implementation": "v2_exact_token_phase",
        "model_path": a.model_path, "transformers": tf_version,
        "num_blocks": a.num_blocks, "seq_len": a.seq_len, "page_size": a.page_size, "results": results}, indent=2))
    for m, r in results.items(): print(m, r["aggregate_ppl"])


if __name__ == "__main__":
    main()
