"""Matched non-thinking LongBench generation with explicit stopping and audit metadata."""
import hashlib
import time
import torch


def stop_ids(model, tok):
    values = getattr(model.generation_config, 'eos_token_id', None)
    values = values if isinstance(values, (list, tuple)) else [values]
    return {int(x) for x in [*values, tok.eos_token_id] if x is not None}


def build_prompt(tok, text, chat, max_context, max_new):
    budget = max_context - max_new - 2
    if budget <= 0:
        raise ValueError('Generation budget exceeds context capacity')
    source = tok(text, add_special_tokens=False).input_ids
    keep = min(len(source), budget)
    for _ in range(12):
        content = text if keep >= len(source) else tok.decode(source[:keep//2], skip_special_tokens=True) + tok.decode(source[-(keep-keep//2):], skip_special_tokens=True)
        prompt = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) if chat else content
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        if n <= budget:
            return prompt
        keep -= n - budget + 32
        if keep <= 0:
            raise ValueError('Cannot fit chat wrapper and content')
    raise ValueError('Prompt truncation failed to converge')


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def decode_loop(token, step, first_pos, max_new, stops):
    generated = []
    reason = 'max_new_tokens'
    for j in range(max_new):
        value = int(token.item())
        generated.append(value)
        if value in stops:
            reason = 'eos'
            break
        if j + 1 < max_new:
            token = step(token, first_pos + j).argmax(-1)
    return generated, reason


def finish(tok, ids, reason, setup, start, n, device):
    sync(device)
    meta = {'generated_token_ids': ids, 'stop_reason': reason, 'generated_tokens': len(ids)}
    return tok.decode(ids, skip_special_tokens=True), setup, time.perf_counter()-start, n, meta


@torch.inference_mode()
def generate_full(model, tok, prompt, max_new, device):
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    start = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    token = out.logits[:, -1].argmax(-1)
    sync(device)
    setup = time.perf_counter()-start
    def step(t, pos):
        nonlocal past
        result = model(input_ids=t.view(1, 1).to(device), past_key_values=past, use_cache=True, logits_to_keep=1)
        past = result.past_key_values
        return result.logits[:, -1]
    start = time.perf_counter()
    generated, reason = decode_loop(token, step, ids.shape[1], max_new, stop_ids(model, tok))
    return finish(tok, generated, reason, setup, start, ids.shape[1], device)


@torch.inference_mode()
def generate_ours(runner, tok, prompt, max_new, device):
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    n = int(ids.numel())
    if n + max_new + 2 > runner.max_tokens:
        raise ValueError('Runner capacity exceeded')
    runner.reset_runtime()
    first = n // runner.page * runner.page
    if not first:
        raise ValueError('Prompt shorter than compression page')
    capacity = (n + max_new + runner.page - 1) // runner.page * runner.page
    runner.prepare_capacity(capacity)
    positions = torch.arange(n + max_new + 2, device=device, dtype=torch.long)
    start = time.perf_counter()
    out = runner.prefill(ids[:first].to(device), positions[:first])
    runner.seal_prefill()
    logits = out.logits[:, -1]
    for pos in range(first, n):
        logits = runner.run_token(ids[pos], pos)[:, -1]
    token = logits.argmax(-1)
    sync(device)
    setup = time.perf_counter()-start
    start = time.perf_counter()
    generated, reason = decode_loop(token, lambda t, pos: runner.run_token(t, pos)[:, -1], n, max_new, stop_ids(runner.model, tok))
    return finish(tok, generated, reason, setup, start, n, device)
