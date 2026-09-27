"""Resumable native Full KV RULER baseline on four local GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qwen3_8b_ruler_lowmem as q
from run_qwen32b_ruler_multigpu import TASKS, install_chunked_mlp


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='/mnt/data_sda1/yhf/Qwen3-32B')
    p.add_argument('--data-dir', default=str(ROOT/'ruler_data_8b_16k'))
    p.add_argument('--output-dir', default=str(ROOT/'results/qwen32b_ruler_full_16k_first100'))
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--source-samples', type=int, default=500)
    p.add_argument('--stop-after', type=int, default=0)
    a = p.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Expose exactly local GPUs 0,1,2,3')
    cfg = AutoConfig.from_pretrained(a.model_path, local_files_only=True)
    mapping = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 3, 'lm_head': 3}
    mapping.update({f'model.layers.{i}': i*4//cfg.num_hidden_layers for i in range(cfg.num_hidden_layers)})
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(model_path=a.model_path, data_dir=a.data_dir, method='full',
                  length=a.length, samples=a.samples, source_samples=a.source_samples, tasks=TASKS,
                  dtype='bfloat16', batch_size=1, greedy=True, max_new=128,
                  template='base', seed=42, transformers=__version__, device_map=mapping,
                  physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), mlp_chunk=512)
    cp = out/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise RuntimeError('Existing output configuration mismatch')
    cp.write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    model = None
    summary = {}
    completed = 0
    for task in TASKS.split(','):
        rows = q.load_samples(Path(a.data_dir), task, a.source_samples)
        if len(rows) != a.source_samples:
            raise RuntimeError(f'{task}: expected {a.source_samples} source samples')
        rows = rows[:a.samples]
        dest = out/f'{task}.jsonl'
        saved = [json.loads(s) for s in dest.open()] if dest.exists() else []
        if len(saved) > a.samples:
            raise RuntimeError('Too many saved predictions')
        for i, row in enumerate(saved):
            if row['ordinal'] != i or row['sample_sha256'] != q.sample_key(rows[i]):
                raise RuntimeError('Resume data mismatch')
        for i in range(len(saved), a.samples):
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(a.model_path,
                    local_files_only=True, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map=mapping).eval()
                install_chunked_mlp(model, 512)
                print('MODEL_READY', flush=True)
            row = rows[i]
            prompt = row['input'] + row.get('answer_prefix', '')
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda:0')
            n = ids.shape[1]
            max_new = min(128, a.length-n)
            if max_new <= 0:
                raise RuntimeError(f'{task}/{i}: no decode room')
            for d in range(4):
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            sync()
            start = time.perf_counter()
            output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = output.past_key_values
            token = output.logits[:, -1].argmax(-1)
            sync()
            setup = time.perf_counter()-start
            decode_start = time.perf_counter()
            generated = []
            # Match Ours' fixed-length greedy loop, including its final forward.
            for _ in range(max_new):
                generated.append(int(token.item()))
                output = model(input_ids=token.view(1, 1).to('cuda:0'),
                               past_key_values=past, use_cache=True, logits_to_keep=1)
                past = output.past_key_values
                token = output.logits[:, -1].argmax(-1)
            sync()
            result = dict(index=row['index'], ordinal=i, sample_sha256=q.sample_key(row),
                outputs=row['outputs'], pred=tokenizer.decode(generated, skip_special_tokens=True),
                prompt_tokens=n, new_tokens=max_new, prompt_setup_s=setup,
                decode_s=time.perf_counter()-decode_start, elapsed_s=time.perf_counter()-start,
                peak_allocated_gib=[torch.cuda.max_memory_allocated(d)/2**30 for d in range(4)])
            with dest.open('a') as f:
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
            del output, past, token, ids
            summary[task] = dict(samples=i+1, target=a.samples, score=q.local_score(dest, task))
            (out/'summary_local.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(dict(task=task, done=i+1, target=a.samples,
                elapsed_s=result['elapsed_s'], peak_gib=result['peak_allocated_gib'])), flush=True)
            completed += 1
            if a.stop_after and completed >= a.stop_after:
                print('STOP_AFTER_REACHED', flush=True)
                return
        summary[task] = dict(samples=a.samples, target=a.samples, score=q.local_score(dest, task))
        (out/'summary_local.json').write_text(json.dumps(summary, indent=2))
        print('DONE', task, summary[task], flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
