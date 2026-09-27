"""Zero-shot MMLU with compressed-history option scoring, sharded by example."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


CHOICES = 'ABCD'


def load_examples(data):
    examples = []
    for path in sorted((Path(data) / 'test').glob('*_test.csv')):
        subject = path.stem.removesuffix('_test')
        with path.open(newline='') as stream:
            for index, row in enumerate(csv.reader(stream)):
                if len(row) != 6 or row[5] not in CHOICES:
                    raise ValueError(f'Invalid MMLU row: {path}/{index}')
                prompt = ('The following are multiple choice questions (with answers) about '
                          + subject.replace('_', ' ') + '.\n\n' + row[0])
                prompt += ''.join(f'\n{choice}. {row[i+1]}' for i, choice in enumerate(CHOICES))
                prompt += '\nAnswer:'
                examples.append(dict(global_index=len(examples), subject=subject,
                                     subject_index=index, gold=row[5], prompt=prompt))
    if len(examples) != 14042 or len({x['subject'] for x in examples}) != 57:
        raise ValueError('Expected official MMLU test: 14,042 examples, 57 subjects')
    return examples


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@torch.inference_mode()
def score_options(runner, ids, choice_ids):
    n = ids.numel()
    # Always reserve the final prompt token for the compressed attention path.
    first = ((n - 1) // runner.page) * runner.page
    if first < runner.page:
        raise ValueError('Prompt too short to test compressed history')
    if n + 2 > runner.max_tokens:
        raise ValueError('Prompt exceeds configured capacity; do not truncate MMLU')
    runner.reset_runtime()
    capacity = ((n + runner.page - 1) // runner.page) * runner.page
    runner.prepare_capacity(capacity)
    positions = torch.arange(n, device='cuda:0', dtype=torch.long)
    before = runner.direct_attention_calls
    runner.prefill(ids[:first].to('cuda:0'), positions[:first])
    runner.seal_prefill()
    for pos in range(first, n):
        logits = runner.run_token(ids[pos], pos)[:, -1]
    direct_calls = runner.direct_attention_calls - before
    if direct_calls <= 0:
        raise RuntimeError('Compressed direct attention was not exercised')
    scores = logits[0, choice_ids].float()
    if not torch.isfinite(scores).all():
        raise RuntimeError('Nonfinite choice logits')
    return scores.cpu().tolist(), direct_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data/share/Meta-Llama-3.1-8B-Instruct')
    ap.add_argument('--data', default='/data/share/mmlu')
    ap.add_argument('--runner-dir', default='/home/jhl/model_inference/llm_infer')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--m', type=int, choices=[64, 80], required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--num-shards', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=4096)
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError('Invalid shard')
    sys.path.insert(0, args.runner_dir)
    import benchmark_autoregressive_lowmem_llama as bench
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    encoded_choices = [tokenizer(' ' + c, add_special_tokens=False).input_ids for c in CHOICES]
    if any(len(ids) != 1 for ids in encoded_choices):
        raise ValueError(f'Choice strings must each be one token: {encoded_choices}')
    choice_ids = [ids[0] for ids in encoded_choices]
    if len(set(choice_ids)) != 4:
        raise ValueError('Choice tokens are not distinct')
    examples = load_examples(args.data)
    selected = [x for x in examples if x['global_index'] % args.num_shards == args.shard]
    if args.limit:
        selected = selected[:args.limit]
    # Preflight every selected prompt before spending time evaluating the shard.
    encoded = [tokenizer(x['prompt'], add_special_tokens=True, return_tensors='pt').input_ids[0]
               for x in selected]
    if any(len(ids) <= 32 or len(ids) + 2 > args.max_tokens for ids in encoded):
        raise ValueError('A prompt falls outside supported lengths')
    config = dict(model=args.model, method='CoFeatKV', runner_method='ours_direct_v2',
                  m=args.m, page_size=32, shots=0, prompt_format='plain MMLU, Answer:',
                  option_strings=[' ' + c for c in CHOICES], option_ids=choice_ids,
                  dtype='bfloat16', max_tokens=args.max_tokens, shard=args.shard,
                  num_shards=args.num_shards, limit=args.limit, target=len(selected),
                  protocol='cofeatkv-compressed-final-prompt-token-v1',
                  dataset_sha256=hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Output configuration mismatch')
    atomic_json(cp, config)
    dest = args.output / 'predictions.jsonl'
    saved = [json.loads(line) for line in dest.read_text().splitlines()] if dest.exists() else []
    if len(saved) > len(selected):
        raise ValueError('Too many existing predictions')
    for old, expected in zip(saved, selected):
        if old['global_index'] != expected['global_index'] or old['gold'] != expected['gold']:
            raise ValueError('Resume input mismatch')
    print(json.dumps(dict(event='PREFLIGHT_OK', **config, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          min_length=min(map(len, encoded)), max_length=max(map(len, encoded)))), flush=True)
    runner = bench.load_runner(args.model, 'ours_direct_v2', args.m, 'cuda:0',
                               max_tokens=args.max_tokens, attn_implementation='sdpa', logits_to_keep=1)
    records = saved
    started = time.perf_counter()
    with dest.open('a') as stream:
        for i in range(len(saved), len(selected)):
            example = selected[i]
            start = time.perf_counter()
            scores, calls = score_options(runner, encoded[i], choice_ids)
            predicted = CHOICES[max(range(4), key=scores.__getitem__)]
            result = {k: v for k, v in example.items() if k != 'prompt'}
            result.update(pred=predicted, correct=predicted == example['gold'], choice_logits=scores,
                          prompt_tokens=len(encoded[i]), direct_attention_calls=calls,
                          elapsed_s=time.perf_counter() - start,
                          prompt_sha256=hashlib.sha256(example['prompt'].encode()).hexdigest())
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            records.append(result)
            if (i + 1) % 25 == 0 or i < 3 or i + 1 == len(selected):
                progress = dict(m=args.m, shard=args.shard, completed=i+1, target=len(selected),
                                accuracy=sum(r['correct'] for r in records)/len(records),
                                elapsed_s=time.perf_counter()-started)
                atomic_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    subjects = {}
    for row in records:
        sub = subjects.setdefault(row['subject'], dict(samples=0, correct=0))
        sub['samples'] += 1
        sub['correct'] += int(row['correct'])
    result = dict(config=config, samples=len(records), correct=sum(r['correct'] for r in records),
                  accuracy=sum(r['correct'] for r in records)/len(records), subjects=subjects,
                  complete=len(records)==len(selected))
    atomic_json(args.output/'result.json', result)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
