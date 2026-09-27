"""Launch only CoFeatKV Llama8B MMLU and validate complete shard coverage."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/jhl/model_inference/llm_infer/mmlu_cofeatkv_20260926')
PYTHON = '/home/jhl/anaconda3/envs/qwen-opt/bin/python'


def worker(gpu):
    m = 64 if gpu < 4 else 80
    shard = gpu % 4
    dest = ROOT / f'm{m}' / f'shard{shard}'
    log = ROOT / 'logs' / f'm{m}_shard{shard}.log'
    command = [PYTHON, '-u', str(ROOT/'code/run_mmlu_llama.py'), '--output', str(dest),
               '--m', str(m), '--shard', str(shard)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with log.open('a') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        (ROOT/'logs'/f'gpu{gpu}.json').write_text(json.dumps(dict(gpu=gpu, m=m, shard=shard,
                                                               pid=process.pid, command=command)))
        rc = process.wait()
    return dict(gpu=gpu, m=m, shard=shard, returncode=rc, log=str(log))


def aggregate(m):
    rows = []
    configs = []
    for shard in range(4):
        dest = ROOT/f'm{m}'/f'shard{shard}'
        summary = json.loads((dest/'result.json').read_text())
        assert summary['complete'] and summary['config']['limit'] == 0
        configs.append(summary['config'])
        rows.extend(json.loads(line) for line in (dest/'predictions.jsonl').read_text().splitlines())
    assert len(rows) == 14042 and {r['global_index'] for r in rows} == set(range(14042))
    assert len({c['dataset_sha256'] for c in configs}) == 1
    assert len({c['runner_sha256'] for c in configs}) == 1
    subjects = {}
    for row in rows:
        sub = subjects.setdefault(row['subject'], dict(samples=0, correct=0))
        sub['samples'] += 1
        sub['correct'] += int(row['correct'])
    assert len(subjects) == 57
    for sub in subjects.values():
        sub['accuracy'] = sub['correct']/sub['samples']
    result = dict(model=configs[0]['model'], method='CoFeatKV', m=m, shots=0,
                  protocol=configs[0]['protocol'], samples=len(rows),
                  correct=sum(r['correct'] for r in rows),
                  accuracy=sum(r['correct'] for r in rows)/len(rows),
                  subject_macro_accuracy=sum(s['accuracy'] for s in subjects.values())/57,
                  subjects=subjects, configs=configs)
    (ROOT/f'm{m}'/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return {k:v for k,v in result.items() if k not in ['subjects', 'configs']}


def main():
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    state = dict(started_at=time.strftime('%F %T'), completed=[], status='running',
                 gpu_assignment={str(g):dict(m=64 if g<4 else 80, shard=g%4) for g in range(8)})
    statefile = ROOT/'launch_state.json'
    if statefile.exists():
        raise RuntimeError('Existing launch state; inspect before relaunch')
    statefile.write_text(json.dumps(state, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, gpu) for gpu in range(8)]
        for future in concurrent.futures.as_completed(futures):
            state['completed'].append(future.result())
            statefile.write_text(json.dumps(state, indent=2))
    state['status'] = 'failed' if any(x['returncode'] for x in state['completed']) else 'complete'
    state['finished_at'] = time.strftime('%F %T')
    if state['status'] == 'complete':
        try:
            state['results'] = [aggregate(m) for m in [64, 80]]
        except Exception as exc:
            state['status'] = 'aggregation_failed'
            state['error'] = repr(exc)
    statefile.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
