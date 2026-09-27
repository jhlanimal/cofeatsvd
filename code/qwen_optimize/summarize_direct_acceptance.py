"""Collect independent timing and corrected token-weighted PPL acceptance runs."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path(__file__).parent/'results')
    args = ap.parse_args()
    root = args.results
    quality = json.loads((root/'direct_fused_quality_final_20260916/quality.json').read_text())
    quality = [r for r in quality if r['method'] == 'Full'] + json.loads((root/'direct_qkv_quality_20260916/quality.json').read_text())
    by_method = {r['method']: r for r in quality}
    rows = []
    for name, mode in [('direct_fused_greedy_final_20260916', 'greedy_standard_full'),
                       ('full_fused_greedy_final_20260916', 'greedy_full_fused'),
                       ('direct_qkv_greedy_20260916', 'greedy_ours_final')]:
        runs = json.loads((root/name/'benchmark_runs.json').read_text())
        if mode == 'greedy_standard_full':
            runs = [r for r in runs if r['method'] == 'Full']
        for method in dict.fromkeys(r['method'] for r in runs):
            rs = [r for r in runs if r['method'] == method]
            times = [r['decode_ms_per_token'] for r in rs]
            row = {'mode': mode, 'method': method, 'repetitions': len(rs),
                   'decode_mean_ms': statistics.mean(times),
                   'decode_median_ms': statistics.median(times),
                   'decode_p95_ms': statistics.quantiles(times, n=20, method='inclusive')[18],
                   'prefill_mean_ms': statistics.mean(r['prefill_total_ms'] for r in rs),
                   'end_to_end_mean_ms': statistics.mean(r['prefill_total_ms'] + r['decode_total_ms'] for r in rs),
                   'peak_allocated_bytes': max(r['peak_allocated'] for r in rs),
                   'teacher_forced_ppl': by_method.get(method, {}).get('ppl')}
            rows.append(row)
    out = root/'direct_acceptance_20260916'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps({'quality': quality, 'timing': rows}, indent=2))
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
