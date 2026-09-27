# Qwen3-8B LongBench result inventory

Snapshot: 2026-09-26 16:40:24, Asia/Shanghai. Running jobs will advance beyond
these counts. This is a discovery and scoring snapshot, not a final benchmark.

## Locations

Host: `jhl@10.102.34.150` (4090 server).
Model verified in Full/CoFeatKV configs: `/data/share/Qwen3-8B`.
Configured context capacity: 32,768 tokens.

Remote root: `/home/jhl/model_inference/llm_infer/`.

| Relative directory | Contents | Status |
|---|---|---|
| `longbench_outputs_qwen8b_full/` | Full, CoFeatKV m80 (`ours`), Palu, OjaKV, ReCalKV | Old generation protocol; incomplete suite |
| `longbench_outputs_qwen8b_fixed_20260926/` | Full, CoFeatKV m64, CoFeatKV m80 | Corrected generation protocol; running |
| `longbench_outputs_qwen8b_10/` | Ten-example-per-task pilots and retry shards | Separate pilots; duplicated/overlapping shards must not be pooled |

Old and corrected task predictions use `<method>/<task>/<task>.jsonl`.
Old method directory names: `full`, `ours`, `Palu`, `OjaKV`, `ReCalKV`.
Corrected method directory names: `full`, `m64`, `m80`.
Each task's `config.json` is adjacent to its prediction file.
`smoke_*` directories contain validation runs and are excluded below.

## Local search

Searched the user's `/home/jhl_1` files, including `llm_inter/svd_distr`,
`llm_inter/cofeatkv`, and `llm_inter/pukkz_svd`, pruning package/cache/tool
directories. No separate local Qwen3-8B LongBench task prediction set was found.
A name search of accessible `/mnt/data_sda1` directories also found no such
result directory; some unrelated users' directories were not readable.

Local summaries are under:
`/home/jhl_1/llm_inter/svd_distr/results/longbench_audit_20260926/`.

- `qwen_raw_scores.json`: old-run scores, prediction paths, and file hashes.
- `qwen_fixed_progress.json`: an EARLIER corrected-run snapshot, not live scores.
- This file: inventory and the timestamped snapshot above.

No running task, generation code, old prediction, or manuscript was modified
by this inventory search.

## Old-run scores

Scores below are from `qwen_raw_scores.json`; task counts were checked against
remote predictions at the snapshot time. Numbers in parentheses are counts
where coverage differs from the task target. Scores are NOT corrected results.

| Method | Qasper | MFQA | HotpotQA | 2WikiMQA | GovReport | QMSum | TREC | RepoBench-P | Total predictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Target count | 200 | 150 | 200 | 200 | 200 | 200 | 200 | 500 | 1850 |
| Full | 12.00 | 14.11 | 2.84 | 7.54 | 28.32 (56) | 17.52 | 71.50 | 62.87 (224) | 1430 |
| CoFeatKV m80 | 11.89 | 13.74 | 2.91 | 7.68 | 27.90 (56) | 17.33 | 72.00 | 61.91 (224) | 1430 |
| Palu | 9.01 | 16.15 | 4.95 | 7.35 | 28.42 (56) | 18.16 | 69.00 | 46.02 (224) | 1430 |
| OjaKV | 7.10 (57) | 14.05 | 2.79 | 7.65 | 20.63 (29) | 16.85 (19) | 70.50 | 37.03 (34) | 889 |
| ReCalKV | 9.15 | 13.60 | 2.47 | 6.53 | 27.66 (56) | 18.02 | 70.00 | 41.79 (224) | 1430 |

Old Full/CoFeatKV QA outputs were affected by unfinished thinking segments
and a generation loop without EOS stopping. These scores must not be used
as validated compression-quality comparisons. No old formal m64 prediction
set was found; m64 is present in the corrected run.

## Corrected-run coverage

Protocol: `nonthinking-eos-reserved-context-v1`.
It explicitly disables thinking on chat tasks, checks EOS, reserves generation
space, and records generated token IDs, stop reasons, and input/prompt hashes.

| Task | Target | Full | m64 | m80 |
|---|---:|---:|---:|---:|
| Qasper | 200 | 200 | 200 | 200 |
| MultiFieldQA-en | 150 | 150 | 149 | 133 |
| HotpotQA | 200 | 200 | 124 | 122 |
| 2WikiMQA | 200 | 200 | 71 | 32 |
| GovReport | 200 | 0 | 0 | 0 |
| QMSum | 200 | 0 | 0 | 0 |
| TREC | 200 | 0 | 0 | 0 |
| RepoBench-P | 500 | 0 | 0 | 0 |
| Total | 1850 | 750 | 544 | 487 |

## Corrected scores on identical samples

Recomputed from saved predictions using the remote LongBench `eval.scorer`.
Input and prompt hashes agree across all three methods for every included row.
These four tasks use QA F1, reported on the 0-100 scale. Only the Qasper
comparison covers its entire task; other rows are interim common prefixes.

| Task | Identical samples | Full | m64 | m80 |
|---|---:|---:|---:|---:|
| Qasper | 200 | 48.09 | 46.37 | 48.13 |
| MultiFieldQA-en | 133 | 50.51 | 51.33 | 51.91 |
| HotpotQA | 122 | 58.91 | 58.81 | 59.46 |
| 2WikiMQA | 32 | 35.38 | 35.38 | 38.50 |

Full's completed entire-task scores are Qasper 48.09 (200), MFQA 53.33
(150), HotpotQA 59.07 (200), and 2WikiMQA 42.95 (200). Do not compare these
entire-task scores against a shorter CoFeatKV prefix.

At the snapshot, the scheduler reported six active jobs and no failures:
GPU0 m80/MFQA, GPU1 m80/2WikiMQA, GPU2 m64/2WikiMQA,
GPU3 m64/MFQA, GPU4 m64/HotpotQA, GPU5 m80/HotpotQA.

The remote `scores_fixed.json` and `scheduler_state.json` are updated by the
existing monitor. This inventory intentionally does not overwrite them.
