# RTX 4090 experiment inventory, 2026-09-26

Host: `jhl@10.102.34.150` (`sdu`). At inspection, all eight RTX 4090 GPUs
were idle and no evaluation process for this user was running. Idle does not
mean that every evaluation completed. This inventory records remote prediction
files and logs, not the older local result snapshots.

Paths below are relative to `/home/jhl/model_inference/llm_infer` unless stated.
Completion refers to prediction coverage; it does not imply that benchmark
scores or comparison protocols have been independently validated.

## RULER: our method

| Model | Context | m | Prediction coverage | Remote path |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 16K | 80 | 11 tasks x 500, complete | `llama_16k_ruler_g{0..5}/ours` |
| Llama-3.1-8B-Instruct | 32K | 80 | 11 tasks x 500, complete | `llama_32k_m80_g{0..5}` |
| Llama-3.1-8B-Instruct | 32K | 64 | 11 tasks x 100, complete | `llama_32k_m64_first100_v2_g{0..5}` |
| Qwen3-8B | 32K | 64 | 11 tasks x 500, complete | `qwen_32k_m64_g{0..5}/ours` |
| Qwen3-8B | 32K | 80 | 5,001 / 5,500; VT has only 1 / 500 | `remote_results_ours_32k_g{0..5}/ours` |
| Qwen3-8B | 32K | 96 | 11 tasks x 100, complete | `results/qwen8b_ruler_32k_m96_first100` |
| Qwen3-32B | 16K | 64 | 11 tasks x 100, complete | `results/qwen32b_ruler_16k_m64_first100` |
| Qwen3-32B | 16K | 80 | 11 tasks x 100, complete | `results/qwen32b_ruler_16k_m80_first100` |
| Qwen3-32B | 32K | 80 | 11 tasks x 100, complete; log ends ALL_DONE | `results/qwen32b_ruler_32k_m80_first100` |

The older `llama_32k_m64_g*` run contains only 416 predictions across six
tasks. Do not concatenate it with the completed first-100 v2 run.
No predictions were found in `results/qwen8b_ruler_32k_p64_m64_first100`
or `results/qwen32b_ruler_32k_p64_m64_first100`; their logs reach MODEL_READY.

## Additional RULER baselines

Absolute root:
`/home/jhl/model_inference/pasvd_baseline/results/ruler_qwen3_100/Qwen3-8B`.
Both `16384` and `32768` contain PaLu and ReCalKV predictions for all
11 tasks, each with 100 records. Files are named `<task>/Palu.json` and
`<task>/ReCalKV.json` but contain newline-delimited JSON, not JSON arrays.
The directory name does not make these results CoFeatKV results.

## LongBench

### Llama-3.1-8B-Instruct, ours m=80

Absolute root: `/home/jhl/model_inference/longbench_results/ours_m80`.
All 1,850 predictions are present, with unique sample indices within tasks:

| Task | Predictions |
|---|---|
| Qasper | 200 / 200 |
| MultiFieldQA-en | 150 / 150 |
| HotpotQA | 200 / 200 |
| 2WikiMQA | 200 / 200 |
| GovReport | 200 / 200 |
| QMSum | 200 / 200 |
| TREC | 200 / 200 |
| RepoBench-P | 500 / 500 |

Resume log: `/home/jhl/model_inference/logs/longbench_ours_m80_resume_g5.log`.
Official scoring import in the remote qwen3_fsdp environment failed because
`jieba` was missing. This audit therefore does not assign final scores.

### Qwen3-8B formal runs

Root: `longbench_outputs_qwen8b_full`.

| Task | Full / ours m80 / PaLu / ReCalKV, each | OjaKV |
|---|---|---|
| Qasper | 200 / 200 | 57 / 200 |
| MultiFieldQA-en | 150 / 150 | 150 / 150 |
| HotpotQA | 200 / 200 | 200 / 200 |
| 2WikiMQA | 200 / 200 | 200 / 200 |
| GovReport | 56 / 200 | 29 / 200 |
| QMSum | 200 / 200 | 19 / 200 |
| TREC | 200 / 200 | 200 / 200 |
| RepoBench-P | 224 / 500 | 34 / 500 |
| Total | 1,430 / 1,850 | 889 / 1,850 |

Files follow `<method>/<task>/<task>.jsonl`.
Our GovReport and RepoBench-P logs end with `no generation room after
truncation`, at indices 56 and 224. The OjaKV Qasper log ends with CUDA OOM.
These runs stopped with errors; they are not currently progressing.

The separate `longbench_outputs_qwen8b_10/scores.json` contains a pilot
evaluation with 10 examples per task. Do not substitute pilot averages for
the formal evaluation or mix their predictions.

### Qwen3-32B

Root: `longbench_outputs_32b`.
Full contains Qasper 200, MultiFieldQA-en 150, HotpotQA 200, 2WikiMQA 200,
and GovReport 56: 806 / 1,850 predictions.
Only one-example Qasper smoke outputs were found for ours m64 and m80;
no completed formal run for our method was found here.

## Paper integration checks

- The previous conclusion that Llama 32K m80 and Llama LongBench were
  incomplete was based on stale snapshots and is superseded by this audit.
- Score the completed Llama LongBench predictions before adding table values.
- Match model revision, sample IDs/count, prompt, context cap, metric and
  compression settings before comparing remote results with supplied tables.
- The screenshot PASVD row is not our method. Never relabel it as CoFeatKV.
- Keep completed 100-example and 500-example RULER evaluations separate.
- This audit did not launch jobs or modify the manuscript.
