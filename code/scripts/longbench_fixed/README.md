# Corrected Qwen3-8B LongBench evaluation (2026-09-26)

Remote host: `jhl@10.102.34.150`.
Remote code: `/home/jhl/model_inference/llm_infer/longbench_fixed_20260926`.
Remote outputs: `/home/jhl/model_inference/llm_infer/longbench_outputs_qwen8b_fixed_20260926`.

Scope: matched FullKV, CoFeatKV m64, CoFeatKV m80; all eight original English
LongBench tasks, 1,850 examples per method. Original results remain untouched.
No PaLu, ReCalKV, OjaKV, Llama or 32B results are represented as corrected by
this run. No manuscript scores have been changed during this rerun.

Search: all discovered LongBench output directories and their config files on
4090 contained no formal Qwen3-8B m64 run. The sole m64 LongBench config found
was the Qwen3-32B one-example smoke run.

Fixes:
- Explicit non-thinking chat template (`enable_thinking=False`) on all chat tasks.
- Stop on model generation EOS IDs and tokenizer EOS, before feeding them back.
- Reserve generation budget and chat wrapper space during middle truncation.
- Retokenize with `add_special_tokens=False` for identical actual input counts.
- Record token IDs, stop reason, source and prompt hashes, and code checksums.
- Reject resume when configuration or input identity changes.

Validation completed:
- Five deterministic unit tests for stopping/budget handling.
- Full/m64/m80 each completed two examples for Qasper, HotpotQA, GovReport,
  RepoBench-P; identical input/prompt hashes, no tokens after EOS.
- Former failure inputs GovReport index 56 and RepoBench-P index 224 now fit
  their context+generation limits in tokenizer tests (full inference pending).

Scheduler launched 2026-09-26 16:14:31 remote server clock, PID 1882296,
uses physical GPUs 0–5 only. Scoring monitor PID 1883375 updates
`scores_fixed.json` every 180 seconds and once the scheduler finishes.
Check live PIDs rather than assuming they remain valid.

Files to monitor:
- `scheduler_state.json`: running, completed and failed tasks.
- `logs/<full|m64|m80>_<task>.log`: per-task logs.
- `scores_fixed.json`: counts/scores plus matched-prefix interim comparisons.

Only compare scores on identical samples. Interim prefix scores are not final
benchmark results and must not replace final manuscript scores. Validate full
coverage and generation logs before updating the paper. Never mix ten-example
pilot results, smoke outputs, old raw runs, and corrected runs.
