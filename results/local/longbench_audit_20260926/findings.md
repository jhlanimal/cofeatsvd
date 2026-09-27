# Table 10 generation audit — 2026-09-26

Remote: jhl@10.102.34.150
Predictions: /home/jhl/model_inference/longbench_results/ours_m80
Generator: /home/jhl/model_inference/llm_infer/llama_8b_ruler_lowmem.py:89–125
Caller: /home/jhl/model_inference/llm_infer/longbench_ours_llama.py
Resume caller: /home/jhl/model_inference/llm_infer/longbench_ours_llama_resume_g5.py
Scorer: /home/jhl/model_inference/LongBench/LongBench/eval.py
Metrics: /home/jhl/model_inference/LongBench/LongBench/metrics.py

## Confirmed implementation defect

The generation loop appends and feeds back tokens until max_new regardless
of end-of-sequence/end-of-turn tokens. It never checks a stop-token set.
The model generation_config.json lists stop IDs 128001, 128008, 128009.
Saved output removes special tokens (`skip_special_tokens=True`) but retains
ordinary role text and later turns. No generated token IDs were saved.
Official QA scoring measures token overlap over the entire saved prediction.
Thus the evaluator reproduces the low scores, but receives contaminated answers.

## Diagnostic only — do NOT publish these as corrected benchmark scores

We split saved predictions once on the literal `assistant\n\n` marker to
measure sensitivity to post-answer continuation. No references were used to
choose the split. This heuristic cannot reliably recover the first stop token.

| Task | n | Predictions containing marker | Original score | Diagnostic split score |
|---|---:|---:|---:|---:|
| Qasper | 200 | 200 | 13.43 | 46.76 |
| MultiFieldQA-en | 150 | 141 | 25.18 | 54.29 |
| HotpotQA | 200 | 198 | 14.06 | 54.69 |
| 2WikiMQA | 200 | 192 | 14.33 | 42.68 |

All four tasks have unique indices and no empty predictions.
Example: HotpotQA index 1 starts with the correct answer, Charles L. Clifford,
then emits role text, David Niven, and repeated answers. Original F1 is
0.428571; taking the first segment produces F1 1.0.

## Consequences and next verification

The Table 10 numbers are scores of a defective generation run, not reliable
measurements of the intended answer quality. Do not attribute the QA loss to
compression, and do not replace the table with the diagnostic heuristic scores.
Stop on the model-defined EOS/EOT tokens during generation; save generated
IDs, stop reason, prompt/configuration identifiers, and model revision.
Verify on a small paired FullKV/CoFeatKV sample with identical prompts,
truncation, generation limits and stopping rules before rerunning all tasks.
The supplied baseline rows have no verified matching run configuration, so
apples-to-apples superiority claims remain unsupported even after this fix.

This audit preserves original predictions and does not modify benchmark
scores, production generation code, or the manuscript.
