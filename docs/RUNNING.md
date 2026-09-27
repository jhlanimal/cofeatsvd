
# CoFeatKV: File Layout and Reproduction Guide

This directory contains the current CoFeatKV paper, source code, figures, and
saved experiment results. It does not contain model weights or datasets.

## Layout

```text
paper/                 LaTeX source, bibliography, figures, and compiled PDF
code/experiments/      RoPE, SVD, RULER, and LongBench experiment scripts
code/scripts/          MMLU and corrected LongBench runners
code/qwen_optimize/    Qwen cache, grouping, decode, profiling, and test code
results/local/         Local experiment results
results/qwen_optimize/ Qwen optimization and runtime results
results/remote/        Copied results from the 4090 server
docs/                  This document
```

## Environment

The original runs used a Python environment with PyTorch, Transformers,
CUDA, NumPy, SciPy, Matplotlib, and the packages required by the benchmark
scripts. Model and dataset paths are machine-specific. Before running, edit
the model and data paths in the selected script or pass its command-line
arguments.

## Recorded software versions

The following versions were recorded from the environments used for the
experiments. The local and remote machines are not identical; use the matching
environment for the corresponding result set.

### Local `qwen-opt` environment

```text
Python       3.11.16
PyTorch      2.5.1+cu121
CUDA in PyTorch 12.1
Transformers 4.57.6
NumPy        1.26.4
SciPy        1.17.1
GPU          NVIDIA GeForce RTX 3090
Driver       575.57.08
```

Activate it with:

```bash
conda activate qwen-opt
```

### Remote 4090 `qwen-opt` environment

```text
Python       3.11.16
PyTorch      2.5.1+cu124
CUDA in PyTorch 12.4
Transformers 4.51.3
NumPy        2.4.6
SciPy        1.17.1
GPU          NVIDIA GeForce RTX 4090
Driver       595.58.03
```

The remote Python executable was:

```text
/home/jhl/anaconda3/envs/qwen-opt/bin/python
```

Check the active environment before a run:

```bash
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
nvidia-smi
```

The system default Python on the development machine was not the experiment
environment and did not provide PyTorch. Do not use it for the commands below.

## Core method and call chain

The method is implemented in the following files:

```text
Compression and grouping:
  code/qwen_optimize/optimized_grouped_compression.py

Reference grouping implementation:
  code/qwen_optimize/reference_grouped_compression.py

Autoregressive cache and direct attention runner:
  code/qwen_optimize/benchmark_autoregressive_full_vs_ours.py
  code/qwen_optimize/benchmark_autoregressive_lowmem.py

CUDA/Triton decode kernels:
  code/qwen_optimize/triton_page_parallel_attention_v2.py
  code/qwen_optimize/triton_compact_decode_attention.py
  code/qwen_optimize/fused_decode_ops.py
```

The standalone optimized compression function is `compress_optimized_exact`
in `optimized_grouped_compression.py`. The archived autoregressive runners
used by the paper call `weighted_coalescing_page` and
`weighted_coalescing_pages_batched` from
`eval_qwen3_multiseq_ppl_diagnostics.py` when a page is sealed. Both paths
produce the compact directions, coefficients, and owner indices used by
CoFeatKV; the former is the optimized standalone implementation, while the
latter is the implementation used by the saved benchmark runs.

The runners also import these auxiliary files from the original source tree:

```text
/home/jhl_1/llm_inter/pukkz_svd/qwen/eval_qwen3_multiseq_ppl_diagnostics.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/analyze_post_rope_k_pages.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/eval_weighted_vs_puzzlekv.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/analyze_attention_scale_similarity.py
```

The current package contains the main files, but these auxiliary files must
also be on `PYTHONPATH` for the original Qwen runtime scripts. For example:

```bash
export PYTHONPATH=/home/jhl_1/llm_inter/iclr/cofeatsvd/code/qwen_optimize:/home/jhl_1/llm_inter/pukkz_svd/qwen:$PYTHONPATH
```

The remote MMLU runs used:

```text
Python: /home/jhl/anaconda3/envs/qwen-opt/bin/python
Model:  /data/share/Qwen3-8B
MMLU:   /data/share/mmlu/test
GPU:    8 x RTX 4090
```

## Build the paper

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd/paper
pdflatex -interaction=nonstopmode iclr2027_conference.tex
pdflatex -interaction=nonstopmode iclr2027_conference.tex
```

The output is `iclr2027_conference.pdf`. Run LaTeX from `paper/` so that the
relative figure and bibliography paths resolve correctly.

## Regenerate figures

The figure scripts are in `paper/scripts/`. For example:

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd
python paper/scripts/plot_feature_similarity.py
python paper/scripts/plot_method_overview_v2.py
```

The scripts expect the corresponding inputs under `results/` or the paths
defined near the top of each script. Check those paths before rerunning.

## Run the core diagnostics

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd
python code/experiments/eval_global_svd_one_sequence.py --help
python code/experiments/compare_attention_outputs_one_sequence.py --help
python code/experiments/analyze_rope_lowrank.py --help
python code/experiments/analyze_grouped_rope_lowrank.py --help
```

These scripts evaluate the pre-RoPE/post-RoPE SVD diagnostics and the
page-local CoFeatKV representation. They require the local WikiText-2 path
and the Qwen model path used by the original configuration.

## Main runtime commands

For the original Qwen autoregressive quality and throughput benchmark:

```bash
conda activate qwen-opt
cd /home/jhl_1/llm_inter/iclr/cofeatsvd/code/qwen_optimize
export PYTHONPATH=$PWD:/home/jhl_1/llm_inter/pukkz_svd/qwen:$PYTHONPATH
python benchmark_autoregressive_full_vs_ours.py --help
```

Use `benchmark_autoregressive_lowmem.py` instead when running the low-memory
path. The exact model, data, output, page size, and group budget are specified
by the script arguments and should be checked with `--help` before a full run.

For the completed Qwen3-8B MMLU runs, the top-level launcher was:

```bash
/home/jhl/anaconda3/envs/qwen-opt/bin/python -u \
  /home/jhl_1/llm_inter/iclr/cofeatsvd/code/scripts/mmlu_cofeatkv/launch_mmlu.py \
  --root /path/to/output \
  --model /data/share/Qwen3-8B
```

This launcher starts `run_mmlu_llama.py` on eight GPUs. GPUs 0--3 use
`m=64`; GPUs 4--7 use `m=80`. The worker creates the model runner, invokes
the CoFeatKV compressed attention path, and writes one prediction file per
shard.

## Run Qwen3-32B RULER

Local and multi-GPU entry points are:

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd
python code/experiments/run_qwen32b_ruler_full_local.py --help
python code/experiments/run_qwen32b_ruler_multigpu.py --help
```

The shell launcher is:

```bash
bash code/experiments/launch_qwen32b_ruler_16k_m64_remote.sh
```

Confirm the model path, dataset path, GPU assignment, context length, and
output directory before starting a full run.

## Run CoFeatKV MMLU

The launcher runs four shards for each group budget. The original mapping was
GPU 0--3 for `m=64` and GPU 4--7 for `m=80`.

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd
python code/scripts/mmlu_cofeatkv/launch_mmlu.py \
  --root /path/to/output \
  --model /path/to/Qwen3-8B
```

The runner evaluates all 57 MMLU subjects with the plain zero-shot prompt.
Each completed run writes shard predictions and an aggregate `result.json`.
The copied remote results are under:

```text
results/remote/mmlu_qwen8b_cofeatkv_20260926/
results/remote/mmlu_llama8b_cofeatkv_20260926/
```

## Run corrected LongBench

The corrected generation and scoring scripts are under
`code/scripts/longbench_fixed/`:

```bash
cd /home/jhl_1/llm_inter/iclr/cofeatsvd
python code/scripts/longbench_fixed/run_qwen_longbench_fixed.py --help
python code/scripts/longbench_fixed/score_fixed.py --help
```

Read `code/scripts/longbench_fixed/README.md` before running. Do not use old
raw outputs as final benchmark results without checking the generation-stop
configuration.

## Reusing saved results

The main paper tables can be traced to the JSON, CSV, NPZ, and Markdown files
under `results/local/`, `results/qwen_optimize/`, and `results/remote/`.
The files are preserved as produced by the original runs; no model weights or
datasets were copied into this package.
