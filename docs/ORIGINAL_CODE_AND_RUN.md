# CoFeatKV 原始代码、环境与运行说明

本文档记录原始代码的位置和原始实验的运行方式。这里的“原始路径”
指实验实际使用的目录，不是当前整理包中的复制路径。

## 1. 环境

### 本机 Qwen 环境

```text
Conda environment: qwen-opt
Python:            3.11.16
PyTorch:           2.5.1+cu121
PyTorch CUDA:      12.1
Transformers:      4.57.6
NumPy:             1.26.4
SciPy:             1.17.1
GPU:               NVIDIA GeForce RTX 3090
Driver:            575.57.08
```

使用：

```bash
conda activate qwen-opt
```

### 4090 服务器环境

服务器：`jhl@10.102.34.150`

```text
Python executable: /home/jhl/anaconda3/envs/qwen-opt/bin/python
Python:            3.11.16
PyTorch:           2.5.1+cu124
PyTorch CUDA:      12.4
Transformers:      4.51.3
NumPy:             2.4.6
SciPy:             1.17.1
GPU:               NVIDIA GeForce RTX 4090
Driver:            595.58.03
```

连接服务器：

```bash
ssh jhl@10.102.34.150
```

## 2. 原始代码路径

### CoFeatKV 核心代码

```text
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/
```

主要文件：

```text
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/benchmark_autoregressive_full_vs_ours.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/benchmark_autoregressive_lowmem.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/optimized_grouped_compression.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/reference_grouped_compression.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/fused_decode_ops.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/triton_compact_decode_attention.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize/triton_page_parallel_attention_v2.py
```

历史实验实际调用的辅助文件：

```text
/home/jhl_1/llm_inter/pukkz_svd/qwen/eval_qwen3_multiseq_ppl_diagnostics.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/src/analyze_attention_scale_similarity.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/src/analyze_post_rope_k_pages.py
/home/jhl_1/llm_inter/pukkz_svd/qwen/src/eval_weighted_vs_puzzlekv.py
```

### SVD、RoPE 和论文诊断代码

```text
/home/jhl_1/llm_inter/svd_distr/experiments/
```

主要入口：

```text
/home/jhl_1/llm_inter/svd_distr/experiments/eval_global_svd_one_sequence.py
/home/jhl_1/llm_inter/svd_distr/experiments/compare_attention_outputs_one_sequence.py
/home/jhl_1/llm_inter/svd_distr/experiments/analyze_rope_lowrank.py
/home/jhl_1/llm_inter/svd_distr/experiments/analyze_grouped_rope_lowrank.py
/home/jhl_1/llm_inter/svd_distr/experiments/eval_qwen3_pre_post_svd_ppl.py
```

### RULER 和 LongBench 代码

```text
/home/jhl_1/llm_inter/svd_distr/experiments/run_qwen32b_ruler_full_local.py
/home/jhl_1/llm_inter/svd_distr/experiments/run_qwen32b_ruler_multigpu.py
/home/jhl_1/llm_inter/svd_distr/experiments/run_qwen32b_longbench.py
/home/jhl_1/llm_inter/svd_distr/experiments/launch_qwen32b_ruler_16k_m64_remote.sh
```

4090 服务器上的 RULER 运行目录：

```text
/home/jhl/model_inference/llm_infer/
```

其中包含模型加载、RULER 数据和多卡 runner。Qwen3-32B 运行脚本实际从
这个目录调用 `run_qwen32b_ruler_multigpu.py`。

### MMLU 代码

```text
/home/jhl_1/llm_inter/svd_distr/scripts/mmlu_cofeatkv/launch_mmlu.py
/home/jhl_1/llm_inter/svd_distr/scripts/mmlu_cofeatkv/run_mmlu_llama.py
```

4090 服务器上运行时，代码副本位于：

```text
/home/jhl/model_inference/llm_infer/mmlu_qwen8b_cofeatkv_20260926/code/
```

## 3. 原始代码调用关系

```text
benchmark_autoregressive_full_vs_ours.py
        |
        +-- weighted_coalescing_page(s)
        |      from eval_qwen3_multiseq_ppl_diagnostics.py
        |
        +-- compact attention / reconstruction
        |      from triton_compact_decode_attention.py
        |
        +-- page-parallel attention
        |      from triton_page_parallel_attention_v2.py
        |
        +-- fused RoPE and RMSNorm
               from fused_decode_ops.py
```

`optimized_grouped_compression.py` 是独立的优化压缩实现和基准代码；保存
到论文结果中的历史 autoregressive runs 主要使用
`eval_qwen3_multiseq_ppl_diagnostics.py` 中的 page compression 函数。

## 4. 运行原始 Qwen autoregressive 实验

```bash
conda activate qwen-opt
cd /home/jhl_1/llm_inter/pukkz_svd/qwen/optimize
export PYTHONPATH=/home/jhl_1/llm_inter/pukkz_svd/qwen/optimize:/home/jhl_1/llm_inter/pukkz_svd/qwen:/home/jhl_1/llm_inter/pukkz_svd/qwen/src:$PYTHONPATH
```

先查看参数：

```bash
python benchmark_autoregressive_full_vs_ours.py --help
python benchmark_autoregressive_lowmem.py --help
```

运行实际实验时，需要根据机器修改模型路径、WikiText-2 路径、输出路径、
`m`、prefill 长度和 decode 长度。不要直接覆盖已有结果目录。

## 5. 运行 4090 上的 Qwen3-32B RULER

原始 shell 入口：

```text
/home/jhl_1/llm_inter/svd_distr/experiments/launch_qwen32b_ruler_16k_m64_remote.sh
```

脚本实际执行：

```bash
cd /home/jhl/model_inference/llm_infer
/home/jhl/anaconda3/envs/qwen-opt/bin/python -u \
  run_qwen32b_ruler_multigpu.py \
  --runner-dir /home/jhl/model_inference/llm_infer \
  --model-path /data/share/Qwen3-32B \
  --data-dir /home/jhl/model_inference/llm_infer/ruler_data_8b_16k \
  --output-dir /home/jhl/model_inference/llm_infer/results/qwen32b_ruler_16k_m64_first100 \
  --length 16384 --m 64 --source-samples 500 --samples 100
```

32K 或 `m=80` 时，修改 `--length`、`--m` 和输出目录。运行前检查 GPU
数量、模型路径和 RULER 数据目录。

## 6. 运行 MMLU

```bash
/home/jhl/anaconda3/envs/qwen-opt/bin/python -u \
  /home/jhl_1/llm_inter/svd_distr/scripts/mmlu_cofeatkv/launch_mmlu.py \
  --root /home/jhl/model_inference/llm_infer/mmlu_qwen8b_cofeatkv_new \
  --model /data/share/Qwen3-8B
```

默认分配：GPU 0--3 运行 `m=64`，GPU 4--7 运行 `m=80`。每个配置使用 4 个
shard，完成后生成 `m64/result.json` 和 `m80/result.json`。

## 7. 运行前检查

```bash
/home/jhl/anaconda3/envs/qwen-opt/bin/python -c \
  "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

确认模型和数据存在：

```bash
ls -ld /data/share/Qwen3-8B
ls -ld /data/share/Qwen3-32B
ls -ld /data/share/mmlu/test
```
