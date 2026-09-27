# Direct decode validation (2026-09-16)

## Execution path

`benchmark_autoregressive_full_vs_ours.py --direct-test` selects the direct
`m=80` implementation. Historical scores use a sparse per-page projection
`q @ A`, followed by `C @ projected_q`. No historical dense K is constructed
by this path, either in PyTorch or inside its attention kernel. The active
page remains exact until its attention output has been consumed. V is dense.

The implementation includes fused BF16 RMSNorm/RoPE, one combined decode QKV
projection, removal of prefill-only capture hooks after prefill, and batched
page sealing across KV heads. Compression importance, assignments and the
weighted representative definition are unchanged. Batched EVD and changed
GEMM shapes can introduce numerical differences; the acceptance bound is
PPL degradation <= 1% relative to Full on the identical evaluation tokens.

## Bugs corrected

- Historical V was incorrectly converted from `[head, token, dim]` to
  `[page, head, page_token, dim]`. Permuting token/head then reshaping does
  not perform this conversion. The correct view splits tokens into pages
  before permuting heads/pages. Random contiguous page tensors did not test
  this integration error.
- The earlier PyTorch reference explicitly built `hist_k`, even though
  `kv_reconstruct_calls` stayed zero. That counter alone was not evidence
  of reconstruction-free execution. Both the direct kernel and current
  `--direct-ref-test` now use compact projections.
- The reduction kernel iterated through 2048 slots regardless of page count.
  It now visits only valid blocks.
- PPL omitted the last prefill position's prediction of the first decode
  input while still dividing by `total_tokens - 1`. Both Full and Ours now
  include that prediction. Old values are retained but should not be mixed
  with corrected PPL results.

## Verification and measurement

- `test_direct_layout.py`: 36 combinations of m, page count and active length,
  with strided V workspaces, against a dense oracle used only in the test.
- `test_fused_decode_ops.py`: tested fused RMSNorm/RoPE outputs match the
  BF16 references bitwise.
- `test_batched_page_builder.py`: 24 page/head cases; owners match exactly,
  effective K matches within 1e-4.
- PPL: 128 prompt tokens + 2048 teacher-forced decode tokens; 2175 predictions.
- Timing: 128 prompt tokens + 2048 greedy decode steps, batch 1, one full
  warmup and three fresh-model repetitions per method. No early EOS exit.
  CUDA events surround synchronized phases; decode includes every page seal.
  Checkpoint loading is outside timing. Prefill is separately recorded.
- `cuda:2` is a **GeForce RTX 3090** in this environment, not the A6000 at
  physical `nvidia-smi` index 2. CUDA enumeration order differs.

Compare both standard native eager Full and native Full with the same
RMSNorm/RoPE fusions. Do not attribute all system speedup to cache compression.
The Full-fused comparison does not additionally combine QKV projections.
No claim is made against FlashAttention, SDPA, CUDA graphs, or an optimized
serving engine.

Current memory usage is not a compact-memory benchmark: the runner still
allocates a full-size K workspace (only its active slice is read by direct
attention), FP32 compact stores, int64 owner metadata, and combined QKV
weights in addition to original weights. Do not infer memory savings from
the reconstruction-free attention path.

Final source/config hashes and raw repetitions are stored in result
directories. `summarize_direct_acceptance.py` collects the final runs without
overwriting earlier experiments.
