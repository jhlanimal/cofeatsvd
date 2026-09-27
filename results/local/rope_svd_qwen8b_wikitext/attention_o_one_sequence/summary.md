# Attention output comparison: one Full sequence

Qwen3-8B; the same first WikiText-2 validation 4096-token window as the
whole-sequence PPL audit (input SHA256 matched). Layers 9,18,27 are zero-based.
Every query head and every token is included. O is before o_proj and residual
addition. All variants consume identical Q/K/V captured from one unmodified
Full base-transformer forward. No generation or PPL evaluation was performed.

Pre: whole-sequence FP32 SVD of normalized pre-RoPE K, with exact original
per-key-position rotation transferred to Q; no dense K reconstruction.
Post: whole-sequence FP32 SVD of post-RoPE K, direct factor attention.
Ours: existing weighted_coalescing_pages_batched, 32-token pages, actual BF16
representatives/coefficients and uint8 owners; active page remains dense.
The existing builder uses the leading eigenvector of each group's weighted
token covariance; no new substitute grouping or global SVD is inserted.
All score, softmax, O and metric reductions are FP32 for this diagnostic.

## Mean token/head cosine similarity (higher is better)

| Layer | Rank/m | Pre | Post | Ours |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 64 | 0.750168 | 0.694947 | 0.993679 |
| 18 | 64 | 0.942136 | 0.914408 | 0.996591 |
| 27 | 64 | 0.727136 | 0.693972 | 0.994967 |
| Mean | 64 | 0.806480 | 0.767776 | 0.995079 |
| 9 | 80 | 0.861355 | 0.729504 | 0.996880 |
| 18 | 80 | 0.958363 | 0.927412 | 0.998234 |
| 27 | 80 | 0.782337 | 0.707848 | 0.998106 |
| Mean | 80 | 0.867352 | 0.788254 | 0.997740 |
| 9 | 96 | 0.908539 | 0.804225 | 0.998598 |
| 18 | 96 | 0.977334 | 0.938468 | 0.999165 |
| 27 | 96 | 0.814917 | 0.731133 | 0.999335 |
| Mean | 96 | 0.900263 | 0.824609 | 0.999033 |

## Relative squared output error (lower is better)

For each layer compute ||O_method-O_full||_F^2 / ||O_full||_F^2, then report
the unweighted mean over three layers. This is not a pooled energy ratio.

| Rank/m | Pre | Post | Ours |
| ---: | ---: | ---: | ---: |
| 64 | 0.888279 | 0.995304 | 0.012063 |
| 80 | 0.716245 | 0.954050 | 0.005137 |
| 96 | 0.587413 | 0.866493 | 0.002100 |

Layer-level errors, cosine medians/p05 and timings are in metrics.csv.
Per-head/token cosines and squared-error/energy arrays are in NPZ files;
the original captured tensors and token IDs are in full_captures.pt.

## Sanity checks and limitations

- FP32 Full O versus captured native SDPA O: cosine >= 0.99999868 and relative
  squared error <= 2.98e-6 in all three layers (BF16 rounding differences).
- Post rank128: cosine 1.0; relative squared error <= 1.06e-9.
- Pre rank128: cosine >= 0.9999977; relative squared error <= 6.93e-6.
  The small difference reflects exact FP32 factor rotation versus BF16 native K rotation.
- Random factor-score tests versus reconstructed reference: Pre maximum
  absolute error 7.63e-6; Ours maximum absolute error 4.77e-6.
- Ours is closer to Full O in all nine tested layer/budget combinations, on
  both mean cosine and relative squared error. This is a single-sequence
  descriptive finding, not a multi-sequence statistical claim.
- Same numeric rank/m does NOT mean matched storage. Global SVD has one
  basis per head, whereas Ours has per-page metadata and active dense K.
- Global SVD uses future-token statistics. This is an offline local-output
  diagnostic, not an online KV-cache benchmark.
- Paging, grouping and active-page handling all differ here. These results
  do not isolate grouping's contribution from paging's contribution.
- No latent-coordinate ordinary-RoPE variant is included.
- Analysis ran on physical A6000 GPU3 by UUID, sharing it with the continuing
  Full 32B run. Timings are not isolated performance measurements.
