# One-sequence whole-K SVD audit

Qwen3-8B, WikiText-2 validation, first 4096 tokens, 4095 scored targets.
All 36 layers and all 8 raw KV heads, BF16 model, FP32 SVD and attention
scores. Each decomposition sees the full 4096 x 128 Key matrix. No page-local
factorizations or dense-current-page exemption. See config.json for provenance.

IMPORTANT: These are offline diagnostic PPL values. Whole-sequence SVD uses
future-token statistics even though the attention mask is causal. They are
not strictly causal online PPL and cannot be mixed with the old Ours table.

| Method | PPL | Relative to matched Full |
| --- | ---: | ---: |
| Native SDPA | 6.214305 | +0.0651% |
| Matched custom Full | 6.210261 | 0% |
| Pre exact RoPE, rank 64 | 8.873879 | +42.8906% |
| Pre exact RoPE, rank 80 | 8.283218 | +33.3796% |
| Pre exact RoPE, rank 96 | 7.722558 | +24.3516% |
| Post rank 64 | 22.318593 | +259.3826% |
| Post rank 80 | 13.590731 | +118.8432% |
| Post rank 96 | 9.799431 | +57.7942% |
| Pre exact RoPE, rank 128 | 6.214353 | +0.0659% |
| Post rank 128 | 6.210865 | +0.0097% |

Naive low-dimensional RoPE on SVD coordinates (explicit theta^(-2j/r)
frequency convention, original 1/sqrt(128) attention scale):

| Rank | PPL |
| ---: | ---: |
| 64 | 20287.683650 |
| 80 | 7944.754136 |
| 96 | 12432.001113 |
| 128 | 15385.841896 |

The naive latent operator changes attention even at full rank: the SVD basis
does not generally commute with RoPE. Its failure is therefore not evidence
that every pre-RoPE compression method loses quality this severely. The exact
Pre implementation preserves the native position operator by moving each
key token's inverse rotation onto the query; it avoids dense K reconstruction
but is not a fast ordinary low-dimensional dot product.

Earlier 13-block results factorized each 128-token page independently, kept
the current page dense, and used exact per-key-token rotations. They measured
a different experiment. A scope warning was added without deleting those data.

Independent score tests against reconstructed reference K passed for ranks
64/80/96/128, maximum absolute Pre score discrepancy <= 1.91e-5 on test inputs.
Whole-model native/custom hidden states are not bit-identical (observed maximum
absolute difference 23.625); attention precision/order differs and errors
propagate through layers. The native/custom Full PPL gap is 0.0651%, and the
matched full-rank controls above recover the baseline closely. Do not describe
the custom path as bit-identical to native SDPA.
