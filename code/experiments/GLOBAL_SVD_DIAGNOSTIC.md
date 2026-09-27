# Whole-sequence SVD diagnostic

Run `eval_global_svd_one_sequence.py --output-dir <new-directory>` in the
qwen-opt environment with `CUDA_VISIBLE_DEVICES=0`. The defaults evaluate
only the first 4096-token WikiText-2 validation window, using the same token
loader as the earlier page-local experiment. No RULER processes are touched.

For every layer and raw KV head, factorize the entire 4096 x 128 K matrix
with FP32 `torch.linalg.svd`. Keep the first r singular components (not the
first r original feature dimensions). There are 36 x 8 = 288 factorizations
per compressed model evaluation. No current-token or current-page exemption
is applied. Query chunking is only a memory optimization, not page-local SVD.

## Three distinct operations

Write the truncated pre-RoPE factorization as K_hat = L B, where L = U_r S_r
and B = Vh_r. Row vectors are used below.

1. **Post**: factorize K after its native rotation; scores are
   `(Q_rot @ B.T) @ L.T`. This is an ordinary low-dimensional dot product.
2. **Pre exact**: keep the original per-key-position rotation. For each key
   position t, rotate the query inversely using the original 128-dimensional
   frequencies, multiply by B.T, then contract with L[t]. This reproduces
   attention against rotated K_hat without materializing a dense K_hat.
   It is much more expensive than an ordinary low-dimensional dot product.
3. **Pre latent**: project Q to Q_pre @ B.T and use L as low-dimensional K;
   apply ordinary rank-dimensional RoPE with theta^(-2j/r) to both. This is
   an explicit naive diagnostic, NOT an equivalent implementation of the
   original positional operator or a claim about all pre-RoPE methods.
   Scale remains 1/sqrt(128) to isolate the positional-operation change.

Even at r=128, the SVD basis generally does not commute with the original
RoPE operator. Applying the ordinary RoPE frequency pairs to SVD coordinates
therefore changes attention even without rank truncation. The r=128 control
is essential; large Pre-latent PPL cannot be attributed solely to compression.

All variants retain V and apply a causal attention mask. However, the global
SVD basis is fitted to the entire sequence, so future tokens affect the basis.
The resulting numbers are OFFLINE diagnostic pseudo-perplexities, not a
strictly causal online KV-cache evaluation. They must not be merged into the
earlier 13-window/page-local Ours table.

Native SDPA and the custom FP32 score/softmax path are both evaluated. The
custom Full is the matched baseline for the custom SVD paths. Small BF16
rounding differences can accumulate through all layers. Exact Pre rotates
the approximated factors in FP32, whereas native K RoPE is evaluated in BF16.

The separate test `test_global_svd_scores.py` compares factor-only scores to
explicit reconstructed-K reference scores on random inputs for ranks
64/80/96/128. Reconstruction is used only by this correctness test.
