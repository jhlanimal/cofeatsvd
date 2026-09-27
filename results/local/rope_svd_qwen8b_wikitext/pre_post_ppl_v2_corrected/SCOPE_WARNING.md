# Scope of these results

These 13-block results are PAGE-LOCAL experiments, not whole-sequence SVD.

- Each 128-token page is factorized independently for each layer and KV head.
- The current page attends with dense K; only previous pages use SVD factors.
- Pre-RoPE scores retain the original 128-dimensional per-key-token rotation by
  applying its inverse to the query. They do not apply ordinary RoPE directly
  to rank-dimensional SVD coordinates.
- Consequently, these results cannot represent whole-sequence SVD or naive
  reduced-dimensional RoPE quality. The small PPL changes do not establish
  that ordinary RoPE commutes with projection onto an SVD basis.

The independent one-sequence audit is in ../global_one_sequence/. Its global
SVD basis is fitted using all sequence tokens, including future tokens; it is
an offline diagnostic and must not be reported as strictly online causal PPL.
