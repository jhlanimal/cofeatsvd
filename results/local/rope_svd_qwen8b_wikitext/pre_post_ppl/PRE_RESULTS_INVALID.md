# Invalid Pre-RoPE Results

The pre64, pre80 and pre96 entries in pre_post_ppl.json are invalid.

The original parser used method[:4] and int(method[4:]), which mapped
pre64 -> (pre6, 4), pre80 -> (pre8, 0), pre96 -> (pre9, 6).
None of these modes selected the pre-RoPE branch. Rank zero retained all
columns because the slice [-0:] is the full slice.

The unused pre-RoPE branch also used an invalid page-centre rotation instead
of per-key-token rotations. Both issues are corrected in the independent
experiments/eval_qwen3_pre_post_svd_ppl.py script. New results are saved in
../pre_post_ppl_v2_corrected/ and rank-128 checks in ../pre_post_v2_sanity/.

Do not use the original pre64/pre80/pre96 entries in tables or figures.
