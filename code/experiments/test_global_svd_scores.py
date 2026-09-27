"""Independent score tests for the whole-sequence diagnostic."""
import torch
from eval_qwen3_pre_post_svd_ppl import rotate, pre_scores


def main():
    torch.manual_seed(42)
    n, d = 257, 128
    k, q = torch.randn(n, d), torch.randn(7, d)
    inv = 1000000. ** (-torch.arange(0, d, 2).float()/d)
    phase = torch.arange(n).float()[:, None] * inv
    phase = torch.cat((phase, phase), -1)
    cos, sin = phase.cos(), phase.sin()
    for rank in (64, 80, 96, 128):
        u, s, vh = torch.linalg.svd(k, full_matrices=False)
        left, right = u[:, :rank]*s[:rank], vh[:rank]
        expected = q @ rotate(left @ right, cos, sin).T
        actual = pre_scores(q, left, right, cos, sin)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print(rank, 'exact-RoPE factor max score error', float((actual-expected).abs().max()))
        kr = rotate(k, cos, sin)
        u, s, vh = torch.linalg.svd(kr, full_matrices=False)
        left, right = u[:, :rank]*s[:rank], vh[:rank]
        torch.testing.assert_close((q @ right.T) @ left.T, q @ (left @ right).T,
                                   rtol=1e-4, atol=1e-4)
    print('All whole-sequence factor score checks passed')


if __name__ == '__main__':
    main()
