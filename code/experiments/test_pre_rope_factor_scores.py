"""Regression checks for method parsing and exact pre-RoPE factor scores."""
import torch

from eval_qwen3_pre_post_svd_ppl import parse_method, pre_scores, rotate, svd_factor


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    for mode in ("pre", "post", "ours"):
        for rank in (64, 80, 96):
            assert parse_method(f"{mode}{rank}") == (mode, rank)
    for name in ("pre0", "pre", "pre-1", "pre64extra"):
        try:
            parse_method(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid method accepted: {name}")
    k = torch.randn(128, 128)
    q = torch.randn(7, 128)
    angles = torch.randn(128, 64)
    cos = torch.cat([angles.cos()] * 2, dim=-1)
    sin = torch.cat([angles.sin()] * 2, dim=-1)
    for rank in (64, 80, 96, 128):
        left, right = svd_factor(k, rank)
        assert left.shape == (128, rank)
        assert right.shape == (rank, 128)
        direct = pre_scores(q, left, right, cos, sin)
        # Reconstruction is a test oracle only, never part of the PPL evaluator.
        expected = q @ rotate(left @ right, cos, sin).T
        torch.testing.assert_close(direct, expected, atol=5e-5, rtol=1e-4)
        print(f"rank={rank}: max_abs_score_error={(direct-expected).abs().max().item():.8g}")
    try:
        svd_factor(k, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("rank zero must fail")
    print("All regression checks passed")


if __name__ == "__main__":
    main()
