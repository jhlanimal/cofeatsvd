import torch
from fused_decode_ops import rms, rope
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


@torch.inference_mode()
def main():
    torch.cuda.set_device(3)
    torch.manual_seed(83)
    for d in (128, 2560):
        for rows in (1, 32, 128):
            x = torch.randn(rows, d, device='cuda:3', dtype=torch.bfloat16)
            w = torch.randn(d, device=x.device, dtype=x.dtype)
            expected = w * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
            actual = rms(x, w, 1e-6)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    q = torch.randn(1, 32, 1, 128, device='cuda:3', dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1, 128, device=q.device, dtype=q.dtype)
    angle = torch.randn(1, 1, 128, device=q.device)
    cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
    expected = apply_rotary_pos_emb(q, k, cos, sin)
    actual = rope(q, k, cos, sin)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    print('Fused RMSNorm and RoPE bitwise match BF16 references on tested inputs')


if __name__ == '__main__':
    main()
