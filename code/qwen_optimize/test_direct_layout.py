"""Regression for head-major V workspaces and direct compact scores."""
import argparse
import torch
from triton_page_parallel_attention_v2 import compact_page_attention_v2


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:2')
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(41)
    for gqa in (2, 4, 8):
     for n in (1, 4, 17):
        for m in (64, 80, 96):
            for active in (1, 31, 32):
                q = torch.randn(8*gqa, 128, device=args.device, dtype=torch.bfloat16)
                c = torch.randn(n, 8, 32, m, device=args.device) / 5
                coeff = torch.randn(n, 8, 128, device=args.device)
                owner = torch.randint(m, (n, 8, 128), device=args.device)
                v = torch.randn(8, 2176, 128, device=args.device, dtype=torch.bfloat16)
                ak = torch.randn(8, 32, 128, device=args.device, dtype=torch.bfloat16)[:, :active]
                hv = v[:, :n * 32].view(8, n, 32, 128).permute(1, 0, 2, 3)
                av = v[:, n * 32:n * 32 + active]
                actual = compact_page_attention_v2(q, c, coeff, owner, hv, ak, av, 128**-0.5, num_warps=2)
                # Dense reconstruction is used only by this independent oracle.
                k = c.gather(-1, owner[:, :, None].expand(-1, -1, 32, -1)) * coeff[:, :, None]
                k = k.permute(1, 0, 2, 3).reshape(8, n * 32, 128)
                k = torch.cat((k, ak.float()), 1).repeat_interleave(gqa, 0)
                scores = torch.einsum('hd,htd->ht', q.float(), k) * 128**-0.5
                expected = torch.einsum('ht,htd->hd', scores.softmax(-1), v[:, :n * 32 + active].float().repeat_interleave(gqa, 0))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    print('81 direct attention cases passed; GQA 2/4/8, strided V, all m and page boundaries')


if __name__ == '__main__':
    main()
