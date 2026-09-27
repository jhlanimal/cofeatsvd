import torch
from benchmark_autoregressive_full_vs_ours import weighted_coalescing_pages_batched, weighted_coalescing_page


@torch.inference_mode()
def main():
    torch.cuda.set_device(3)
    torch.manual_seed(128)
    k = torch.randn(8, 32, 128, device='cuda:3', dtype=torch.bfloat16)
    q = torch.randn(8, 32, 4, 128, device='cuda:3', dtype=torch.bfloat16)
    for m in (64, 80, 96):
        bc, bf, bo = weighted_coalescing_pages_batched(k, q, m)
        for head in range(8):
            c, f, o = weighted_coalescing_page(k[head], q[head], m, return_compact=True)
            torch.testing.assert_close(bo[head], o, atol=0, rtol=0)
            expected = c[:, o] * f[None]
            actual = bc[head, :, bo[head]] * bf[head][None]
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    print('Batched page builder: owners identical and effective K within 1e-4 for 24 pages')


if __name__ == '__main__':
    main()
