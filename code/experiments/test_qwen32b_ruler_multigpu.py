#!/usr/bin/env python3
"""Run in the runner directory with at least two visible GPUs."""
import copy
import torch
from accelerate import dispatch_model
from transformers import Qwen3Config, Qwen3ForCausalLM
import benchmark_autoregressive_lowmem as bench
from run_qwen32b_ruler_multigpu import (
    runner_class, install_device_contexts, install_chunked_mlp, install_gqa_adapter,
)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    install_gqa_adapter(bench)
    config = Qwen3Config(vocab_size=128, hidden_size=256, intermediate_size=512,
                        num_hidden_layers=2, num_attention_heads=64,
                        num_key_value_heads=8, head_dim=128, sliding_window=None)
    config._attn_implementation = 'sdpa'
    native = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    split = copy.deepcopy(native)
    native.to('cuda:0')
    split = dispatch_model(split, device_map={
        'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.layers.0': 0,
        'model.layers.1': 1, 'model.norm': 1, 'lm_head': 1,
    })
    runners = []
    for model, cls in [(native, bench.Runner), (split, runner_class(bench))]:
        bench.install_rms(model)
        runner = cls(model, 'ours_direct_v2', 64, max_tokens=160, logits_to_keep=1)
        install_device_contexts(model)
        install_chunked_mlp(model, 32)
        runners.append(runner)
    ids = torch.randint(0, 128, (100,), device='cuda:0')
    results = []
    for runner in runners:
        runner.prepare_capacity(128)
        values = [runner.prefill(ids[:64], torch.arange(64, device='cuda:0')).logits.cpu()]
        runner.seal_prefill()
        for pos in range(64, 100):
            values.append(runner.run_token(ids[pos], pos).cpu())
        results.append(torch.cat(values, dim=1))
    print('SHARD_TEST_MAX_ERROR', (results[0].float() - results[1].float()).abs().max().item())
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    print('SHARD_TEST_PASS: prefill + 36 decode tokens including page seal')

    q = torch.randn(64, 128, device='cuda:0', dtype=torch.bfloat16)
    c = torch.randn(2, 8, 32, 64, device='cuda:0', dtype=torch.bfloat16)
    coeff = torch.randn(2, 8, 128, device='cuda:0', dtype=torch.bfloat16)
    owner = torch.randint(0, 64, (2, 8, 128), device='cuda:0', dtype=torch.uint8)
    v = torch.randn(2, 8, 32, 128, device='cuda:0', dtype=torch.bfloat16)
    ak = torch.randn(8, 3, 128, device='cuda:0', dtype=torch.bfloat16)
    av = torch.randn_like(ak)
    got = bench.compact_page_attention_v2(q, c, coeff, owner, v, ak, av, 128**-.5)
    # Reconstruction is used only as a correctness oracle, never in evaluation.
    k = (c.float().gather(-1, owner.long()[:, :, None, :].expand(-1, -1, 32, -1))
         * coeff.float()[:, :, None, :]).permute(1, 0, 2, 3).reshape(8, 64, 128)
    k = torch.cat((k, ak.float()), 1)
    val = torch.cat((v.float().permute(1, 0, 2, 3).reshape(8, 64, 128), av.float()), 1)
    scores = torch.einsum('hgd,htd->hgt', q.float().reshape(8, 8, 128), k) * 128**-.5
    expected = torch.einsum('hgt,htd->hgd', scores.softmax(-1), val).reshape(64, 128)
    print('GQA_TEST_MAX_ERROR', (got - expected).abs().max().item())
    torch.testing.assert_close(got, expected, rtol=.001, atol=.001)
    print('GQA_TEST_PASS')


if __name__ == '__main__':
    main()
