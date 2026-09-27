import argparse
from pathlib import Path
import torch
from benchmark_autoregressive_full_vs_ours import load_runner, load_tokens, DEFAULT_DATA, AutoTokenizer


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    model_path = '/data/share/Qwen3-4B-Instruct-2507'
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = load_tokens(tok, DEFAULT_DATA, 256).to(args.device)
    runner = load_runner(model_path, 'ours_direct_v2', 80, args.device)
    runner.prefill(ids[:128], runner.position_cache[:128])
    runner.seal_prefill()
    for pos in range(128, 160):
        runner.run_token(ids[pos], pos)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for pos in range(160, 192):
            with torch.profiler.record_function('decode_token'):
                runner.run_token(ids[pos], pos)
        torch.cuda.synchronize()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
    (out/'cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))
    prof.export_chrome_trace(str(out/'trace.json'))
    print((out/'cuda.txt').read_text())


if __name__ == '__main__':
    main()
