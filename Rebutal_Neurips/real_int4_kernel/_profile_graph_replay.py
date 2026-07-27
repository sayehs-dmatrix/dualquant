import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, transformers
from transformers import StaticCache
import whole_step_graph_decode as W

dev = torch.device("cuda:0")
mid = "meta-llama/Llama-3.1-8B"
mode = sys.argv[1] if len(sys.argv) > 1 else "int4"

model = W.build_int4_model(mid, dev) if mode == "int4" else W.build_bf16_model(mid, dev)
tok = transformers.AutoTokenizer.from_pretrained(mid, use_fast=False)
ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids.to(dev)

_, _, g, s_in, s_pos, s_lg = W.graph_decode(model, ids, 4, dev, ids.shape[1] + 300, capture=True)

pos = ids.shape[1] + 4
s_pos.fill_(pos)
for _ in range(20):
    g.replay(); pos += 1; s_pos.fill_(pos)
torch.cuda.synchronize()

N = 30
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(N):
        g.replay(); pos += 1; s_pos.fill_(pos)
    torch.cuda.synchronize()

ka = prof.key_averages()
total_cuda = sum(e.self_device_time_total for e in ka)
print(f"\n=== {mode}: per-replay GPU time = {total_cuda/N/1000:.3f} ms  ({N} replays) ===")
rows = sorted([e for e in ka if e.self_device_time_total > 0],
              key=lambda e: -e.self_device_time_total)
print(f"{'kernel':<62} {'ms/step':>9} {'%':>6} {'calls/step':>11}")
for e in rows[:14]:
    print(f"{e.key[:60]:<62} {e.self_device_time_total/N/1000:>9.3f} "
          f"{100*e.self_device_time_total/total_cuda:>6.1f} {e.count/N:>11.1f}")
