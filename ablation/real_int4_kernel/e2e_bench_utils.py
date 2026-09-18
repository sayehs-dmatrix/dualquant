"""Shared end-to-end benchmarking helpers used by both bf16_baseline_e2e.py
and full_model_w4a4.py, so the two report directly comparable numbers:
  - prefill_bench: single forward pass over a (batch, seq_len) prompt,
    no KV-cache growth -- compute/bandwidth-bound, comparable to the
    paper's operator-level microbenchmarks at batch>1.
  - decode_bench: autoregressive generate() with KV-cache, batch=1 --
    the realistic serving path, dominated by per-step kernel-launch
    overhead at M=1.
"""
import time

import torch


def prefill_bench(model, ids, batch_size=1, warmup=3, repeats=10):
    ids_b = ids.repeat(batch_size, 1) if batch_size > 1 else ids
    with torch.no_grad():
        for _ in range(warmup):
            model(ids_b, use_cache=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(repeats):
            model(ids_b, use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    latency_ms = elapsed / repeats * 1000
    tokens_per_sec = batch_size * ids_b.shape[1] / (elapsed / repeats)
    return dict(batch_size=batch_size, seq_len=ids_b.shape[1],
                 latency_ms=latency_ms, tokens_per_sec=tokens_per_sec,
                 peak_mem_gb=peak_mem_gb)


def decode_bench(model, ids, tokenizer, max_new_tokens=100, warmup_calls=1, repeats=3, batch_size=1):
    # batch_size>1 repeats the SAME prompt across the batch dim (like
    # prefill_bench) -- every row decodes identically under greedy decoding,
    # but this exercises the real HF generate()/KV-cache plumbing at M=batch_size
    # per decode step, not just a synthetic shape. attention_mask is passed
    # explicitly (all-ones -- no padding, since every row is the same length)
    # to avoid HF's implicit-mask warning and guarantee correct batched decode.
    ids_b = ids.repeat(batch_size, 1) if batch_size > 1 else ids
    attention_mask = torch.ones_like(ids_b)
    with torch.no_grad():
        for _ in range(warmup_calls):
            model.generate(ids_b, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n_tokens = 0
        for _ in range(repeats):
            out = model.generate(ids_b, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
            n_tokens += (out.shape[1] - ids_b.shape[1]) * batch_size
        torch.cuda.synchronize()
        elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return dict(tokens_per_sec=n_tokens / elapsed, latency_per_call_s=elapsed / repeats,
                 peak_mem_gb=peak_mem_gb, batch_size=batch_size)
