"""Full GPU memory breakdown (not just weights) for BF16 vs the real packed INT4 path.

Why this needs care: the INT4 path allocates its weights with cudaMallocAsync inside
nunchaku, which torch.cuda.memory_allocated()/max_memory_allocated() CANNOT see, while
the BF16 path's weights are ordinary torch tensors that those counters DO see. Naively
comparing peak_mem between the two therefore compares different things (it is what made
an early run look like a 7.6x saving). So:

  weights_total = mem_get_info delta after load        (sees everything)
  weights_torch = torch.cuda.memory_allocated()        (torch-visible part)
  weights_nunchaku = weights_total - weights_torch     (cudaMallocAsync part)

nunchaku's buffers are allocated once and persist (persistent cached scratch), so its
contribution is constant after warm-up. Transient per-step memory (activations + KV
cache) is all torch-side, so:

  peak_total = weights_total + (torch_max_allocated - weights_torch)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ activations + KV

Usage:
    python memory_breakdown.py --mode bf16
    python memory_breakdown.py --mode int4
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODEBASE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_NUNCHAKU_DIR = os.environ.get("NUNCHAKU_DIR", "")
for p in (_CODEBASE_ROOT, _HERE, _NUNCHAKU_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers

GB = 1e9


def analytic_breakdown(cfg, mode, group_size=64):
    """Parameter-count-based expectation, to separate 'what quantization can do' from
    'what the implementation actually costs'."""
    h, inter, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    kv = cfg.num_key_value_heads * (h // cfg.num_attention_heads)
    per_layer = (h * h) + (kv * h) * 2 + (h * h) + 3 * (inter * h)
    layers = per_layer * L
    embed = cfg.vocab_size * h
    lm_head = cfg.vocab_size * h
    if mode == "bf16":
        return {"quantizable layers": layers * 2 / GB,
                "embedding (BF16)": embed * 2 / GB,
                "lm_head (BF16)": lm_head * 2 / GB}
    return {"quantizable layers (INT4 packed)": layers * 0.5 / GB,
            "per-64-group scales (BF16)": (layers / group_size) * 2 / GB,
            "embedding (BF16, NOT quantized)": embed * 2 / GB,
            "lm_head (BF16, NOT quantized)": lm_head * 2 / GB}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bf16", "int4"], required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    cli = p.parse_args()

    dev = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.empty_cache()
    free0, total = torch.cuda.mem_get_info()
    baseline = (total - free0) / GB   # CUDA context etc., excluded from all deltas below

    from main import load_model, build_cfg, _load_all_hyperparams
    if cli.mode == "int4":
        os.environ["NUNCHAKU_FWD_MODE"] = "graph_beta"
        import full_model_nunchaku as fmn
        args = fmn._make_args(cli.model, 128, 2048)
        c = build_cfg(args, _load_all_hyperparams(args.method_cfg))
        c["_model_id"] = cli.model
        model = load_model(cli.model)
        model.config.use_cache = True
        fmn.convert_model_to_nunchaku(model, c)
        model = model.to(dev).eval()
        stream_ctx = torch.cuda.stream(fmn.GRAPH_STREAM)
    else:
        model = load_model(cli.model).to(dev).eval()
        model.config.use_cache = True
        import contextlib
        stream_ctx = contextlib.nullcontext()

    tok = transformers.AutoTokenizer.from_pretrained(cli.model, use_fast=False)
    torch.cuda.synchronize()

    # ---- reclaim retained-but-unused memory before measuring ----
    # Two separate caches hold freed blocks after model conversion:
    #  (1) torch's caching allocator (reserved >> allocated after 224 temporaries)
    #  (2) nunchaku's cudaMallocAsync pool (retains freed blocks by default)
    # Neither is model footprint, but both show up in mem_get_info. Reclaim both.
    free_pre, _ = torch.cuda.mem_get_info()
    pre_trim_total = (total - free_pre) / GB - baseline
    torch_reserved_pre = torch.cuda.memory_reserved() / GB

    torch.cuda.empty_cache()
    if cli.mode == "int4":
        import nunchaku_min
        nunchaku_min.trim_memory_pool(0)
    torch.cuda.synchronize()

    free1, _ = torch.cuda.mem_get_info()
    weights_total = (total - free1) / GB - baseline
    weights_torch = torch.cuda.memory_allocated() / GB
    weights_nunchaku = weights_total - weights_torch

    print(f"\n{'='*72}\n{cli.mode.upper()} — GPU memory breakdown ({cli.model})\n{'='*72}")
    print(f"CUDA context / baseline (excluded from deltas): {baseline:.3f} GB\n")

    print("Analytic expectation from parameter counts:")
    exp = analytic_breakdown(model.config, cli.mode)
    for k, v in exp.items():
        print(f"  {k:<42} {v:>7.3f} GB")
    print(f"  {'-- analytic subtotal':<42} {sum(exp.values()):>7.3f} GB")

    print("\nReclaiming retained caches after load:")
    print(f"  {'total before reclaim':<42} {pre_trim_total:>7.3f} GB")
    print(f"  {'  (torch reserved was)':<42} {torch_reserved_pre:>7.3f} GB")
    print(f"  {'total after empty_cache + pool trim':<42} {weights_total:>7.3f} GB")
    print(f"  {'  => reclaimed':<42} {pre_trim_total - weights_total:>7.3f} GB")

    print("\nMeasured, resident after load (no forward pass yet):")
    print(f"  {'torch-visible (memory_allocated)':<42} {weights_torch:>7.3f} GB")
    # For BF16 this residual is only torch allocator reserve + cuBLAS workspace;
    # for INT4 it is dominated by nunchaku's cudaMallocAsync weights.
    label = ("not torch-visible (nunchaku weights + reserve)" if cli.mode == "int4"
             else "not torch-visible (alloc reserve, cuBLAS ws)")
    print(f"  {label:<42} {weights_nunchaku:>7.3f} GB")
    print(f"  {'== TOTAL WEIGHTS (mem_get_info)':<42} {weights_total:>7.3f} GB")
    print(f"  {'implementation overhead vs analytic':<42} "
          f"{weights_total - sum(exp.values()):>7.3f} GB")

    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)

    # Analytic KV-cache size, so it can be separated from other activations.
    c = model.config
    head_dim = c.hidden_size // c.num_attention_heads
    kv_per_tok_per_layer = 2 * c.num_key_value_heads * head_dim * 2  # K and V, BF16
    kv_bytes = lambda bs, toks: bs * toks * kv_per_tok_per_layer * c.num_hidden_layers / GB

    NEW = 100
    prompt_len = ids.shape[1]
    rows = []
    with stream_ctx, torch.no_grad():
        for bs in (1, 16):
            ids_b = ids.repeat(bs, 1) if bs > 1 else ids
            model(ids_b, use_cache=False)  # warm up (also allocates nunchaku act scratch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            model(ids_b, use_cache=False)
            torch.cuda.synchronize()
            transient = torch.cuda.max_memory_allocated() / GB - weights_torch
            rows.append((f"prefill batch={bs} (seq={prompt_len}, no KV)", bs, transient, 0.0))

        for bs in (1, 16):
            ids_b = ids.repeat(bs, 1) if bs > 1 else ids
            am = torch.ones_like(ids_b)
            gen = dict(attention_mask=am, max_new_tokens=NEW, do_sample=False,
                       pad_token_id=tok.eos_token_id)
            model.generate(ids_b, **gen)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            model.generate(ids_b, **gen)
            torch.cuda.synchronize()
            transient = torch.cuda.max_memory_allocated() / GB - weights_torch
            rows.append((f"decode  batch={bs} (+{NEW} tok)", bs,
                         transient, kv_bytes(bs, prompt_len + NEW)))

    # Persistent footprint AFTER the forwards: now includes nunchaku's INT4
    # activation scratch, which did not exist at load time. The earlier "total"
    # column was undercounting exactly this.
    torch.cuda.empty_cache()
    if cli.mode == "int4":
        import nunchaku_min
        nunchaku_min.trim_memory_pool(0)
    torch.cuda.synchronize()
    free2, _ = torch.cuda.mem_get_info()
    persistent = (total - free2) / GB - baseline
    act_scratch = persistent - weights_total

    scratch_label = ("INT4 act operands + scales + workspaces" if cli.mode == "int4"
                     else "cuBLAS/cuDNN workspaces")
    print(f"\nPersistent after first forward: {persistent:.3f} GB "
          f"(= weights {weights_total:.3f} + first-forward scratch {act_scratch:.3f}"
          f" [{scratch_label}])")

    print("\nFull memory accounting per workload:")
    print(f"  {'workload':<34} {'weights':>8} {'actscr':>8} {'KVcache':>8} {'otheract':>9} {'TOTAL':>8}")
    for name, bs, transient, kv in rows:
        other_act = transient - kv
        tot_w = weights_total + act_scratch + transient
        print(f"  {name:<34} {weights_total:>7.3f}  {act_scratch:>7.3f}  "
              f"{kv:>7.3f}  {other_act:>8.3f}  {tot_w:>7.3f}")
    print("  (weights incl. embedding + lm_head, both left in BF16; actscr = persistent")
    print("   INT4 packed activation operands + their scales; KVcache is BF16, analytic;")
    print("   otheract = BF16 residual stream / logits / temporaries, measured)")

    # ---- activation memory, split by precision ----
    # The GEMM's activation OPERAND is real INT4 (verified: 4.00 bits/element,
    # 16 levels, per-64-channel scales). What stays BF16 is the residual stream
    # between layers and the KV cache. Both are reported separately here so the
    # activation story is not overstated in either direction.
    if cli.mode == "int4":
        act_b = asc_b = 0
        n_layers = 0
        for m in model.modules():
            if hasattr(m, "gemm") and hasattr(m.gemm, "act_buffer_bytes"):
                a, s = m.gemm.act_buffer_bytes()
                act_b += a; asc_b += s; n_layers += 1
        print(f"\nActivation memory by precision ({n_layers} quantized linears):")
        print(f"  {'INT4 packed activation operands (4 bit/elem)':<46} {act_b/GB:>7.3f} GB")
        print(f"  {'their per-64-group scales (BF16)':<46} {asc_b/GB:>7.3f} GB")
        print(f"  {'BF16 KV cache (decode b1 / b16)':<46} "
              f"{rows[2][3]:>7.3f} / {rows[3][3]:.3f} GB")
        print(f"  {'BF16 other activations (decode b1 / b16)':<46} "
              f"{rows[2][2]-rows[2][3]:>7.3f} / {rows[3][2]-rows[3][3]:.3f} GB")
        print("  note: INT4 activation buffers are persistent per-layer scratch and are")
        print("        already included in the TOTAL WEIGHTS figure above (nunchaku side).")


if __name__ == "__main__":
    main()
