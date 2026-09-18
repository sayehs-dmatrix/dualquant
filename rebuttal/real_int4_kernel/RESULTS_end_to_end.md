# DualQuant W4A4 — real end-to-end inference results (Llama-3.1-8B)

Addresses the reviewer objection that accuracy was measured via fake quantization while
efficiency came only from operator-level microbenchmarks. Every number here is end-to-end
through a **real packed INT4 inference path** (MIT nunchaku / SVDQuant CUTLASS `GEMM_W4A4`),
not simulated quantization.

## Setup

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B` |
| GPU | NVIDIA RTX 4090 (24 GB, ~1008 GB/s peak HBM bandwidth) |
| Baseline | same model, BF16, cuBLAS |
| INT4 path | DualQuant BCD (`mat_q` + per-channel `beta`) → nunchaku packed W4A4 CUTLASS GEMM |
| Quantized | all 7 projections × 32 layers (q,k,v,o,gate,up,down) = 224 linear layers |
| Not quantized | embedding, `lm_head`, norms (standard practice) |
| Prefill workload | **82-token prompt**, single forward, `use_cache=False` |
| Decode workload | same 82-token prompt, then **100 generated tokens**, greedy, KV-cache |
| Accuracy workload | WikiText-2 test, non-overlapping **8192**-token windows (35 windows) |

The **8192 length applies only to the perplexity run** in §1. Every latency, throughput and
memory number in §2–§4 uses the 82-token prompt and, where decoding, 100 generated tokens
(prompt + generated = 182 cached positions). Throughput is
`batch x tokens / latency`: e.g. prefill batch=16 is 16 x 82 / 146.75 ms = 8940 tok/s;
decode is one token per sequence per replayed step.

### What is ours vs. third-party

The W4A4 GEMM is nunchaku's CUTLASS kernel, not ours. On top of it we implemented:

- a CUDA kernel applying DualQuant's per-channel `beta` **inside** nunchaku's
  activation-quantization kernel, during the global-memory load, so the fused scaling costs
  no separate kernel launch (`fused_beta_mul.cu`, plus a `beta` path threaded through
  `load_act_to_fpsum`);
- `QuantizedGEMM::forward_static`, a capture-safe entry point that performs no allocation and
  inherits the caller's stream, which is what makes whole-model CUDA-graph capture possible;
- a `nocache` path that allocates transiently, needed for long-context (8192) accuracy runs;
- the benchmark and profiling harnesses.

### The path is genuinely W4A4

Verified rather than assumed: the activation operand handed to the GEMM is **exactly
4.00 bits/element** (packed `[M, K/2]`, 16 levels spanning [-8, 7], one BF16 scale per 64
channels), and the emitted SASS is `IMMA.16864.S4.S4` — the INT4 tensor-core instruction.
Both operands are INT4. Per-layer SQNR against FP32 on a real DualQuant layer is 19.0 dB.

### Measurement protocol

Batch-1 inference in an eager PyTorch loop is CPU-dispatch-bound (~1365 kernel launches per
token against ~8 ms of GPU work), which measures framework overhead rather than
quantization. All throughput numbers below therefore capture the **entire** forward step
(embedding, all 32 blocks including attention and norms, LM head) into a single CUDA graph
and replay it, with the **identical treatment applied to the BF16 baseline**, so the
comparison isolates quantization.

Every configuration is correctness-gated before being timed: prefill logits are bit-exact
against the uncaptured run (max |diff| = 0) and decode reproduces identical generated
tokens. All results are warm runs, stable to <0.2% across repetitions. Memory uses
`torch.cuda.mem_get_info()` (whole-GPU accounting) — **not** `memory_allocated()`, which
cannot see nunchaku's `cudaMallocAsync` buffers and under-reports the INT4 footprint ~3.5x.

---

## 1. Accuracy through the real packed path

WikiText-2 PPL at sequence length 8192, matching the paper's reported setting, both rows
measured under identical settings:

| Llama-3.1-8B | Path | WikiText-2 PPL |
|---|---|---|
| Baseline | BF16 | **5.61** |
| DualQuant W4A4 | real packed INT4 kernel | **6.74** |

The packed path reproduces the paper's fake-quantization result to within 0.01 PPL (6.74 vs
6.73 reported), and the BF16 baseline reproduces the paper's reference (5.61 vs 5.62). The
fake-quantization accuracy in the paper is therefore an accurate proxy for the deployed
low-precision path, not an optimistic simulation of it.

---

## 2. Memory

### 2.1 Component breakdown (GB)

| Component | BF16 | DualQuant W4A4 |
|---|---|---|
| Quantized layer weights (6.98 B params) | 13.959 | **3.490** (exactly 4.00x) |
| Per-64-group weight scales (BF16) | — | 0.218 |
| Embedding (BF16, not quantized) | 1.051 | 1.051 |
| `lm_head` (BF16, not quantized) | 1.051 | 1.051 |
| Runtime/allocator overhead | 0.136 | 0.107 |
| **Weights subtotal** | **16.196** | **5.916** (2.74x) |
| First-forward scratch | 0.103 (cuBLAS ws) | 0.300 (INT4 act operands + scales + ws) |
| KV cache (BF16) — **prefill** batch 1 / 16 | 0 / 0 | 0 / 0 |
| KV cache (BF16) — **decode** batch 1 / 16 | 0.024 / 0.382 | 0.024 / 0.382 |
| Other activations (BF16 residual, logits) — **prefill** batch 1 / 16 | 0.030 / 0.356 | 0.030 / 0.356 |
| Other activations (BF16 residual, logits) — **decode** batch 1 / 16 | 0.012 / 0.065 | 0.012 / 0.065 |

INT4 packed activation operands are 0.080 GB of that scratch across all 224 layers. The KV
cache and residual stream stay BF16 in both paths, so they are identical and do not shrink.

The two activation rows are listed per workload because they genuinely differ. Prefill runs
with `use_cache=False`, so it holds **no** KV cache, and its activation term is dominated by
the full-sequence logits tensor (`[batch x 82 x 128256]` BF16 = 0.021 / 0.337 GB) rather
than the single-position `[batch x 1 x 128256]` of a decode step. Each total in §2.2 is
`weights subtotal + first-forward scratch + that workload's KV + its other activations`.
That is why prefill batch=1 sits 0.006 GB *below* decode batch=1 — it drops 0.024 GB of KV
and adds 0.018 GB of logits — and 0.091 GB below at batch=16 (−0.382 + 0.291).

### 2.2 End-to-end memory (GB)

| Workload | BF16 | DualQuant W4A4 | Reduction |
|---|---|---|---|
| Prefill batch=1 | 16.329 | **6.246** | **2.61x** |
| Prefill batch=16 | 16.655 | **6.572** | **2.53x** |
| Decode batch=1 | 16.335 | **6.252** | **2.61x** |
| Decode batch=16 | 16.746 | **6.663** | **2.51x** |

---

## 3. Latency and throughput

### 3.1 Component breakdown of GPU time (ms), **BF16 → W4A4**

| Component | Prefill b1 | Prefill b16 | Decode b1 | Decode b16 |
|---|---|---|---|---|
| Quantized linears (224) | 17.330 → **8.818** | 121.490 → **32.981** | 14.747 → **8.235** | 16.315 → **9.180** |
| Activation quantization (W4A4 only) | — → 0.823 | — → 3.089 | — → 0.702 | — → 0.716 |
| `lm_head` (BF16 in both) | 1.189 → 1.189 | 9.257 → 9.257 | 1.073 → 1.073 | 1.103 → 1.103 |
| Attention | 0.191 → 0.202 | 0.930 → 0.939 | 0.424 → 0.422 | 1.730 → 1.735 |
| Norms / RoPE / KV-cache / residual | 2.430 → 2.332 | 14.029 → 13.374 | 1.602 → 1.636 | 4.178 → 4.139 |
| **Total GPU** | **21.140 → 13.363** | **145.705 → 59.640** | **17.847 → 12.067** | **23.325 → 16.873** |

Each column sums to that workload's measured total. Profiler totals run ~1% above the
wall-clock timings in §3.2 from profiling overhead. The `lm_head` row is measured directly in
the INT4 path (where it is the only cuBLAS work); the BF16 "quantized linears" figure is that
path's cuBLAS total minus the same `lm_head` value (identical op and shape in both).

### 3.2 End-to-end latency and throughput

| Workload | BF16 | DualQuant W4A4 | Speedup |
|---|---|---|---|
| Prefill batch=1 | 3904.6 tok/s (21.00 ms) | **6421.4 tok/s (12.77 ms)** | **1.64x** |
| Prefill batch=16 | 8940.3 tok/s (146.75 ms) | **22003.4 tok/s (59.63 ms)** | **2.46x** |
| Decode batch=1 | 56.6 tok/s (17.66 ms/step) | **83.7 tok/s (11.95 ms/step)** | **1.48x** |
| Decode batch=16 | 694.5 tok/s (23.04 ms/step) | **964.1 tok/s (16.60 ms/step)** | **1.39x** |

On the quantized linear layers alone — the only part quantization touches — the speedups are
1.97x, **3.68x**, 1.79x and 1.78x. The unquantized remainder (`lm_head`, attention, norms,
RoPE, KV cache: bit-identical work in both paths) is 27.9% / 39.6% / 25.9% / 41.3% of the
INT4 step.

---

## 4. Quantization / dequantization / format-conversion overhead

| Overhead | Cost | Share of the W4A4 step |
|---|---|---|
| Activation quantization (incl. DualQuant's fused `beta` scaling) | 0.702–3.089 ms | **4.2 – 6.2%** |
| Weight dequantization at runtime | **none** | 0% |
| Format conversion / weight packing | one-time, offline | 0% at inference |
| DualQuant `beta` scaling as a separate kernel | **none** (fused) | 0% |

- Activation quantization is measured, not estimated, and is the only per-step quantization
  cost in the pipeline.
- There is no runtime dequantization: weights stay packed in INT4 and are consumed directly
  by the INT4 tensor-core instruction, with per-group scales applied in the GEMM epilogue
  rather than by materialising a dequantized weight matrix.
- The fused `beta` scaling costs no separate launch because it is applied inside the
  activation-quantization kernel while the activation is read from global memory.

---

## 5. Engineering notes

### Fixed: a 1.41 GB fixed allocation in upstream nunchaku

`QuantizedGEMM::init()` called `cudaDeviceSetLimit(cudaLimitStackSize, 8192)`. That limit is
*per thread*, so the driver reserves it for every resident thread on the device — on an
RTX 4090 (128 SMs × ~1440 threads) a fixed ~1.5 GB, paid even by a 128×128 layer.
`cuobjdump -res-usage` reports `STACK:0` for all 903 compiled kernels, so the raised limit
bought nothing. Leaving the CUDA default cut the fixed cost from 1509.95 MB to 100.66 MB,
with decode speed unchanged and the graph decode still bit-exact. Ruled out along the way:
allocator behaviour (plain `cudaMalloc` differed by 31 MB), code size (the `.so` is 16.8 MB),
and kernel count (byte-identical with 903 vs 1229 kernels compiled).

### Remaining kernel-level headroom: no split-K

True GPU time, cold cache, 32 distinct weights per shape:

| Shape | Weight bytes | GEMM time | Bandwidth |
|---|---|---|---|
| gate/up_proj (K=4096, 64 K-iterations) | 31.2 MB | 37.7 us | **828 GB/s** |
| down_proj (K=14336, 224 K-iterations) | 31.2 MB | 90.5 us | **345 GB/s** |

Identical bytes, 2.4x the time: the limiter is **K-loop length**, not bandwidth and not
occupancy. cuBLAS reaches ~960 GB/s on `down_proj` because its GEMV path splits K; this
kernel does not. Adding split-K is the main remaining lever for batch-1 decode.

Two hypotheses tested and rejected, recorded so they are not re-tried: running the
independent q/k/v and gate/up projections on separate CUDA streams (**0.79x — slower**;
fork/join event overhead exceeds the overlap at these kernel sizes), and halving `BLOCK_N`
to 64 to double the block count (**0.2% change**: `down_proj` 90.69 → 90.46 us — which is
what disproved the occupancy explanation; reverted to the upstream value).

---

## 6. Reproduction

```bash
# accuracy through the real packed path (and BF16 reference), paper's 8192 setting
python ppl_eval.py --mode bf16 --seqlen 8192
python ppl_eval.py --mode int4 --seqlen 8192

# memory: component breakdown + totals per workload
python memory_breakdown.py --mode bf16
python memory_breakdown.py --mode int4

# throughput: all four configs, correctness-gated then timed
python graph_bench_all.py --mode bf16
python graph_bench_all.py --mode int4

# per-component GPU time breakdown for all four configs
python profile_breakdown.py --mode bf16
python profile_breakdown.py --mode int4
```

`wrap_cache/` holds the cached DualQuant BCD output (`mat_q` + `beta`) per projection, so
reruns take ~16 s instead of ~35–70 min.

Note: the repository's legacy PPL evaluator cannot produce these numbers — it sets its window
to `model.config.max_position_embeddings` (131072 for Llama-3.1-8B) and OOMs on a 24 GB GPU
at any precision, BF16 included. `ppl_eval.py` uses the standard non-overlapping-window
convention instead; its BF16 result matching the paper's reference is what validates it.
