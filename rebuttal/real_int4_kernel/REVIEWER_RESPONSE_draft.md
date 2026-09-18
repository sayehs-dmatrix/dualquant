# Response: end-to-end results from the actual packed W4A4 inference path

We thank the reviewer for this point. We agree that operator-level microbenchmarks alone do
not establish system-level benefit, and that accuracy must come from the same packed path
that produces the efficiency numbers. We have therefore built a **real packed W4A4
inference path for Llama-3.1-8B** and re-measured everything end-to-end on it: perplexity,
latency, throughput, memory, and quantization/format-conversion overhead. All numbers below
come from that single path — no fake quantization, and no operator-level extrapolation.

**Implementation.** DualQuant's BCD produces the quantized-domain weight `mat_q` and the
per-input-channel scale `β` for all 224 linear layers (7 projections × 32 layers). These are
packed into INT4 and executed by the CUTLASS W4A4 GEMM from MIT's nunchaku / SVDQuant, so
the GEMM itself is an established third-party kernel rather than one of our own. On top of
it we implemented: (i) a CUDA kernel that applies DualQuant's per-channel `β` **inside**
nunchaku's activation-quantization kernel, during the global-memory load, so the fused
scaling costs no separate kernel launch; (ii) a capture-safe GEMM entry point that performs
no allocation and inherits the caller's stream, which is what makes whole-model CUDA-graph
capture possible; and (iii) the benchmark harness itself. Embedding, `lm_head` and norms are
left in BF16, as is standard.

We verified the path is genuinely W4A4 rather than a packed-storage-with-float-math
shortcut: the activation operand handed to the GEMM is **exactly 4.00 bits/element**
(packed `[M, K/2]`, 16 levels spanning [-8, 7], one BF16 scale per 64 channels), and the
emitted SASS is `IMMA.16864.S4.S4`, i.e. the INT4 tensor-core instruction. Both operands are
INT4.

**Measurement protocol.** Batch-1 inference in an eager PyTorch loop is limited by CPU
dispatch (~1365 kernel launches per token against ~8 ms of GPU work), which measures
framework overhead rather than quantization. We therefore capture the **entire** forward
step (embedding, all 32 blocks including attention and norms, LM head) into a single CUDA
graph and replay it, and we apply **exactly the same treatment to the BF16 baseline**, so
the comparison isolates quantization. Every configuration is correctness-gated before it is
timed: prefill logits are bit-exact against the uncaptured run (max |diff| = 0) and decode
reproduces identical generated tokens. All results are warm runs on one RTX 4090 (24 GB) and
are stable to <0.2% across repetitions.

---

## 1. Accuracy from the actual packed path

WikiText-2 perplexity evaluated **through the packed INT4 kernel**, at the same sequence
length (8192) as the paper's reported results, with the BF16 baseline measured under
identical settings:

| Llama-3.1-8B | Path | WikiText-2 PPL |
|---|---|---|
| Baseline | BF16 | **5.61** |
| DualQuant W4A4 | real packed INT4 kernel | **6.74** |

**The real packed path reproduces the paper's fake-quantization result to within 0.01 PPL**
(6.74 measured on the packed kernel vs 6.73 reported with fake quantization), and the BF16
baseline reproduces the paper's reference (5.61 vs 5.62). The fake-quantization accuracy
reported in the paper is therefore an accurate proxy for the deployed low-precision path,
not an optimistic simulation of it: the same weights and activations, executed as packed
INT4 through INT4 tensor cores, give the same perplexity. Per-layer SQNR against FP32 on a
real DualQuant layer is 19.0 dB, consistent with the expected W4A4 noise floor.

## 2. Memory

### 2.1 Component breakdown (GB)

| Component | BF16 | DualQuant W4A4 |
|---|---|---|
| Quantized layer weights (6.98 B params) | 13.959 | **3.490** |
| Per-64-group weight scales (BF16) | — | 0.218 |
| Embedding (BF16, not quantized) | 1.051 | 1.051 |
| `lm_head` (BF16, not quantized) | 1.051 | 1.051 |
| Runtime/allocator overhead | 0.136 | 0.107 |
| **Weights subtotal** | **16.196** | **5.916** |
| First-forward scratch | 0.103 | 0.300 |
| KV cache (BF16) — **prefill** batch 1 / 16 | 0 / 0 | 0 / 0 |
| KV cache (BF16) — **decode** batch 1 / 16 | 0.024 / 0.382 | 0.024 / 0.382 |
| Other activations (BF16 residual, logits) — **prefill** batch 1 / 16 | 0.030 / 0.356 | 0.030 / 0.356 |
| Other activations (BF16 residual, logits) — **decode** batch 1 / 16 | 0.012 / 0.065 | 0.012 / 0.065 |

INT4 packed activation operands account for 0.080 GB of the scratch across all 224 layers;
the KV cache and residual stream remain BF16 in both paths and are therefore identical.

The activation rows are per workload because they differ: prefill runs with
`use_cache=False` and so holds **no** KV cache, while its activation term is dominated by
the full-sequence logits tensor (`[batch x 82 x 128256]` BF16 = 0.021 / 0.337 GB) instead
of the single-position `[batch x 1 x 128256]` of a decode step. Each total in §2.2 is
`weights subtotal + first-forward scratch + that workload's KV + its other activations`,
which is why prefill batch=1 is 0.006 GB *below* decode batch=1 (−0.024 KV, +0.018 logits)
and 0.091 GB below at batch=16.

### 2.2 End-to-end memory (GB)

| Workload | BF16 | DualQuant W4A4 | Reduction |
|---|---|---|---|
| Prefill batch=1 | 16.329 | **6.246** | **2.61x** |
| Prefill batch=16 | 16.655 | **6.572** | **2.53x** |
| Decode batch=1 | 16.335 | **6.252** | **2.61x** |
| Decode batch=16 | 16.746 | **6.663** | **2.51x** |

Measured with `torch.cuda.mem_get_info()` (whole-GPU accounting); `memory_allocated()` is
not used because it cannot see the kernel's `cudaMallocAsync` buffers and under-reports the
INT4 footprint by ~3.5x. Weights alone: 16.196 → 5.916 GB (**2.74x**).

---

## 3. Latency and throughput

**Workload lengths.** Prefill is an **82-token** prompt in a single forward with
`use_cache=False`; decode continues that prompt for **100 generated tokens** greedily with a
KV-cache (182 cached positions at the end). The **8192 length in §1 applies only to the
perplexity run** — it is not the throughput or memory workload. Throughput is
`batch x tokens / latency`, e.g. prefill batch=16 is 16 x 82 / 146.75 ms = 8940 tok/s, and
decode advances one token per sequence per step.

### 3.1 Component breakdown of GPU time (ms), **BF16 → W4A4**

| Component | Prefill b1 | Prefill b16 | Decode b1 | Decode b16 |
|---|---|---|---|---|
| Quantized linears (224) | 17.330 → **8.818** | 121.490 → **32.981** | 14.747 → **8.235** | 16.315 → **9.180** |
| Activation quantization (W4A4 only) | — → 0.823 | — → 3.089 | — → 0.702 | — → 0.716 |
| `lm_head` (BF16 in both) | 1.189 → 1.189 | 9.257 → 9.257 | 1.073 → 1.073 | 1.103 → 1.103 |
| Attention | 0.191 → 0.202 | 0.930 → 0.939 | 0.424 → 0.422 | 1.730 → 1.735 |
| Norms / RoPE / KV-cache / residual | 2.430 → 2.332 | 14.029 → 13.374 | 1.602 → 1.636 | 4.178 → 4.139 |
| **Total GPU** | **21.140 → 13.363** | **145.705 → 59.640** | **17.847 → 12.067** | **23.325 → 16.873** |

Each column sums to that workload's measured total. On the quantized linear layers alone —
the only part of the network quantization touches — the speedups are 1.97x, **3.68x**, 1.79x
and 1.78x respectively.

### 3.2 End-to-end latency and throughput

| Workload | BF16 | DualQuant W4A4 | Speedup |
|---|---|---|---|
| Prefill batch=1 | 3904.6 tok/s (21.00 ms) | **6421.4 tok/s (12.77 ms)** | **1.64x** |
| Prefill batch=16 | 8940.3 tok/s (146.75 ms) | **22003.4 tok/s (59.63 ms)** | **2.46x** |
| Decode batch=1 | 56.6 tok/s (17.66 ms/step) | **83.7 tok/s (11.95 ms/step)** | **1.48x** |
| Decode batch=16 | 694.5 tok/s (23.04 ms/step) | **964.1 tok/s (16.60 ms/step)** | **1.39x** |

---

## 4. Quantization / dequantization and format-conversion overhead

The reviewer asks specifically whether these overheads dominate end-to-end execution. They
do not, and the breakdown in §3.1 isolates each of them:

| Overhead | Cost | Share of the W4A4 step |
|---|---|---|
| Activation quantization (incl. DualQuant's fused `β` scaling) | 0.702–3.089 ms | **4.2 – 6.2%** |
| Weight dequantization at runtime | **none** | 0% |
| Format conversion / weight packing | one-time, offline | 0% at inference |
| DualQuant `β` scaling as a separate kernel | **none** (fused) | 0% |

Behind that table:

- **Activation quantization is measured, not estimated**, and is 4.2–6.2% of end-to-end GPU
  time across all four workloads. It is the only per-step quantization cost in the pipeline.
- **There is no runtime dequantization.** Weights stay packed in INT4 and are consumed
  directly by the INT4 tensor-core instruction; the per-group scales are applied in the GEMM
  epilogue, not by materialising a dequantized weight matrix.
- **DualQuant's fused scaling costs no separate launch**, because we apply `β` inside the
  activation-quantization kernel while the activation is being read from global memory. This
  is what the paper's "negligible fused-scaling overhead" claim reduces to at the system
  level, now measured end-to-end rather than at operator level.
- Weight packing is a one-time offline transformation, not part of the inference path.

---

## 5. Summary

On the actual packed W4A4 path for Llama-3.1-8B, with the BF16 baseline given the identical
CUDA-graph treatment and every configuration correctness-gated:

- **Accuracy from the packed path matches the reported fake-quantization result** (6.74 vs
  6.73 WikiText-2 PPL), with the BF16 baseline reproducing the paper's reference (5.61 vs 5.62).
- **2.51–2.61x lower total memory** end-to-end (2.74x on weights alone).
- **1.64x / 2.46x faster prefill** and **1.48x / 1.39x faster decode** at batch 1 / 16.
- **Quantization overhead is 4.2–6.2%** of end-to-end GPU time, with no runtime
  dequantization and no separate kernel for the fused scaling.

These are full-model numbers from the deployed packed path, not operator-level
microbenchmarks. We will add the results, the measurement protocol, and the correctness
checks to the revised paper.
