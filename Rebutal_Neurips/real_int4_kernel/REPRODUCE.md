# Reproducing the end-to-end W4A4 results

Everything needed to regenerate the numbers in `RESULTS_end_to_end.md` (and therefore
`REVIEWER_RESPONSE_draft.md`) from a clean machine.

The exact environment used is recorded in **`ENVIRONMENT.txt`**: RTX 4090 (24 GB), driver
560.35.03, CUDA toolkit 12.6, Python 3.12.13, torch 2.6.0+cu124, transformers 4.46.3,
triton 3.2.0.

> **IMPORTANT — paths.** The scripts were developed with the patched kernel built under a
> scratch directory, so several files hard-code that location. After building, fix them:
> ```bash
> grep -rln "/tmp/claude-0" .          # lists every file needing the edit
> grep -rl  "/tmp/claude-0" . | xargs sed -i "s|/tmp/claude-0/[^\"']*/svdquant_repo|<your svdquant path>|g"
> ```
> The value must end at the directory containing
> `build/lib.linux-x86_64-cpython-312/nunchaku_min*.so`.

---

## 1. Python environment

The pinned versions matter — a newer `transformers` breaks this legacy codebase, and a
mismatched torch/CUDA build silently produces a torch that cannot run on the driver.

```bash
source /venv/main/bin/activate          # or your own venv

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.2.0
pip install transformers==4.46.3        # NOT newer: 5.x breaks this codebase
pip install accelerate sentencepiece datasets
```

Two traps worth knowing:

- Installing `dmx-compressor` pulls a `torch` built for cu130, which will not run on this
  driver. If you install it, reinstall torch afterwards with the command above.
- `transformers` 4.46.3 is required. `StaticCache`'s signature and the Llama attention
  internals used by the CUDA-graph harness differ in later versions.

Model access:

```bash
export HUGGING_FACE_HUB_TOKEN=<your token>   # meta-llama/Llama-3.1-8B is gated
```

---

## 2. Build the patched nunchaku (SVDQuant) kernel

The W4A4 GEMM is third-party; our changes to it are shipped here as a patch plus three new
files. Base upstream commit: **`e9ad053`** of `https://github.com/dbw6/svdquant.git`.

```bash
git clone --recursive https://github.com/dbw6/svdquant.git
cd svdquant
git checkout e9ad053
git submodule update --init --recursive          # cutlass, json, mio, spdlog
# exact submodule commits used are recorded in nunchaku_patches/submodule_pins.txt
# (cutlass a75b4ac matters: the W4A4 kernel is built against it)

PATCHES=<path to>/real_int4_kernel/nunchaku_patches

# 1) modifications to existing upstream files
git apply "$PATCHES/0001-dualquant-w4a4-modifications.patch"

# 2) files that do not exist upstream
cp -r "$PATCHES/new_files/." .

# 3) build (only the W4A4 path; skips FluxModel/SanaModel/flash-attn)
export MAX_JOBS=32
export CUDA_HOME=/usr/local/cuda
python setup_minimal.py build_ext --inplace
```

This produces `build/lib.linux-x86_64-cpython-312/nunchaku_min*.so`. The benchmark scripts
locate it via a hard-coded path near the top of each file — **edit `_NUNCHAKU_DIR` in
`full_model_nunchaku.py`, `graph_bench_all.py`, `ppl_eval.py`, `memory_breakdown.py` and
`profile_breakdown.py` to point at your build directory.**

Build takes ~10 min; `gemm_w4a4_launch_bf16.cu` alone is several minutes of CUTLASS
template instantiation.

### 2.1 What the patch changes, and why

| Change | File | Purpose |
|---|---|---|
| `beta` applied inside the activation-quantize kernel | `src/kernels/zgemm/gemm_w4a4.cuh` (`load_act_to_fpsum`, `quantize_w4a4_fuse_lora_kernel::Arguments`) | Applies DualQuant's per-channel scale during the global-memory load, so the fused scaling needs no separate kernel launch. Deliberately NOT routed through upstream's `smooth`, which is consumed in a lane-permuted packed layout and would apply `beta` to the wrong channels. |
| `beta` threaded through the launch API | `zgemm.h`, `gemm_w4a4_launch.cuh`, `gemm_w4a4_launch_impl.cuh`, `gemm_w4a4.cu` | Optional trailing `Tensor beta = {}` argument; existing call sites are unaffected. |
| `mul_broadcast_lastdim_bf16` | `nunchaku/csrc/fused_beta_mul.cu` (new) | Standalone `y[m,k] = x[m,k] * beta[k]` kernel, used by the non-fused `forward_beta` path and the graph variants. |
| `forward_static`, `forward_beta`, `forward_graph*`, `quantize_probe`, `act_buffer_bytes`, `nocache` support | `nunchaku/csrc/gemm.h` | `forward_static` is the capture-safe path: no allocation, and it pushes the caller's stream onto nunchaku's internal stream stack so its kernels land inside a CUDA graph. `quantize_probe` returns the packed activation so the 4-bit claim can be verified from Python. |
| **Removed** `cudaDeviceSetLimit(cudaLimitStackSize, 8192)` | `nunchaku/csrc/gemm.h` (`init`) | That limit is *per thread*; the driver reserves it for every resident thread, costing a fixed ~1.5 GB on a 4090 even for a 128×128 layer. All 903 kernels report `STACK:0`, so it bought nothing. Now opt-in via `NUNCHAKU_STACK_SIZE`. **This is worth ~1.41 GB and is required to reproduce the memory numbers.** |
| Explicit stream on raw kernel launches | `gemm_w4a4_launch_impl.cuh` | Upstream launched with no stream argument, i.e. always the legacy stream 0, so the kernels were never recorded during graph capture. Required for correct CUDA-graph capture. |
| `cudaFuncSetAttribute` guarded by a once-flag | `gemm_w4a4_launch_impl.cuh` | Host-side call, illegal during graph capture, and idempotent. |
| `lora_act_out.zero_()` skipped when empty | `gemm_w4a4_launch_impl.cuh` | `lora_rank = 0` here, so this was a blocking `cudaMemset` of 0 bytes on every call. |
| `BLOCK_M` 256 → 128 | `src/kernels/zgemm/gemm_base.cuh` + matching constant in `src/Linear.cpp` | Upstream's 256 was sized for diffusion batch sizes and padded every decode activation to 256 rows. Both constants must match or `quantize()` asserts. |
| `NUNCHAKU_SYNC_ALLOC` escape hatch | `src/Tensor.h` | Optional plain `cudaMalloc` instead of `cudaMallocAsync`. Off by default; kept only because it was used to rule out allocator behaviour during diagnosis. |
| `NUNCHAKU_BF16_ONLY` | `gemm_w4a4.cu` + `setup_minimal.py` | Skips instantiating the unused FP16 W4A4 template tree (327 of 1229 kernels). Purely a build-time saving; measured to have **no** effect on memory or speed. |

`BLOCK_N` is left at the upstream value of 128 — halving it was tested and produced a 0.2%
change, so it was reverted.

---

## 3. First run: DualQuant BCD cache

Converting the model runs DualQuant's BCD per projection, which takes ~35–70 min for
Llama-3.1-8B on CPU. `wrap_cache.py` caches the resulting `mat_q` + `beta` per projection,
keyed on (model, layer, block size), so **subsequent runs take ~16 s**. The cache is
independent of which GEMM kernel consumes it.

If `wrap_cache/` is already populated (224 `.pt` files for Llama-3.1-8B), skip ahead. To
build it from scratch, just run any of the commands in §4 once and let it populate.

---

## 4. Commands that produce each reported number

Run each twice and take the second (warm) run: the first pass leaves GPU clocks boosted
(verify with `nvidia-smi --query-gpu=clocks.sm --format=csv`: ~2745 MHz warm vs ~210 MHz idle).

```bash
cd Quantization_Repo_July2025/Dualquant_codebase_20260508/Rebutal_Neurips/real_int4_kernel

# §1 Accuracy — WikiText-2 PPL at the paper's 8192 setting
python ppl_eval.py --mode bf16 --seqlen 8192      # -> 5.61
python ppl_eval.py --mode int4 --seqlen 8192      # -> 6.74

# §2 Memory — component breakdown and totals per workload
python memory_breakdown.py --mode bf16
python memory_breakdown.py --mode int4

# §3.2 Throughput — all four configs, correctness-gated then timed
python graph_bench_all.py --mode bf16
python graph_bench_all.py --mode int4

# §3.1 Per-component GPU time breakdown for all four configs
python profile_breakdown.py --mode bf16
python profile_breakdown.py --mode int4
```

### Verifying the path really is W4A4

```bash
# activation operand is exactly 4.00 bits/element, 16 levels in [-8,7]
python verification/verify_act_int4.py

# INT4 tensor-core instruction is actually emitted
cuobjdump -sass <svdquant>/build/lib.linux-x86_64-cpython-312/nunchaku_min*.so \
  | grep -o "IMMA\.16864\.S4\.S4" | head
```

---

## 5. Script inventory

**Canonical — these produce the reported numbers:**

| Script | Produces |
|---|---|
| `ppl_eval.py` | WikiText-2 PPL through the packed path and for BF16 |
| `memory_breakdown.py` | memory component breakdown + per-workload totals |
| `graph_bench_all.py` | prefill/decode × batch 1/16 throughput, with correctness gates |
| `profile_breakdown.py` | per-component GPU time for all four workloads |
| `full_model_nunchaku.py` | model conversion (`NunchakuW4A4Linear`) imported by the above |
| `wrap_cache.py` | disk cache for DualQuant BCD output |
| `bf16_baseline_e2e.py`, `e2e_bench_utils.py` | BF16 baseline model construction and shared helpers |

**Verification / diagnostic scripts** (in `verification/`, run after fixing paths as above):

| Script | Checks |
|---|---|
| `verify_act_int4.py` | activation operand is exactly 4.00 bits/element, 16 levels in [-8,7] |
| `test_awq_real_layer.py` | SQNR of the packed path vs FP32 on a real cached DualQuant layer (19.0 dB) |
| `test_forward_graph_beta_ns_correctness.py`, `test_forward_fast_correctness.py` | correctness of the graph / fast GEMM entry points across M and repeated calls |
| `test_pershape_gpu.py` | true per-shape GPU time and bandwidth, cold cache, 32 distinct weights |
| `split_quant_gemm.py` | splits each layer's cost into quantize vs GEMM |
| `ceiling_analysis.py` | achieved TOP/s and GB/s vs BF16 across M |
| `test_multistream.py` | the rejected multi-stream overlap experiment (0.79x) |
| `test_awq_gemv_repack.py`, `test_fused_speed.py`, `debug_offbyone.py` | AWQ packing validation, fused-beta timing, PPL harness off-by-one check |

**Superseded — kept for history, do not use for reported numbers:**
`whole_step_graph_decode.py` (decode-only precursor to `graph_bench_all.py`),
`_profile_graph_replay.py` (single-workload precursor to `profile_breakdown.py`),
`full_model_w4a4.py` / `fused_w4a4_kernel.py` (hand-rolled Triton W4A4 path, lost to cuBLAS
BF16 and was replaced by the nunchaku kernel), `awq_gemv_pack.py` (validated AWQ GEMV
weight-repacking artifact from an experiment that was a net regression end-to-end).

---

## 6. Notes and gotchas

- **The repo's legacy PPL evaluator cannot produce these numbers.**
  `legacy_vendor/eval_utils.py` sets its window to `model.config.max_position_embeddings`
  (131072 for Llama-3.1-8B) and asks HuggingFace to compute the loss over it, which OOMs a
  24 GB GPU at any precision, BF16 included. `ppl_eval.py` uses the standard
  non-overlapping-window convention and accumulates cross-entropy in fp32 slices; its BF16
  result matching the paper's reference (5.61 vs 5.62) is what validates it.
- **PPL is not comparable across sequence lengths.** 8192 matches the paper. The same setup
  at 2048 gives different absolute values for both precisions.
- **Long-context accuracy runs need the `nocache` path.** The fast paths keep persistent
  per-layer scratch; at M=8192, `cached_out` alone is ~15 GB across 224 layers.
  `ppl_eval.py` selects `NUNCHAKU_FWD_MODE=nocache` for this reason.
- **`forward_static` requires a non-default CUDA stream** and must not see an allocation
  during capture. In practice: run the prefill (which sizes the buffers) *before* the
  decode-shape warm-up, and do not run a different M afterwards, or a reallocation will free
  buffers the captured graph points to. `graph_bench_all.py` documents the required ordering.
- **Do not measure INT4 memory with `torch.cuda.memory_allocated()`** — it cannot see
  nunchaku's `cudaMallocAsync` buffers and under-reports by ~3.5x. Use
  `torch.cuda.mem_get_info()`.
