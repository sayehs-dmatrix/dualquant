// Tiny dedicated kernel for the DualQuant ablation: y[m,k] = x[m,k] * beta[k].
// Exists purely to avoid dispatching a separate aten::mul from Python for
// DualQuant's per-channel activation rescale -- profiling showed that single
// Python-level multiply costs ~16.5us of CPU dispatch overhead per call
// (ATen dispatcher, autograd guard, device dispatch) versus ~1.4us of actual
// GPU work, and at LLM decode call volumes (224 linear layers/token) that
// dispatch overhead alone was ~8.5ms of a ~33ms decode step. Calling this
// from C++ inside forward_fast/forward_beta folds the multiply into the
// same Python->C++ transition as the GEMM itself, with no ATen involved.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nunchaku::kernels {

__global__ void mul_broadcast_lastdim_bf16_kernel(
    const __nv_bfloat16 *x, const __nv_bfloat16 *beta, __nv_bfloat16 *out, long long numel, int K
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int k = (int)(idx % K);
        out[idx] = __hmul(x[idx], beta[k]);
    }
}

void mul_broadcast_lastdim_bf16(
    const __nv_bfloat16 *x, const __nv_bfloat16 *beta, __nv_bfloat16 *out, long long numel, int K,
    cudaStream_t stream
) {
    const int threads = 256;
    const long long blocks = (numel + threads - 1) / threads;
    mul_broadcast_lastdim_bf16_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(x, beta, out, numel, K);
}

// Only nvcc (compiling this .cu file) can legally take the address of a
// __global__ function for use with the raw driver/runtime graph-node APIs
// (cudaGraphExecKernelNodeSetParams etc.) -- gemm.h is compiled as ordinary
// C++ by the host compiler, so it needs this pointer handed to it rather
// than taking the address itself.
void* get_mul_broadcast_lastdim_bf16_kernel_ptr() {
    return reinterpret_cast<void*>(&mul_broadcast_lastdim_bf16_kernel);
}

int mul_broadcast_lastdim_bf16_launch_threads() { return 256; }

};  // namespace nunchaku::kernels
