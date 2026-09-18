// Minimal pybind for the DualQuant rebuttal: only exposes QuantizedGEMM
// (the standalone W4A4 GEMM module), skipping FluxModel/SanaModel and
// Block-Sparse-Attention entirely (not needed, and avoids compiling
// flash-attention kernels which would dominate build time).
#include "gemm.h"
#include "utils.h"
#include "ops.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Standalone binding for validating the AWQ GEMV weight-repacking format
    // before wiring it into QuantizedGEMM's decode path.
    m.def("gemv_awq", &nunchaku::ops::gemv_awq);

    // nunchaku allocates every Tensor with cudaMallocAsync, whose pool RETAINS
    // freed blocks instead of returning them to the driver. Model conversion
    // creates a lot of short-lived temporaries, so the pool can hold on to
    // ~1GB+ of reserved-but-unused memory that shows up in mem_get_info as if
    // it were model footprint. Trimming the pool to 0 after loading returns it.
    m.def("trim_memory_pool", [](int deviceId) {
        cudaMemPool_t pool;
        checkCUDA(cudaDeviceGetDefaultMemPool(&pool, deviceId));
        checkCUDA(cudaMemPoolTrimTo(pool, 0));
    }, py::arg("deviceId") = 0);

    py::class_<QuantizedGEMM>(m, "QuantizedGEMM")
        .def(py::init<>())
        .def("init", &QuantizedGEMM::init,
            py::arg("in_features"), py::arg("out_features"),
            py::arg("bias"), py::arg("bf16"), py::arg("deviceId")
        )
        .def("reset", &QuantizedGEMM::reset)
        .def("forward", &QuantizedGEMM::forward)
        .def("forward_fast", &QuantizedGEMM::forward_fast)
        .def("forward_graph", &QuantizedGEMM::forward_graph)
        .def("forward_beta", &QuantizedGEMM::forward_beta)
        .def("forward_static", &QuantizedGEMM::forward_static)
        .def("forward_graph_beta", &QuantizedGEMM::forward_graph_beta)
        .def("forward_graph_beta_fused", &QuantizedGEMM::forward_graph_beta_fused)
        .def("quantize", &QuantizedGEMM::quantize)
        .def("quantize_probe", &QuantizedGEMM::quantize_probe)
        .def("act_buffer_bytes", &QuantizedGEMM::act_buffer_bytes)
        .def("load_weight", &QuantizedGEMM::load_weight)
        .def("load_smooth", &QuantizedGEMM::load_smooth)
    ;

    m.def_submodule("utils")
        .def("set_log_level", [](const std::string &level) {
            spdlog::set_level(spdlog::level::from_str(level));
        })
    ;
}
