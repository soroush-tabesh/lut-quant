#include <torch/extension.h>
#include <codebook_quant.h>

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("codebook_quantize", &codebook_quantize, "Quantize a tensor using a codebook (CUDA) BF16",
          py::arg("input"), py::arg("codebook"));
    m.def("codebook_quantize_f", &codebook_quantize_f, "Quantize a tensor using a codebook (CUDA) FP32",
          py::arg("input"), py::arg("codebook"));
}
