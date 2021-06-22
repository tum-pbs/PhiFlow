#include <torch/extension.h>

#include <vector>

// CUDA Declarations
std::vector<torch::Tensor> vector_addition_cuda(
    torch::Tensor vector1,
    torch::Tensor vector2,
    torch::Tensor out,
    int n);

// C++ interfaces
#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> vector_addition(
    torch::Tensor vector1,
    torch::Tensor vector2,
    torch::Tensor out,
    int n) {
    CHECK_INPUT(vector1);
    CHECK_INPUT(vector2);
    CHECK_INPUT(out);
  return vector_addition_cuda(vector1, vector2, out, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_addition", &vector_addition, "Simple vector addition (CUDA)");
}