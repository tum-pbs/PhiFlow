#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
cuda_bi_conjugate_gradient(
    const int64_t orderL,       // Order of the polynomial
    torch::Tensor csr_values,   // VAL              A
    torch::Tensor csr_cols,     // COL_IDX          ColIdxSp
    torch::Tensor csr_rows,     // ROW_IDX_PTR      RowIdxSpPTR
    int64_t csr_dim0,           // rows in A
    int64_t csr_dim1,           // columns in A     vecLength
    int64_t nnz,
    torch::Tensor yTen,
    torch::Tensor xTen,
    torch::Tensor rtol,
    torch::Tensor atol,
    torch::Tensor max_iter,
    bool trj);

// C++ interface

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA_TENSOR(x); CHECK_CONTIGUOUS(x)

namespace phi_torch_cuda {

std::vector<torch::Tensor>
bi_conjugate_gradient(
    const int64_t orderL,
    torch::Tensor csr_values,   // VAL              A
    torch::Tensor csr_cols,     // COL_IDX          ColIdxSp
    torch::Tensor csr_rows,     // ROW_IDX_PTR      RowIdxSpPTR
    int64_t csr_dim0,           // rows in A
    int64_t csr_dim1,           // columns in A     vecLength
    int64_t nnz,
    torch::Tensor yTen,
    torch::Tensor xTen,
    torch::Tensor rtol,
    torch::Tensor atol,
    torch::Tensor max_iter,
    bool trj)
{
    CHECK_INPUT(csr_values);
    CHECK_INPUT(csr_cols);
    CHECK_INPUT(csr_rows);
    CHECK_INPUT(yTen);
    CHECK_INPUT(xTen);
    CHECK_INPUT(rtol);
    CHECK_INPUT(atol);
    CHECK_INPUT(max_iter);

    return cuda_bi_conjugate_gradient(
        orderL, csr_values, csr_cols, csr_rows, csr_dim0, csr_dim1, nnz, yTen, xTen, rtol, atol, max_iter, trj
    );
}

// Creates a python module that can be imported in python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "bicgL module with cuda support";         // Optional module docstring
    m.def("bi_conjugate_gradient", &bi_conjugate_gradient, "Generalized bi-conjugate gradient method of order L");
}

// Allows for JIT to track this functions.
TORCH_LIBRARY(phi_torch_cuda, m) {
    m.def("bi_conjugate_gradient", &bi_conjugate_gradient);
}

}   // namespace phi_torch_cuda