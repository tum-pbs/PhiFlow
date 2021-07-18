#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <pytorch_custom.hpp>

torch::Tensor cublas_matmul(
    torch::Tensor matrix_a, // matrix_a has shape [m,k]
    torch::Tensor matrix_b) // matrix_b has shape [k,n]
    {
/*
    A ε [m,k] : Torch -> Row major
    B ε [k,n] : Torch -> Row major
    C ε [m, n] : Torch -> Row major

    C= α* A x B + βC
    where α and β are scalars, and A , B and C are matrices
    stored in column-major format

    From stackoverflow:
    https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

    "We will trick cuBLAS into computing (AB)^T, which will be
    outputted in column major order and will thus look like AB
    when we slyly interpret it in row-major order. So instead
    of computing AB = C, we do B^T A^T = C^T. Luckily, B^T and
    A^T we already obtained by the very action of creating A
    and B in row-major order, so we can simply bypass the
    transposition with CUBLAS_OP_N. So change the line to
    cublasSgemm(
        handle,CUBLAS_OP_N,CUBLAS_OP_N,
        n,m,k, &al, d_b,n, d_a,k, &bet, d_c,n)"

*/

    CHECK_INPUT(matrix_a);
    CHECK_INPUT(matrix_b);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    const int64_t m = matrix_a.size(0);
    const int64_t k = matrix_b.size(0);
    const int64_t n = matrix_b.size(1);

    torch::Tensor output = torch::zeros({m, n}, matrix_a.options());

    if (matrix_a.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  matrix_b.data_ptr<double>(), n, matrix_a.data_ptr<double>(), k,
                  &beta, output.data_ptr<double>(), n);
    } else if (matrix_a.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                  matrix_b.data_ptr<float>(), n, matrix_a.data_ptr<float>(), k,
                  &beta, output.data_ptr<float>(), n);
    }
    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}

// cusparseHandle_t at::cuda::getCurrentCUDASparseHandle()

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublas_matmul", &cublas_matmul, "GEMM on CUBLAS");
}