#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>

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

torch::Tensor cusparse_matmul(
                        const at::Tensor& dA_csrOffsets,
                        const at::Tensor& dA_columns,
                        const at::Tensor& dA_values,
                        const at::Tensor& dB,
                        int A_num_rows) {

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;


    auto   B_num_rows      = dB.size(0);
    auto   B_num_cols      = dB.size(1);
    auto   A_num_cols      = B_num_rows;
    auto   C_num_rows      = A_num_rows;
    auto   C_num_cols      = B_num_cols;

    auto   A_nnz           = dA_values.size(0);
    auto   ldb             = B_num_cols;
    auto   ldc             = A_num_rows;
    float alpha           = 1.0f;
    float beta            = 0.0f;

    torch::Tensor dC = at::zeros({C_num_rows, C_num_cols}, dB.options());

    CHECK_CUSPARSE( cusparseCreate(&handle) );
    int version;
    cusparseGetVersion(handle, &version);
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets.data_ptr<int>(), dA_columns.data_ptr<int>(),
                                      dA_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_cols, B_num_rows, ldb, dB.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )

    return dC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cublas_matmul", &cublas_matmul, "GEMM on CUBLAS");
  m.def("cusparse_matmul", &cusparse_matmul, "Sparse(CSR), Dense matrix multiplication on CUSPARSE");
}