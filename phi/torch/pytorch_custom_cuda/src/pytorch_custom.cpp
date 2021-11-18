#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h> // SpMM, SpMV
#include <vector>
#include "pytorch_custom.hpp"

#define GET_VARIABLE_NAME(Variable) (#Variable)
#define TRUE 1
#define FALSE 0

class Dense_matrix_cusparse {
    public:
        cusparseDnMatDescr_t mat;
        int dim0;
        int dim1;
        Dense_matrix_cusparse();
        Dense_matrix_cusparse(torch::Tensor torch_mat);
        
};

class Sparse_csr_matrix_cusparse {
    public:
        cusparseSpMatDescr_t mat;
        int dim0;
        int dim1;
        int nnz;
        Sparse_csr_matrix_cusparse();
        Sparse_csr_matrix_cusparse(torch::Tensor csr_vals, torch::Tensor csr_cols, torch::Tensor csr_rows,
                                   int rows, int cols, int nnz);
};

Dense_matrix_cusparse::Dense_matrix_cusparse() {}
Dense_matrix_cusparse::Dense_matrix_cusparse(torch::Tensor torch_mat) {
            auto shape = torch_mat.sizes();
            dim0 = shape[0];
            if(shape.size() < 2) {
                dim1 = 1;
            }
            else {
                dim1 = shape[1];
            }
            CHECK_CUSPARSE( cusparseCreateDnMat(&mat, dim0, dim1, dim0,
                torch_mat.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_COL) )
}

Sparse_csr_matrix_cusparse::Sparse_csr_matrix_cusparse(){}
Sparse_csr_matrix_cusparse::Sparse_csr_matrix_cusparse(torch::Tensor csr_vals, torch::Tensor csr_cols, torch::Tensor csr_rows,
                                                       int rows, int cols, int nnz) {
            dim0 = rows;
            dim1 = cols;
            CHECK_CUSPARSE( cusparseCreateCsr(&mat, dim0, dim1, nnz, csr_rows.data_ptr<int>(),
                            csr_cols.data_ptr<int>(), csr_vals.data_ptr<float>(),
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
}

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
    CHECK_CUDA(cudaGetLastError());

    return output;
}

torch::Tensor memory_transpose(torch::Tensor vec) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    auto dim_i = vec.size(0);
    auto dim_j = vec.size(1);
    float alpha = 1;
    float beta = 0;
    torch::Tensor C = at::empty({dim_i, dim_j}, vec.options());
    cublasSgeam(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                dim_i, dim_j,
                &alpha,
                vec.data_ptr<float>(), dim_i,
                &beta,
                vec.data_ptr<float>(), dim_i,
                C.data_ptr<float>(), dim_i);
    return C;
}

torch::Tensor cusparse_SpMM(
                        const at::Tensor& dA_csrOffsets,
                        const at::Tensor& dA_columns,
                        const at::Tensor& dA_values,
                        const at::Tensor& dB,
                        const int dim_i,
                        const int dim_j)
{

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t handle = NULL;
    void* dBuffer         = NULL;
    size_t bufferSize     = 0;
    int dim_k             = dB.size(1);
    auto   A_nnz          = dA_values.size(0);
    float alpha           = 1.0f;
    float beta            = 0.0f;

    torch::Tensor dC = torch::empty({dim_i, dim_k}, dB.options());
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    int version;
    cusparseGetVersion(handle, &version);
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_i, dim_j, A_nnz,
                                      dA_csrOffsets.data_ptr<int>(), dA_columns.data_ptr<int>(),
                                      dA_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, dim_j, dim_k, dim_k, dB.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, dim_i, dim_k, dim_k, dC.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    return dC;
}

/*
b = [0 1 2
     3 4 5]      i = 2, j = 3
A = [1 1 1 1
     1 1 1 1
     1 1 1 1]    j = 3, k = 4
c = [3  3  3  3
    12 12 12 12]

*/
torch::Tensor cusparse_SpMM_BA(
                        const at::Tensor& dA_csrOffsets,
                        const at::Tensor& dA_columns,
                        const at::Tensor& dA_values,
                        const at::Tensor& dB,
                        const int dim_j,
                        const int dim_k)
{

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t handle = NULL;
    void* dBuffer         = NULL;
    size_t bufferSize     = 0;
    int dim_i             = dB.size(0);
    auto   A_nnz          = dA_values.size(0);
    float alpha           = 1.0f;
    float beta            = 0.0f;

    torch::Tensor dC = torch::empty({dim_k, dim_i}, dB.options());
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    int version;
    cusparseGetVersion(handle, &version);
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_j, dim_k, A_nnz,
                                      dA_csrOffsets.data_ptr<int>(), dA_columns.data_ptr<int>(),
                                      dA_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, dim_j, dim_i, dim_j, dB.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, dim_k, dim_i, dim_k, dC.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    return dC.view({dim_i, dim_k});
}

std::vector<Dense_matrix_cusparse> create_cusparse_dense_matrices(torch::Tensor x0) {
    std::vector<Dense_matrix_cusparse> x;
    x.reserve(x0.size(0));

    for(int i = 0; i < x0.size(0); i++) {
        Dense_matrix_cusparse x_i(x0[i]);
        x.push_back(x_i);
    }
    return x;
}

template<typename T>
torch::Tensor create_gpu_tensor(std::vector<T> data, at::IntArrayRef shape) {
    if(typeid(T) == typeid(float)){
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tens = torch::from_blob(&data[0], shape, options);
        auto gpu_tens = tens.to(torch::Device(torch::kCUDA, 0));
        return gpu_tens;
    }
    else if(typeid(T) == typeid(int)) {
        auto options = torch::TensorOptions().dtype(torch::kInt);
        auto tens = torch::from_blob(&data[0], shape, options);
        auto gpu_tens = tens.to(torch::Device(torch::kCUDA, 0));
        return gpu_tens;
    }
    else {
        fprintf(stderr, "error: %s: create_gpu_tensor type undexpected at line %d\n", __FILE__,       \
                __LINE__);                                                     \
        exit(1);  
    }
    
}

template<typename T>
void print_variable(char* name, T variable) {
    std::cout << name << "\n" << variable << std::endl;
}

torch::Tensor conjugate_gradient(torch::Tensor csr_values, torch::Tensor csr_cols, torch::Tensor csr_rows, int csr_dim0, int csr_dim1, int nnz,
                        torch::Tensor y, torch::Tensor x,  float rtol, float atol, int max_iter, bool trj) {
/* ASSUMPTIONS
        x0 is a matrix that contains columns of vectors (each vector is an independent problem to solve)
        y is a matrix that contains columns of vectors (each vector is an independent problem to solve)
*/
    std::vector<std::vector<torch::Tensor>> trajectory;
    int batch_size = x.size(0);
    torch::Tensor sum_y = torch::sum(torch::mul(y,y), -1);
    torch::Tensor atol_sq = create_gpu_tensor(std::vector<float>{atol*atol}, torch::IntArrayRef{1});
    torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
    torch::Tensor lin_x = cusparse_SpMM_BA(csr_rows, csr_cols, csr_values, x, csr_dim0, csr_dim1);
    auto residual = y - lin_x;
    auto dx = residual;
    int it_counter = 0;
    auto residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
    auto rsq0 = residual_squared;
    auto diverged = torch::any(~x.isfinite(), -1, TRUE);
    auto iterations = torch::zeros_like(diverged, x.options());
    auto function_evaluations = torch::ones_like(iterations, x.options());
    auto converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
    auto finished = converged | diverged | (iterations >= max_iter);
    auto not_finished = ~finished;
    auto dummy_ones = torch::ones_like(finished, torch::kCUDA);
    while(!torch::equal(finished, dummy_ones)) {
        it_counter += 1;
        iterations = iterations + not_finished;
        torch::Tensor dy = cusparse_SpMM_BA(csr_rows, csr_cols, csr_values, dx,
                                         csr_dim0, csr_dim1);
        function_evaluations += not_finished;
        auto dx_dy = torch::sum(residual * dy, -1, TRUE);
        auto step_size = residual_squared / dx_dy; // Account for nan values
        step_size = torch::mul(step_size, not_finished);
        x += (step_size * dx);
        if(it_counter % 50 == 0) {
            residual = y - cusparse_SpMM_BA(csr_rows, csr_cols, csr_values, x, csr_dim0, csr_dim1);
            function_evaluations += 1;
        }
        else {
            residual = (residual - step_size * dy);
        }
        auto residual_squared_old = residual_squared;
        residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
        dx = residual + (residual_squared / residual_squared_old) * dx; // Account for nan values
        diverged = torch::any(residual_squared / rsq0 > 100) & (iterations >= 8);
        converged = torch::all(residual_squared <= tolerance_sq);
        finished = converged | diverged | (iterations >= max_iter);
        not_finished = ~finished;
    }
    return x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conjugate_gradient", &conjugate_gradient, "Conjugate gradient function");
  m.def("cublas_matmul", &cublas_matmul, "GEMM on CUBLAS");
  m.def("cusparse_SpMM", &cusparse_SpMM, "Sparse(CSR) times dense matrix multiplication on CUSPARSE");
  m.def("cusparse_SpMM_BA", &cusparse_SpMM_BA, "Dense matrix times Sparse(CSR) matrix multiplication on CUSPARSE");
}