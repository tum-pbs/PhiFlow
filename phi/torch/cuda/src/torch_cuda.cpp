#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h> // SpMM, SpMV
#include <vector>
#include "torch_cuda.hpp"
#include <ctime>
#include <chrono>

#define TRUE 1
#define FALSE 0

namespace torch_cuda {

    // GLOBAL VARIABLES
    void *globalBuffer = NULL;
    size_t globalBufferSize = 0;
    cusparseHandle_t globalHandle = NULL;
    cusparseSpMatDescr_t globalSparseMatrixA;
    torch::Tensor dC;
    cusparseDnVecDescr_t dC_cusparse;

    void allocateBuffer(size_t newSize) {
        if(newSize > globalBufferSize) {
            CHECK_CUDA( cudaFree(globalBuffer) )
            CHECK_CUDA( cudaMalloc(&globalBuffer, newSize) )
            globalBufferSize = newSize;
        }
    }

    torch::Tensor nan_division(torch::Tensor a, torch::Tensor b) {
        auto res = a/b;
        res.index_put_({~torch::isfinite(res)}, 0);
        return res;
    }

    template<typename T>
    void print_variable(char* name, T variable) {
        std::cout << name << "\n" << variable << std::endl;
    }

    torch::Tensor floatSpMM(const at::Tensor& dA_csrOffsets,
                            const at::Tensor& dA_columns,
                            const at::Tensor& dA_values,
                            at::Tensor& dB,
                            const int64_t dim_i,
                            const int64_t dim_j)
    {
        cusparseHandle_t     handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        auto A_nnz          = dA_values.size(0);
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        float alpha           = 1.0f;
        float beta            = 0.0f;

        dB = dB.view({dim_j});
        CHECK_CUSPARSE( cusparseCreate(&handle) )
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_i, dim_j, A_nnz,
                                          dA_csrOffsets.data_ptr<int64_t>(), dA_columns.data_ptr<int64_t>(),
                                          dA_values.data_ptr<float>(),
                                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
        // Create dense vector X
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<float>(), CUDA_R_32F) )
        // Create dense vector y
        torch::Tensor dC = torch::empty({dim_i}, dB.options());
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, dim_i, dC.data_ptr<float>(), CUDA_R_32F) )
        // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                     CUSPARSE_CSRMV_ALG2, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // execute SpMV
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                     CUSPARSE_CSRMV_ALG2, dBuffer) )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
        CHECK_CUSPARSE( cusparseDestroy(handle) )
        dB = dB.view({dim_j, 1});
        dC = dC.view({dim_i, 1});
        return dC;
    }

    torch::Tensor doubleSpMM(const at::Tensor& dA_csrOffsets,
                            const at::Tensor& dA_columns,
                            const at::Tensor& dA_values,
                            at::Tensor& dB,
                            const int64_t dim_i,
                            const int64_t dim_j)
    {
        cusparseHandle_t     handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        auto A_nnz          = dA_values.size(0);
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;
        double alpha           = 1.0f;
        double beta            = 0.0f;

        dB = dB.view({dim_j});
        CHECK_CUSPARSE( cusparseCreate(&handle) )
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_i, dim_j, A_nnz,
                                          dA_csrOffsets.data_ptr<int64_t>(), dA_columns.data_ptr<int64_t>(),
                                          dA_values.data_ptr<double>(),
                                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
        // Create dense vector X
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<double>(), CUDA_R_64F) )
        // Create dense vector y
        torch::Tensor dC = torch::empty({dim_i}, dB.options());
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, dim_i, dC.data_ptr<double>(), CUDA_R_64F) )
        // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                     handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                     CUSPARSE_CSRMV_ALG2, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // execute SpMV
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                     CUSPARSE_CSRMV_ALG2, dBuffer) )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
        CHECK_CUSPARSE( cusparseDestroy(handle) )
        dB = dB.view({dim_j, 1});
        dC = dC.view({dim_i, 1});
        return dC;
    }

    torch::Tensor cusparse_SpMM(
                            const at::Tensor& dA_csrOffsets,
                            const at::Tensor& dA_columns,
                            const at::Tensor& dA_values,
                            at::Tensor& dB,
                            const int64_t dim_i,
                            const int64_t dim_j)
    {
        if(dB.dtype() == torch::kFloat32) {
            return floatSpMM(dA_csrOffsets, dA_columns, dA_values, dB, dim_i, dim_j);
        }
        else if(dB.dtype() == torch::kFloat64) {
            return doubleSpMM(dA_csrOffsets, dA_columns, dA_values, dB, dim_i, dim_j);
        }
        else {
            fprintf(stderr, "error: %s: cusparse_SpMM type undexpected at line %d\n", __FILE__,       \
                    __LINE__);                                                     \
            exit(1);
        }
    }

    torch::Tensor __cusparse_SpMM(
                            at::Tensor& dB,
                            const int64_t dim_i,
                            const int64_t dim_j)
    {

        cusparseDnVecDescr_t vecX;
        size_t               bufferSize = 0;

        dB = dB.view({dim_j});

        // Create dense vector X
        if(dB.dtype() == torch::kFloat32) {
            float alpha           = 1.0f;
            float beta            = 0.0f;
            CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<float>(), CUDA_R_32F) )
            // allocate an external buffer if needed
            CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                        globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_32F,
                                        CUSPARSE_CSRMV_ALG2, &bufferSize) )
            allocateBuffer(bufferSize);
            // execute SpMV
            CHECK_CUSPARSE( cusparseSpMV(globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_32F,
                                        CUSPARSE_CSRMV_ALG2, globalBuffer) )
        }
        else if(dB.dtype() == torch::kFloat64) {
            double alpha           = 1.0f;
            double beta            = 0.0f;
            CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<double>(), CUDA_R_64F) )
            // allocate an external buffer if needed
            CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                        globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_64F,
                                        CUSPARSE_CSRMV_ALG2, &bufferSize) )
            allocateBuffer(bufferSize);
            // execute SpMV
            CHECK_CUSPARSE( cusparseSpMV(globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_64F,
                                        CUSPARSE_CSRMV_ALG2, globalBuffer) )
        }

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
        dB = dB.view({1, dim_j});
        dC = dC.view({1, dim_i});
        return dC;
    }

    std::vector<at::Tensor> conjugate_gradient(torch::Tensor csr_values, torch::Tensor csr_cols, torch::Tensor csr_rows, int64_t csr_dim0, int64_t csr_dim1, int64_t nnz,
                            torch::Tensor y, torch::Tensor x, torch::Tensor rtol, torch::Tensor atol, torch::Tensor max_iter, bool trj) {

        if(trj) {
            std::cout << "Trajectory tracing not supported. Return only final values" << std::endl;
        }

        CHECK_CUSPARSE( cusparseCreate(&globalHandle) )
        dC = torch::empty({csr_dim0}, x.options());
        if(csr_values.dtype() == torch::kFloat32) {
            CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
                                              csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
                                              csr_values.data_ptr<float>(),
                                              CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
            CHECK_CUSPARSE( cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<float>(), CUDA_R_32F) )
            CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
                                            csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
                                            csr_values.data_ptr<float>(),
                                            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
        }
        else if(csr_values.dtype() == torch::kFloat64) {
            CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
                                              csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
                                              csr_values.data_ptr<double>(),
                                              CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
            CHECK_CUSPARSE( cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<double>(), CUDA_R_64F) )
            CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
                                            csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
                                            csr_values.data_ptr<double>(),
                                            CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
        }

        torch::Tensor sum_y = torch::sum(torch::mul(y,y), -1);
        torch::Tensor atol_sq = atol * atol;
        torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
        torch::Tensor lin_x = __cusparse_SpMM(x, csr_dim0, csr_dim1);
        auto residual = y - lin_x;
        auto dx = residual;
        int64_t it_counter = 0;
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
            torch::Tensor dy = __cusparse_SpMM(dx, csr_dim0, csr_dim1);
            function_evaluations += not_finished;
            auto dx_dy = torch::sum(residual * dy, -1, TRUE);
            auto step_size = nan_division(residual_squared, dx_dy); // Account for nan values
            step_size = torch::mul(step_size, not_finished);
            x += (step_size * dx);
            if(it_counter % 50 == 0) {
                residual = y - __cusparse_SpMM(x, csr_dim0, csr_dim1);
                function_evaluations += 1;
            }
            else {
                residual = (residual - step_size * dy);
            }
            auto residual_squared_old = residual_squared;
            residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
            dx = residual + (nan_division(residual_squared, residual_squared_old)) * dx; // Account for nan values
            diverged = torch::any(residual_squared / rsq0 > 100) & (iterations >= 8);
            converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
            finished = converged | diverged | (iterations >= max_iter);
            not_finished = ~finished;
        }

        // CLOSE HANDLE AND CSR MATRIX REPRESENTATION
        CHECK_CUSPARSE( cusparseDestroySpMat(globalSparseMatrixA) )
        CHECK_CUSPARSE( cusparseDestroy(globalHandle) )
        CHECK_CUSPARSE( cusparseDestroyDnVec(dC_cusparse) )

        CHECK_CUDA( cudaFree(globalBuffer) )

        return {x, residual, torch::squeeze(iterations, -1), torch::squeeze(function_evaluations, -1),
        torch::squeeze(converged, -1), torch::squeeze(diverged, -1)};
    }


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("conjugate_gradient", &conjugate_gradient, "Conjugate gradient function");
      m.def("cusparse_SpMM", &cusparse_SpMM, "Sparse(CSR) times dense matrix multiplication on CUSPARSE");
    }

    TORCH_LIBRARY(torch_cuda, m) {
      m.def("conjugate_gradient", &conjugate_gradient);
      m.def("cusparse_SpMM", &cusparse_SpMM);
    }
}