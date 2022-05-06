#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <vector>
#include "phi_torch_cuda.hpp"
#include <ctime>
#include <chrono>

#define TRUE 1
#define FALSE 0

namespace phi_torch_cuda {
    /*
        The following variables are public.

        They are instantiated at the beginning of the call to conjugate_gradient() and released at the end. They are
        used in the conjugate_gradient() sub-functions that perform cuSPARSE operations.
    */
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

    /*
        - "Sparse CSR * Dense" matrix multiplication.
        - 32 bit floating point precision.
        - This function is called from cusparse_SpMV()

        The algorithm that is chosen to perform the multiplication is the following:
        CUSPARSE_SPMV_CSR_ALG2. Reference: https://docs.nvidia.com/cuda/cusparse/index.html

        This algorithm has been chosen because:
        1. Why Matrix*Vector operation?: The sparse * dense matrix multiplications that are performed in the solution of the
        conjugate gradient method are always [Sparse matrix * Dense vector]. A Matrix * Matrix multiplication could
        also be used. This would imply making sure that we are interpreting the dense matrix with the right dimensions and possibly some transpositions.
        (PyTorch tensors are row-major. CUDA is column-major). On the other hand a Vector is dimension agnostic and
        this requires no extra work.
        2. Why Algorithm 2?: The algorithm 1 is faster, but it does not provide deterministic results. Which may lead
        to a non-convergent solution.
        3. Why CUSPARSE_OPERATION_NON_TRANSPOSE?: The routine is 3x faster when we do not transpose the matrix.
    */
    torch::Tensor floatSpMV(const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            torch::Tensor& dB,
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

        auto dimension1_b = dB.size(0);
        auto dimension2_b = dB.size(1);

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
        dB = dB.view({dimension1_b, dimension2_b});
        if(dimension1_b == 1) {
            dC = dC.view({1, dim_i});
        }
        else {
            dC = dC.view({dim_i, 1});
        }
        return dC;
    }

    /*
        - "Sparse CSR * Dense" matrix multiplication.
        - 64 bit floating point precision.
        - This function is called from cusparse_SpMV()

        The algorithm that is chosen to perform the multiplication is the following:
        CUSPARSE_SPMV_CSR_ALG2. Reference: https://docs.nvidia.com/cuda/cusparse/index.html

        This algorithm has been chosen because:
        1. Why Matrix*Vector operation?: The sparse * dense matrix multiplications that are performed in the solution of the
        conjugate gradient method are always [Sparse matrix * Dense vector]. A Matrix * Matrix multiplication could
        also be used. This would imply making sure that we are interpreting the dense matrix with the right dimensions and possibly some transpositions.
        (PyTorch tensors are row-major. CUDA is column-major). On the other hand a Vector is dimension agnostic and
        this requires no extra work.
        2. Why Algorithm 2?: The algorithm 1 is faster, but it does not provide deterministic results. Which may lead
        to a non-convergent solution.
        3. Why CUSPARSE_OPERATION_NON_TRANSPOSE?: The routine is 3x faster when we do not transpose the matrix.
    */
    torch::Tensor doubleSpMV(const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            torch::Tensor& dB,
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

        auto dimension1_b = dB.size(0);
        auto dimension2_b = dB.size(1);
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
        dB = dB.view({dimension1_b, dimension2_b});
        if(dimension1_b == 1) {
            dC = dC.view({1, dim_i});
        }
        else {
            dC = dC.view({dim_i, 1});
        }
        return dC;
    }

    torch::Tensor floatSpMM(
                            const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            const torch::Tensor& dB,
                            const int64_t A_num_rows, const int64_t A_num_cols, const int64_t B_num_rows, const int64_t B_num_cols)
    {

        cusparseHandle_t     handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;

        auto   C_num_rows      = A_num_rows;
        auto   C_num_cols      = B_num_cols;

        auto   A_nnz           = dA_values.size(0);
        auto   ldb             = B_num_cols;
        auto   ldc             = A_num_rows;
        float alpha           = 1.0f;
        float beta            = 0.0f;

        torch::Tensor dC = torch::zeros({C_num_rows, C_num_cols}, dB.options());

        CHECK_CUSPARSE( cusparseCreate(&handle) );
        int version;
        cusparseGetVersion(handle, &version);
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                          dA_csrOffsets.data_ptr<int64_t>(), dA_columns.data_ptr<int64_t>(),
                                          dA_values.data_ptr<float>(),
                                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
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
                                     CUSPARSE_CSRMM_ALG1, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_TRANSPOSE,
                                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_CSRMM_ALG1, dBuffer) )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )

        return dC;
    }

    torch::Tensor doubleSpMM(
                            const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            const torch::Tensor& dB,
                            const int64_t A_num_rows, const int64_t A_num_cols, const int64_t B_num_rows, const int64_t B_num_cols)
    {

        cusparseHandle_t     handle = NULL;
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        void*                dBuffer    = NULL;
        size_t               bufferSize = 0;

        auto   C_num_rows      = A_num_rows;
        auto   C_num_cols      = B_num_cols;

        auto   A_nnz           = dA_values.size(0);
        auto   ldb             = B_num_cols;
        auto   ldc             = A_num_rows;
        double alpha           = 1.0f;
        double beta            = 0.0f;

        torch::Tensor dC = torch::zeros({C_num_rows, C_num_cols}, dB.options());

        CHECK_CUSPARSE( cusparseCreate(&handle) );
        int version;
        cusparseGetVersion(handle, &version);
        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                          dA_csrOffsets.data_ptr<int64_t>(), dA_columns.data_ptr<int64_t>(),
                                          dA_values.data_ptr<double>(),
                                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
        // Create dense matrix B
        CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_cols, B_num_rows, ldb, dB.data_ptr<double>(),
                                            CUDA_R_64F, CUSPARSE_ORDER_COL) )
        // Create dense matrix C
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, dC.data_ptr<double>(),
                                            CUDA_R_64F, CUSPARSE_ORDER_COL) )

        // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                     handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_TRANSPOSE,
                                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_CSRMM_ALG1, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        // execute SpMM
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_TRANSPOSE,
                                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_CSRMM_ALG1, dBuffer) )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )

        return dC;
    }


    /*
        Sparse versus dense matrix multiplication using CUSPARSE.
        The "__" indicates that this function is called internally, from conjugate_gradient(). This is because
        we make use of global variables that are not passed as arguments.
     */
    torch::Tensor __cusparse_SpMV(
                            torch::Tensor& dB,
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

    torch::Tensor cusparse_SpMM(
                            const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            const torch::Tensor& dB,
                            const int64_t A_num_rows, const int64_t A_num_cols, const int64_t B_num_rows, const int64_t B_num_cols)
    {
        {
            if(dB.dtype() == torch::kFloat32) {
                return floatSpMM(dA_csrOffsets, dA_columns, dA_values, dB, A_num_rows, A_num_cols, B_num_rows, B_num_cols);
            }
            else if(dB.dtype() == torch::kFloat64) {
                return doubleSpMM(dA_csrOffsets, dA_columns, dA_values, dB, A_num_rows, A_num_cols, B_num_rows, B_num_cols);
            }
            else {
                fprintf(stderr, "error: %s: cusparse_SpMM type undexpected at line %d\n", __FILE__,       \
                        __LINE__);                                                     \
                exit(1);
            }
        }
    }

    /*
        Sparse versus Dense Matrix Multiplication.
        This function is externally called from Python.
    */
    torch::Tensor cusparse_SpMV(
                            const torch::Tensor& dA_csrOffsets,
                            const torch::Tensor& dA_columns,
                            const torch::Tensor& dA_values,
                            torch::Tensor& dB,
                            const int64_t dim_i,
                            const int64_t dim_j)
    {
        if(dB.dtype() == torch::kFloat32) {
            return floatSpMV(dA_csrOffsets, dA_columns, dA_values, dB, dim_i, dim_j);
        }
        else if(dB.dtype() == torch::kFloat64) {
            return doubleSpMV(dA_csrOffsets, dA_columns, dA_values, dB, dim_i, dim_j);
        }
        else {
            fprintf(stderr, "error: %s: cusparse_SpMV type undexpected at line %d\n", __FILE__,       \
                    __LINE__);                                                     \
            exit(1);
        }
    }



    std::vector<torch::Tensor> conjugate_gradient(torch::Tensor csr_values, torch::Tensor csr_cols, torch::Tensor csr_rows, int64_t csr_dim0, int64_t csr_dim1, int64_t nnz,
                            torch::Tensor y, torch::Tensor x, torch::Tensor rtol, torch::Tensor atol, torch::Tensor max_iter, bool trj) {

        if(trj) {
            std::cout << "Trajectory tracing not supported. Return only final values" << std::endl;
        }

        CHECK_CUSPARSE( cusparseCreate(&globalHandle) )

        // Create sparse matrix A in CSR format
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

        // CREATE HANDLE AND CSR MATRIX REPRESENTATION
        torch::Tensor sum_y = torch::sum(torch::mul(y,y), -1);
        torch::Tensor atol_sq = atol * atol;
        torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
        torch::Tensor lin_x = __cusparse_SpMV(x, csr_dim0, csr_dim1);
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
            torch::Tensor dy = __cusparse_SpMV(dx, csr_dim0, csr_dim1);
            function_evaluations += not_finished;
            auto dx_dy = torch::sum(residual * dy, -1, TRUE);
            auto step_size = nan_division(residual_squared, dx_dy); // Account for nan values
            step_size = torch::mul(step_size, not_finished);
            x += (step_size * dx);
            if(it_counter % 50 == 0) {
                residual = y - __cusparse_SpMV(x, csr_dim0, csr_dim1);
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


        // Create result tensor with the following variables: x, residual, iterations, function_evaluations, converged, diverged}
        return {x, residual, torch::squeeze(iterations, -1), torch::squeeze(function_evaluations, -1),
        torch::squeeze(converged, -1), torch::squeeze(diverged, -1)};
    }


    /*
        Creates a python module that can be imported in python.
    */
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("conjugate_gradient", &conjugate_gradient, "Conjugate gradient function");
      m.def("cusparse_SpMM", &cusparse_SpMM, "Sparse(CSR) times dense matrix multiplication on CUSPARSE");
      m.def("cusparse_SpMV", &cusparse_SpMV, "Sparse(CSR) times dense vector multiplication on CUSPARSE");
    }

    /*
        Allows for JIT to track this functions.
    */
    TORCH_LIBRARY(phi_torch_cuda, m) {
      m.def("conjugate_gradient", &conjugate_gradient);
      m.def("cusparse_SpMV", &cusparse_SpMV);
      m.def("cusparse_SpMM", &cusparse_SpMM);
    }
}