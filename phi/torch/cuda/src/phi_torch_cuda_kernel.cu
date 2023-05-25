#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <vector>
#include <ctime>
#include <chrono>
#include "phi_torch_cuda.hpp"

#define TRUE 1
#define FALSE 0

// set each vector element to a specified value
__global__ static void
setVectorVal_kernel(
    double* const __restrict__ x,
    const int64_t vecLength,
    const double value)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vecLength) {
        x[idx] = value;
    }
}

// MODE 0: y = Ax
// MODE 1: y = b - Ax
// MODE 2: y = b - Ax;    r0_t = y
template<int Mode>
__global__ static void
ybAxL_CSR_kernel(
    double* const __restrict__ y,
    double* const __restrict__ r0_t,
    const double* const __restrict__ b,
    const double* const __restrict__ A,
    const double* const __restrict__ x,
    const int64_t* const __restrict__ COL_IDX,
    const int64_t* const __restrict__ ROW_IDX_PTR,
    const int64_t vecLength)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    while(row < vecLength) {
        const int64_t row_start = ROW_IDX_PTR[row];
        const int64_t row_end = ROW_IDX_PTR[row+1];
        double sum = 0.0;
        for(int64_t element = row_start; element < row_end; ++element) {
            sum += A[element] * x[COL_IDX[element]];
        }
        switch(Mode) {
            case 0:
                y[row] = sum;
                break;
            case 1:
                y[row] = b[row] - sum;
                break;
            case 2:
                y[row] = b[row] - sum;
                r0_t[row] = y[row];
                break;
        }
        row += stride;
    }
}


// y[k] = x[k] - beta * y[k]
__global__ static void
DAXPY1L_kernel(
    double* const __restrict__ y,
    const double* const __restrict__ x,
    const double* const __restrict__ beta,   // scalar coefficient
    const int64_t vecLength)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    while(row < vecLength) {
        y[row] = x[row] - beta[0] * y[row];
        row += stride;
    }
}


// The following variables are public.
// They are instantiated at the beginning of the call to conjugate_gradient() and released at the end. They are
// used in the conjugate_gradient() sub-functions that perform cuSPARSE operations.
void *globalBuffer = NULL;
size_t globalBufferSize = 0;
cusparseHandle_t globalHandle = NULL;
cusparseSpMatDescr_t globalSparseMatrixA;
torch::Tensor dC;
cusparseDnVecDescr_t dC_cusparse;

void allocateBuffer(size_t newSize) {
    if(newSize > globalBufferSize) {
        CHECK_CUDA(cudaFree(globalBuffer));
        CHECK_CUDA(cudaMalloc(&globalBuffer, newSize));
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


// Sparse versus dense matrix multiplication using CUSPARSE.
// The "__" indicates that this function is called internally, from conjugate_gradient(). This is because
// we make use of global variables that are not passed as arguments.
torch::Tensor
__cusparse_SpMV(torch::Tensor& dB, const int64_t dim_i, const int64_t dim_j) {
    cusparseDnVecDescr_t vecX;
    size_t               bufferSize = 0;

    dB = dB.view({dim_j});

    // Create dense vector X
    double alpha           = 1.0f;
    double beta            = 0.0f;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<double>(), CUDA_R_64F));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_64F,
        CUSPARSE_CSRMV_ALG2, &bufferSize)
    );
    allocateBuffer(bufferSize);
    // execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(
        globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_64F,
        CUSPARSE_CSRMV_ALG2, globalBuffer)
    );

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    dB = dB.view({1, dim_j});
    dC = dC.view({1, dim_i});
    return dC;
}

// Solve the system of linear equations A · x = y.
// This method need not provide a gradient for the operation.

// Args:
//     method: Which algorithm to use. One of `('auto', 'CG', 'CG-adaptive', 'BiCGStabL')`.
//     lin: Linear operation. One of
//         * sparse/dense matrix valid for all instances
//         * tuple/list of sparse/dense matrices for varying matrices along batch, must have the same nonzero locations.
//         * linear function A(x), must be called on all instances in parallel
//     y: target result of A * x. 2nd order tensor (batch, vector) or list of vectors.
//     x0: Initial guess of size (batch, parameters)
//     rtol: Relative tolerance of size (batch,)
//     atol: Absolute tolerance of size (batch,)
//     max_iter: Maximum number of iterations of size (batch,)
//     trj: Whether to record and return the optimization trajectory as a `List[SolveResult]`.

// Returns:
//     result: `SolveResult` or `List[SolveResult]`, depending on `trj`.

std::vector<torch::Tensor>
cuda_bi_conjugate_gradient(
    const int64_t L,                // Order of the polynomial
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
    printf("----------- Entered into C++ code -----------\n");

    if(trj) {
        std::cout << "Trajectory tracing not supported. Return only final values" << std::endl;
    }

    // Initialize cuBLAS library handle and associate streams
    cublasHandle_t cuBLASHandle;
    // cudaStream_t cuStream;
    // CHECK_CUDA(cudaStreamCreate(&cuStream));
    CHECK_CUBLAS(cublasCreate(&cuBLASHandle));
    // CHECK_CUBLAS(cublasSetStream(cuBLASHandle, cuStream));

    CHECK_CUSPARSE(cusparseCreate(&globalHandle));

    // Create sparse matrix A in CSR format
    dC = torch::empty({csr_dim0}, xTen.options());
    CHECK_CUSPARSE(cusparseCreateCsr(
        &globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
        csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(), csr_values.data_ptr<double>(),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)
    );
    CHECK_CUSPARSE(cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<double>(), CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
        csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(), csr_values.data_ptr<double>(),
        CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)
    );


    // CREATE HANDLE AND CSR MATRIX REPRESENTATION
    torch::Tensor sum_y = torch::sum(torch::mul(yTen, yTen), -1);
    torch::Tensor atol_sq = atol * atol;
    torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
    torch::Tensor lin_x = __cusparse_SpMV(xTen, csr_dim0, csr_dim1);
    auto resid_Ten = yTen - lin_x;
    auto dx = resid_Ten;
    auto residual_squared = torch::sum(torch::mul(resid_Ten, resid_Ten), -1, TRUE);
    auto rsq0 = residual_squared;
    auto diverged = torch::any(~xTen.isfinite(), -1, TRUE);
    auto iterations = torch::zeros_like(diverged, xTen.options());
    auto function_evaluations = torch::ones_like(iterations, xTen.options());
    auto converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
    auto finished = converged | diverged | (iterations >= max_iter);
    auto not_finished = ~finished;
    auto dummy_ones = torch::ones_like(finished, torch::kCUDA);

    std::cout << "xTen.size(-1):        " << xTen.size(-1)                              << std::endl;
    std::cout << "max_iter.size(-1):    " << max_iter.size(-1)  << "    " << max_iter   << std::endl;
    std::cout << "atol.size(-1):        " << atol.size(-1)      << "    " << atol       << std::endl;
    std::cout << "rtol.size(-1):        " << rtol.size(-1)      << "    " << rtol       << std::endl;
    std::cout << "converged.size(-1):   " << converged.size(-1) << "    " << converged  << std::endl;
    std::cout << "diverged.size(-1):    " << diverged.size(-1)  << "    " << diverged   << std::endl;
    std::cout << "iterations.size(-1):  " << iterations.size(-1)<< "    " << iterations << std::endl;


    auto residualTen = torch::empty_like(xTen, torch::kCUDA);
    // auto uTen = torch::zeros_like(xTen, torch::kCUDA);

    auto options1 =
        torch::TensorOptions()
            .dtype(torch::kFloat64)
            .device(torch::kCUDA);

    auto residual0Ten = torch::empty({L+1, xTen.size(-1)}, options1);

    double* A = csr_values.data_ptr<double>();
    double* x = xTen.data_ptr<double>();
    double* b = yTen.data_ptr<double>();
    double* res_ptr = residualTen.data_ptr<double>();
    // double* r0_t = residualTen.data_ptr<double>();
    // double* r = residual0Ten.data_ptr<double>();
    // double* u = uTen.data_ptr<double>();

    // std::cout << "Printing the yTen tensor" << yTen << std::endl;

    int64_t* ColIdxSp = csr_cols.data_ptr<int64_t>();       // column index of each non-zero element 
    int64_t* RowIdxSpPTR = csr_rows.data_ptr<int64_t>();    // global indices of first non-zero elements in every row 

    int64_t vecLength = csr_dim1;
    int64_t iterMax = 1000;
    double accuracy = 1e-9;
    printf("vecLength: %ld \n", vecLength);

    // Allocate managed memory
    double *r0_t, *r, *u;

    CHECK_CUDA(cudaMallocManaged((void**)&r0_t, sizeof(double) * vecLength));
    CHECK_CUDA(cudaMallocManaged((void**)&r, sizeof(double) * vecLength * (L + 1)));
    CHECK_CUDA(cudaMallocManaged((void**)&u, sizeof(double) * vecLength * (L + 1)));

    // Size L+1 is used just for indexing convenience to improve readability
    // and 1-to-1 match with the algorithm from the original paper (1993)
    // Just a few elements are wasted in terms of memory, since L is small (L<8)
    // It can be easily reduced down to L if needed by the corresponding index decrease by 1
    double tau[L+1][L+1];    
    double sigma[L+1];
    double gamma[L+1];       // gamma
    double gamma_p[L+1];     // gamma prime
    double gamma_pp[L+1];    // gamma double prime

    // Grid and block parameters for vector management (initialization)
    const int64_t gridMultiplier = 8;      // number of elements in adaptive block (control parameter)
    const int64_t blockDim = 256;
    const int64_t gridDim = (vecLength - 1) / blockDim + 1;
    const int64_t gridDimMult = (vecLength - 1) / (blockDim * gridMultiplier) + 1;

    // set x[k] = 1.0  (x0 arbitrary (initial guess)                        // already computed as first iteration (see above)
    // setVectorVal_kernel<<<gridDim, blockDim, 0>>>(x, vecLength, 1.0);
    // CHECK_CUDA(cudaDeviceSynchronize());
    
    // set u[k] = 0.0  (u0 = 0)
    setVectorVal_kernel<<<gridDim, blockDim, 0>>>(u, vecLength, 0.0);
    CHECK_CUDA(cudaDeviceSynchronize());

    // r0 = b - Ax0; r0_t[k] = r[k]   (MODE 2)
    ybAxL_CSR_kernel<2><<<gridDimMult, blockDim>>>(r, r0_t, b, A, x, ColIdxSp, RowIdxSpPTR, vecLength);
    CHECK_CUDA(cudaDeviceSynchronize());

    double* rho_old;
    double* rho_new;
    double* alpha;
    double* beta;
    double* omega;
    double* gamma_scalar;
    double* HELPER;
    double* b_2norm;
    double* residual_k;
    double* residual_relative;

    CHECK_CUDA(cudaMallocManaged((void**)&rho_old, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&rho_new, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&alpha, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&beta, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&omega, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&gamma_scalar, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&HELPER, sizeof(double) * 3));
    CHECK_CUDA(cudaMallocManaged((void**)&b_2norm, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&residual_k, sizeof(double) * 1));
    CHECK_CUDA(cudaMallocManaged((void**)&residual_relative, sizeof(double) * 1));

    rho_old[0] = 1.0;  // rho   = 1
    rho_new[0] = 1.0;  // rho   = 1
    alpha[0]   = 1.0;  // alpha = 1
    omega[0]   = 1.0;  // omega = 1
 
    CHECK_CUDA(cudaDeviceSynchronize());
    auto bNorm2Torch = torch::sum(torch::mul(yTen, yTen));
    // std::cout << "2-norm of yTen: " << bNorm2Torch << std::endl;
    CHECK_CUBLAS(cublasDnrm2(cuBLASHandle, vecLength, b, 1, b_2norm));        // compute 2-norm of the right hand side (b)
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUBLAS(cublasDnrm2(cuBLASHandle, vecLength, r, 1, residual_k));
    CHECK_CUDA(cudaDeviceSynchronize());
    

    residual_relative[0] = residual_k[0] / b_2norm[0];
    
    uint64_t iterationCounter = 0;

    // while(!torch::equal(finished, dummy_ones)) {
    //     it_counter += 1;
    //     iterations = iterations + not_finished;
    //     torch::Tensor dy = __cusparse_SpMV(dx, csr_dim0, csr_dim1);
    //     function_evaluations += not_finished;
    //     auto dx_dy = torch::sum(resid_Ten * dy, -1, TRUE);
    //     auto step_size = nan_division(residual_squared, dx_dy); // Account for nan values
    //     step_size = torch::mul(step_size, not_finished);
    //     x += (step_size * dx);
    //     if(it_counter % 50 == 0) {
    //         resid_Ten = y - __cusparse_SpMV(x, csr_dim0, csr_dim1);
    //         function_evaluations += 1;
    //     }
    //     else {
    //         resid_Ten = (resid_Ten - step_size * dy);
    //     }
    //     auto residual_squared_old = residual_squared;
    //     residual_squared = torch::sum(torch::mul(resid_Ten, resid_Ten), -1, TRUE);
    //     dx = resid_Ten + (nan_division(residual_squared, residual_squared_old)) * dx; // Account for nan values
    //     diverged = torch::any(residual_squared / rsq0 > 100) & (iterations >= 8);
    //     converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
    //     finished = converged | diverged | (iterations >= max_iter);
    //     not_finished = ~finished;
    // }


    while((residual_relative[0] > accuracy) && (iterationCounter < iterMax)) {
    // while(!torch::equal(finished, dummy_ones)) {

        rho_old[0] = -omega[0] * rho_old[0];

        // Bi-CG part
        for (int j=0; j<=L-1; ++j) {
            size_t offset_j = vecLength * j;
            CHECK_CUBLAS(cublasDdot(cuBLASHandle, vecLength, &r[offset_j], 1, r0_t, 1, rho_new));     // compute rho_new
            CHECK_CUDA(cudaDeviceSynchronize());
            beta[0] = alpha[0] * rho_new[0] / rho_old[0];
            rho_old[0] = rho_new[0];

            // parallelize among streams
            for (int i=0; i<=j; ++i) {
                size_t offset_i = vecLength * i;
                DAXPY1L_kernel<<<gridDimMult, blockDim>>>(&u[offset_i], &r[offset_i], beta, vecLength);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            // MODE 0
            ybAxL_CSR_kernel<0><<<gridDimMult, blockDim>>>(
                &u[offset_j+vecLength], nullptr, nullptr, A, &u[offset_j], ColIdxSp, RowIdxSpPTR, vecLength);
            CHECK_CUDA(cudaDeviceSynchronize());
            function_evaluations += 1;

            CHECK_CUBLAS(cublasDdot(cuBLASHandle, vecLength, &u[offset_j+vecLength], 1, r0_t, 1, gamma_scalar));
            CHECK_CUDA(cudaDeviceSynchronize());
            alpha[0] = rho_old[0] / gamma_scalar[0];

            HELPER[0] = -alpha[0];
            for (int i=0; i<=j; ++i) {
                size_t offset_i = vecLength * i;
                // y[k] = y[k] - alpha * x[k]
                CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, HELPER, &u[offset_i+vecLength], 1, &r[offset_i], 1));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            // MODE 0
            ybAxL_CSR_kernel<0><<<gridDimMult, blockDim>>>(
                &r[offset_j+vecLength], nullptr, nullptr, A, &r[offset_j], ColIdxSp, RowIdxSpPTR, vecLength);
            CHECK_CUDA(cudaDeviceSynchronize());
            function_evaluations += 1;

            // x[k] = x[k] + alpha * u[k]
            HELPER[0] = alpha[0];
            CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, HELPER, &u[0], 1, x, 1));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // MR part
        for (int j=1; j<=L; ++j) {
            size_t offset_j = vecLength * j;
            for (int i=1; i<=j-1; ++i) {
                size_t offset_i = vecLength * i;
                CHECK_CUBLAS(cublasDdot(cuBLASHandle, vecLength, &r[offset_j], 1, &r[offset_i], 1, &tau[i][j]));
                CHECK_CUDA(cudaDeviceSynchronize());
                tau[i][j] = tau[i][j] / sigma[i];
                HELPER[0] = -tau[i][j];              // copy just to use it as Managed device memory

                // r_j[k] -= tau_i_j * r_i[k]
                CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, HELPER, &r[offset_i], 1, &r[offset_j], 1));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            CHECK_CUBLAS(cublasDdot(cuBLASHandle, vecLength, &r[offset_j], 1, &r[offset_j], 1, &sigma[j]));
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUBLAS(cublasDdot(cuBLASHandle, vecLength, &r[offset_j], 1, &r[0], 1, &gamma_p[j]));
            CHECK_CUDA(cudaDeviceSynchronize());
            gamma_p[j] = gamma_p[j] / sigma[j];
        }

        gamma[L] = gamma_p[L];
        omega[0] = gamma[L];

        for (int j=L-1; j>=1; --j) {
            gamma[j] = gamma_p[j];
            for (int i=j+1; i<=L; ++i) {
                gamma[j] -= tau[j][i] * gamma[i];
            }
        }

        for (int j=1; j<=L-1; ++j) {
            gamma_pp[j] = gamma[j+1];
            for (int i=j+1; i<=L-1; ++i) {
                gamma_pp[j] += tau[j][i] * gamma[i+1];
            }
        }

        size_t offset_L = vecLength * L;

        HELPER[0] = gamma[1]; 
        HELPER[1] = -gamma_p[L]; 
        HELPER[2] = -gamma[L]; 

        CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[0], &r[0], 1, x, 1)); // x[k] += gamma[1] * r[k]
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[1], &r[offset_L], 1, &r[0], 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[2], &u[offset_L], 1, &u[0], 1));
        CHECK_CUDA(cudaDeviceSynchronize());

        // BLAS2_GEMV or BLAS3_GEMM
        for (int j=1; j<=L-1; ++j) {
            size_t offset = vecLength * j;
            HELPER[0] = -gamma[j]; 
            HELPER[1] = gamma_pp[j]; 
            HELPER[2] = -gamma_p[j]; 

            CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[0], &u[offset], 1, &u[0], 1));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[1], &r[offset], 1, x, 1));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUBLAS(cublasDaxpy(cuBLASHandle, vecLength, &HELPER[2], &r[offset], 1, &r[0], 1));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        CHECK_CUBLAS(cublasDnrm2(cuBLASHandle, vecLength, &r[0], 1, residual_k));
        CHECK_CUDA(cudaDeviceSynchronize());
        residual_relative[0] = residual_k[0] / b_2norm[0];

        iterationCounter += L;
    }

    // double* iterTenData = iterations.data_ptr<double>();
    // *iterTenData = iterationCounter;

    CHECK_CUBLAS(cublasDcopy(cuBLASHandle, vecLength, &r[0], 1, res_ptr, 1));     // copy raw residual into tensor
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUBLAS(cublasDestroy(cuBLASHandle));

    // CLOSE HANDLE AND CSR MATRIX REPRESENTATION
    CHECK_CUSPARSE(cusparseDestroySpMat(globalSparseMatrixA));
    CHECK_CUSPARSE(cusparseDestroy(globalHandle));
    CHECK_CUSPARSE(cusparseDestroyDnVec(dC_cusparse));
    CHECK_CUDA(cudaFree(globalBuffer));

    CHECK_CUDA(cudaFree(r0_t));
    CHECK_CUDA(cudaFree(r));
    CHECK_CUDA(cudaFree(u));
    CHECK_CUDA(cudaFree(rho_old));
    CHECK_CUDA(cudaFree(rho_new));
    CHECK_CUDA(cudaFree(alpha));
    CHECK_CUDA(cudaFree(beta));
    CHECK_CUDA(cudaFree(omega));
    CHECK_CUDA(cudaFree(gamma_scalar));
    CHECK_CUDA(cudaFree(HELPER));
    CHECK_CUDA(cudaFree(b_2norm));
    CHECK_CUDA(cudaFree(residual_k));
    CHECK_CUDA(cudaFree(residual_relative));


    // Create result tensor with the following variables:
    //     {x, residual, iterations, function_evaluations, converged, diverged}
    return {
        xTen,
        residualTen,
        torch::squeeze(iterations, -1),                 // iterations
        torch::squeeze(function_evaluations, -1),       // function_evaluations
        torch::squeeze(converged, -1),                  // converged
        torch::squeeze(diverged, -1)                    // diverged
    };
}