// #include <ATen/cuda/CUDAContext.h>
// #include <torch/extension.h>
// #include <cstdio>
// #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
// #include <cuda.h>
// #include <cublas_v2.h>
// #include <cusparse.h>
// #include <vector>
// #include "phi_torch_cuda.hpp"
// #include <ctime>
// #include <chrono>

// #define TRUE 1
// #define FALSE 0

// namespace phi_torch_cuda {
//     /*
//         The following variables are public.

//         They are instantiated at the beginning of the call to conjugate_gradient() and released at the end. They are
//         used in the conjugate_gradient() sub-functions that perform cuSPARSE operations.
//     */
//     void *globalBuffer = NULL;
//     size_t globalBufferSize = 0;
//     cusparseHandle_t globalHandle = NULL;
//     cusparseSpMatDescr_t globalSparseMatrixA;
//     torch::Tensor dC;
//     cusparseDnVecDescr_t dC_cusparse;

//     void allocateBuffer(size_t newSize) {
//         if(newSize > globalBufferSize) {
//             CHECK_CUDA( cudaFree(globalBuffer) )
//             CHECK_CUDA( cudaMalloc(&globalBuffer, newSize) )
//             globalBufferSize = newSize;
//         }
//     }



// std::vector<torch::Tensor>
// conjugate_gradient(
//     torch::Tensor csr_values, 
//     torch::Tensor csr_cols, 
//     torch::Tensor csr_rows, 
//     int64_t csr_dim0, 
//     int64_t csr_dim1, 
//     int64_t nnz,
//     torch::Tensor y, 
//     torch::Tensor x, 
//     torch::Tensor rtol, 
//     torch::Tensor atol, 
//     torch::Tensor max_iter, 
//     bool trj)
// {

//         if(trj) {
//             std::cout << "Trajectory tracing not supported. Return only final values" << std::endl;
//         }

//         CHECK_CUSPARSE( cusparseCreate(&globalHandle) )

//         // Create sparse matrix A in CSR format
//         dC = torch::empty({csr_dim0}, x.options());
//         if(csr_values.dtype() == torch::kFloat32) {
//             CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
//                                               csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
//                                               csr_values.data_ptr<float>(),
//                                               CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
//                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
//             CHECK_CUSPARSE( cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<float>(), CUDA_R_32F) )
//             CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
//                                             csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
//                                             csr_values.data_ptr<float>(),
//                                             CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
//                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
//         }
//         else if(csr_values.dtype() == torch::kFloat64) {
//             CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
//                                               csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
//                                               csr_values.data_ptr<double>(),
//                                               CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
//                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
//             CHECK_CUSPARSE( cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<double>(), CUDA_R_64F) )
//             CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
//                                             csr_rows.data_ptr<int64_t>(), csr_cols.data_ptr<int64_t>(),
//                                             csr_values.data_ptr<double>(),
//                                             CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
//                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
//         }

//         // CREATE HANDLE AND CSR MATRIX REPRESENTATION
//         torch::Tensor sum_y = torch::sum(torch::mul(y,y), -1);
//         torch::Tensor atol_sq = atol * atol;
//         torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
//         torch::Tensor lin_x = __cusparse_SpMV(x, csr_dim0, csr_dim1);
//         auto residual = y - lin_x;
//         auto dx = residual;
//         int64_t it_counter = 0;
//         auto residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
//         auto rsq0 = residual_squared;
//         auto diverged = torch::any(~x.isfinite(), -1, TRUE);
//         auto iterations = torch::zeros_like(diverged, x.options());
//         auto function_evaluations = torch::ones_like(iterations, x.options());
//         auto converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
//         auto finished = converged | diverged | (iterations >= max_iter);
//         auto not_finished = ~finished;
//         auto dummy_ones = torch::ones_like(finished, torch::kCUDA);


//         while(!torch::equal(finished, dummy_ones)) {
//             it_counter += 1;
//             iterations = iterations + not_finished;
//             torch::Tensor dy = __cusparse_SpMV(dx, csr_dim0, csr_dim1);
//             function_evaluations += not_finished;
//             auto dx_dy = torch::sum(residual * dy, -1, TRUE);
//             auto step_size = nan_division(residual_squared, dx_dy); // Account for nan values
//             step_size = torch::mul(step_size, not_finished);
//             x += (step_size * dx);
//             if(it_counter % 50 == 0) {
//                 residual = y - __cusparse_SpMV(x, csr_dim0, csr_dim1);
//                 function_evaluations += 1;
//             }
//             else {
//                 residual = (residual - step_size * dy);
//             }
//             auto residual_squared_old = residual_squared;
//             residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
//             dx = residual + (nan_division(residual_squared, residual_squared_old)) * dx; // Account for nan values
//             diverged = torch::any(residual_squared / rsq0 > 100) & (iterations >= 8);
//             converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
//             finished = converged | diverged | (iterations >= max_iter);
//             not_finished = ~finished;
//         }

//         // CLOSE HANDLE AND CSR MATRIX REPRESENTATION
//         CHECK_CUSPARSE( cusparseDestroySpMat(globalSparseMatrixA) )
//         CHECK_CUSPARSE( cusparseDestroy(globalHandle) )
//         CHECK_CUSPARSE( cusparseDestroyDnVec(dC_cusparse) )
//         CHECK_CUDA( cudaFree(globalBuffer) )


//         // Create result tensor with the following variables: x, residual, iterations, function_evaluations, converged, diverged}
//         return {x, residual, torch::squeeze(iterations, -1), torch::squeeze(function_evaluations, -1),
//         torch::squeeze(converged, -1), torch::squeeze(diverged, -1)};
//     }


//     /*
//         Creates a python module that can be imported in python.
//     */
//     PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//       m.def("conjugate_gradient", &conjugate_gradient, "Conjugate gradient function");
//       m.def("cusparse_SpMM", &cusparse_SpMM, "Sparse(CSR) times dense matrix multiplication on CUSPARSE");
//       m.def("cusparse_SpMV", &cusparse_SpMV, "Sparse(CSR) times dense vector multiplication on CUSPARSE");
//     }

//     /*
//         Allows for JIT to track this functions.
//     */
//     TORCH_LIBRARY(phi_torch_cuda, m) {
//       m.def("conjugate_gradient", &conjugate_gradient);
//       m.def("cusparse_SpMV", &cusparse_SpMV);
//       m.def("cusparse_SpMM", &cusparse_SpMM);
//     }
// }


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

static void 
CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(EXIT_FAILURE);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

// set each vector element to a specified value
__global__ static void
setVectorVal_kernel(
    double* const __restrict__ x,
    const uint32_t vecLength,
    const double value)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vecLength) {
        x[idx] = value;
    }
}

// MODE 0: y = Ax
// MODE 1: y = b - Ax
// MODE 2: y = b - Ax;    r0_t = y
__global__ static void
ybAxL_CSR_kernel0(
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
        y[row] = sum;
        row += stride;
    }
}


__global__ static void
ybAxL_CSR_kernel2(
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
        y[row] = b[row] - sum;
        r0_t[row] = y[row];
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


namespace phi_torch_cuda {

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
        CUDA_CHECK_RETURN(cudaFree(globalBuffer));
        CUDA_CHECK_RETURN(cudaMalloc(&globalBuffer, newSize));
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
__cusparse_SpMV(
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

std::vector<torch::Tensor>
bi_conjugate_gradient(
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

    printf("----------- 1 -----------\n");
    const int L = 2;    // order of the polynomial
    // Initialize cuBLAS library handle and associate streams
    cublasHandle_t cuBLASHandle[L+3];
    cudaStream_t cuStream[L+3];
    for(int k = 0; k < L + 3; ++k) {
        CUDA_CHECK_RETURN(cudaStreamCreate(&cuStream[k]));
        cublasCreate(&cuBLASHandle[k]);
        cublasSetStream(cuBLASHandle[k], cuStream[k]);
    }


    printf("----------- 2 -----------\n");
    double* A = csr_values.data_ptr<double>();
    double* x = xTen.data_ptr<double>();
    double* b = yTen.data_ptr<double>();

    int64_t* ColIdxSp = csr_cols.data_ptr<int64_t>();       // column index of each non-zero element 
    int64_t* RowIdxSpPTR = csr_rows.data_ptr<int64_t>();    // global indices of first non-zero elements in every row 

    int64_t vecLength = csr_dim1;
    int64_t iterMax = 1000;
    double accuracy = 1e-9;
    printf("vecLength: %ld \n", vecLength);

    printf("----------- 3 -----------\n");
    // Allocate managed memory
    double *r0_t, *r, *u;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r0_t, sizeof(double) * vecLength));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&r, sizeof(double) * vecLength * (L + 1)));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&u, sizeof(double) * vecLength * (L + 1)));

    printf("----------- 4 -----------\n");
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

    printf("----------- 5 -----------\n");
    // set x[k] = 1.0  (x0 arbitrary (initial guess)
    setVectorVal_kernel<<<gridDim, blockDim, 0, cuStream[0]>>>(x, vecLength, 1.0);
    
    // set u[k] = 0.0  (u0 = 0)
    setVectorVal_kernel<<<gridDim, blockDim, 0, cuStream[1]>>>(u, vecLength, 0.0);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // r0 = b - Ax0; r0_t[k] = r[k]   (MODE 2)
    ybAxL_CSR_kernel2<<<gridDimMult, blockDim>>>(r, r0_t, b, A, x, ColIdxSp, RowIdxSpPTR, vecLength);

    printf("----------- 6 -----------\n");
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

    printf("----------- 7 -----------\n");
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&rho_old, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&rho_new, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&alpha, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&beta, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&omega, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&gamma_scalar, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&HELPER, sizeof(double) * 3));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&b_2norm, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&residual_k, sizeof(double) * 1));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&residual_relative, sizeof(double) * 1));

    rho_old[0] = 1.0;  // rho   = 1
    rho_new[0] = 1.0;  // rho   = 1
    alpha[0]   = 1.0;  // alpha = 1
    omega[0]   = 1.0;  // omega = 1
 
    printf("----------- 8 -----------\n");
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cublasDnrm2(cuBLASHandle[0], vecLength, b, 1, b_2norm);        // compute 2-norm of the right hand side (b)
    
    printf("----------- 9 -----------\n");
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cublasDnrm2(cuBLASHandle[1], vecLength, r, 1, residual_k);
    
    printf("----------- 10 -----------\n");
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    residual_relative[0] = residual_k[0] / b_2norm[0];
    printf("----------- 11 -----------\n");
    
    size_t iteration = 0;

    printf("----------- 12 -----------\n");
    while((residual_relative[0] > accuracy) && (iteration < iterMax)) {

        rho_old[0] = -omega[0] * rho_old[0];

        // Bi-CG part
        for (int j=0; j<=L-1; ++j) {
            size_t offset_j = vecLength * j;
            cublasDdot(cuBLASHandle[0], vecLength, &r[offset_j], 1, r0_t, 1, rho_new);     // compute rho_new
            beta[0] = alpha[0] * rho_new[0] / rho_old[0];
            rho_old[0] = rho_new[0];

            // parallelize among streams
            for (int i=0; i<=j; ++i) {
                size_t offset_i = vecLength * i;
                DAXPY1L_kernel<<<gridDimMult, blockDim, 0, cuStream[i]>>>(&u[offset_i], &r[offset_i], beta, vecLength);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            ybAxL_CSR_kernel0<<<gridDimMult, blockDim>>>(
                &u[offset_j+vecLength], nullptr, nullptr, A, &u[offset_j], ColIdxSp, RowIdxSpPTR, vecLength);
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            cublasDdot(cuBLASHandle[0], vecLength, &u[offset_j+vecLength], 1, r0_t, 1, gamma_scalar);
            alpha[0] = rho_old[0] / gamma_scalar[0];

            HELPER[0] = -alpha[0];
            for (int i=0; i<=j; ++i) {
                size_t offset_i = vecLength * i;
                // y[k] = y[k] - alpha * x[k]
                cublasDaxpy(cuBLASHandle[0], vecLength, HELPER, &u[offset_i+vecLength], 1, &r[offset_i], 1);
            }

            ybAxL_CSR_kernel0<<<gridDimMult, blockDim, 0, cuStream[0]>>>(
                &r[offset_j+vecLength], nullptr, nullptr, A, &r[offset_j], ColIdxSp, RowIdxSpPTR, vecLength);

            // x[k] = x[k] + alpha * u[k]
            HELPER[0] = alpha[0];
            cublasDaxpy(cuBLASHandle[1], vecLength, HELPER, &u[0], 1, x, 1);
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }

        // MR part
        for (int j=1; j<=L; ++j) {
            size_t offset_j = vecLength * j;
            for (int i=1; i<=j-1; ++i) {
                size_t offset_i = vecLength * i;
                cublasDdot(cuBLASHandle[0], vecLength, &r[offset_j], 1, &r[offset_i], 1, &tau[i][j]);
                tau[i][j] = tau[i][j] / sigma[i];
                HELPER[0] = -tau[i][j];              // copy just to use it as Managed device memory

                // r_j[k] -= tau_i_j * r_i[k]
                cublasDaxpy(cuBLASHandle[0], vecLength, HELPER, &r[offset_i], 1, &r[offset_j], 1);
            }

            cublasDdot(cuBLASHandle[0], vecLength, &r[offset_j], 1, &r[offset_j], 1, &sigma[j]);
            cublasDdot(cuBLASHandle[1], vecLength, &r[offset_j], 1, &r[0], 1, &gamma_p[j]);
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
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

        cublasDaxpy(cuBLASHandle[0], vecLength, &HELPER[0], &r[0], 1, x, 1); // x[k] += gamma[1] * r[k]
        cublasDaxpy(cuBLASHandle[0], vecLength, &HELPER[1], &r[offset_L], 1, &r[0], 1);
        cublasDaxpy(cuBLASHandle[1], vecLength, &HELPER[2], &u[offset_L], 1, &u[0], 1);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // BLAS2_GEMV or BLAS3_GEMM
        for (int j=1; j<=L-1; ++j) {
            size_t offset = vecLength * j;
            HELPER[0] = -gamma[j]; 
            HELPER[1] = gamma_pp[j]; 
            HELPER[2] = -gamma_p[j]; 

            cublasDaxpy(cuBLASHandle[0], vecLength, &HELPER[0], &u[offset], 1, &u[0], 1);
            cublasDaxpy(cuBLASHandle[1], vecLength, &HELPER[1], &r[offset], 1, x, 1);
            cublasDaxpy(cuBLASHandle[2], vecLength, &HELPER[2], &r[offset], 1, &r[0], 1);
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }

        cublasDnrm2(cuBLASHandle[0], vecLength, &r[0], 1, residual_k);
        residual_relative[0] = residual_k[0] / b_2norm[0];

        iteration += L;
    }

    for(int k = 0; k < L + 3; ++k) {
        cublasDestroy(cuBLASHandle[k]);
        CUDA_CHECK_RETURN(cudaStreamDestroy(cuStream[k]));
    }

    CUDA_CHECK_RETURN(cudaFree(r0_t));
    CUDA_CHECK_RETURN(cudaFree(r));
    CUDA_CHECK_RETURN(cudaFree(u));
    CUDA_CHECK_RETURN(cudaFree(rho_old));
    CUDA_CHECK_RETURN(cudaFree(rho_new));
    CUDA_CHECK_RETURN(cudaFree(alpha));
    CUDA_CHECK_RETURN(cudaFree(beta));
    CUDA_CHECK_RETURN(cudaFree(omega));
    CUDA_CHECK_RETURN(cudaFree(gamma_scalar));
    CUDA_CHECK_RETURN(cudaFree(HELPER));
    CUDA_CHECK_RETURN(cudaFree(b_2norm));
    CUDA_CHECK_RETURN(cudaFree(residual_k));
    CUDA_CHECK_RETURN(cudaFree(residual_relative));

    // Dummy plugs to test main functionality
    torch::Tensor sum_y = torch::sum(torch::mul(yTen,yTen), -1);
    torch::Tensor atol_sq = atol * atol;
    torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
    // torch::Tensor lin_x = __cusparse_SpMV(xTen, csr_dim0, csr_dim1);
    // auto residual = yTen - lin_x;
    auto residual = yTen;
    // auto dx = residual;
    // int64_t it_counter = 0;
    auto residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
    // auto rsq0 = residual_squared;
    auto diverged = torch::any(~xTen.isfinite(), -1, TRUE);
    auto iterations = torch::zeros_like(diverged, xTen.options());
    auto function_evaluations = torch::ones_like(iterations, xTen.options());
    auto converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
    auto finished = converged | diverged | (iterations >= max_iter);
    // auto not_finished = ~finished;
    // auto dummy_ones = torch::ones_like(finished, torch::kCUDA);


    // Create result tensor with the following variables:
    //     {x, residual, iterations, function_evaluations, converged, diverged}
    return {
        xTen,
        residual,
        torch::squeeze(iterations, -1),                 // iterations
        torch::squeeze(function_evaluations, -1),       // function_evaluations
        torch::squeeze(converged, -1),                  // converged
        torch::squeeze(diverged, -1)                    // diverged
    };
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