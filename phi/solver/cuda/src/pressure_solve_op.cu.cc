
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>

using namespace std;

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

__global__ void calcZ_v4(const int *dimensions, const int dimProduct, const int maxDataPerRow, const signed char *laplaceMatrix, const float *p, float *z) {
    extern __shared__ int diagonalOffsets[];

    // Build diagonalOffsets on the first thread of each block and write it to shared memory
    if(threadIdx.x == 0) {
        const int diagonal = maxDataPerRow / 2;
        diagonalOffsets[diagonal] = 0;
        int factor = 1;

        for(int i = 0, offset = 1; i < diagonal; i++, offset++) {
            diagonalOffsets[diagonal - offset] = -factor;
            diagonalOffsets[diagonal + offset] = factor;
            factor *= dimensions[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dimProduct) {
        const int diagonal = row * maxDataPerRow;
        float tmp = 0;
        for(int i = diagonal; i < diagonal + maxDataPerRow; i++) {
            // when accessing out of bound memory in p, laplaceMatrix[i] is always zero. So no illegal mem-access will be made.
            // Anyway, if this causes problems add this:
            // if(row + offsets[i - diagonal] >= 0 && row + offsets[i - diagonal] < dimProduct)
            tmp += (signed char)laplaceMatrix[i] * p[row + diagonalOffsets[i - diagonal]]; // No modulo here (as the general way in the thesis suggests)
        }
        z[row] = tmp;
    }
}

__global__ void checkResiduum(const int dimProduct, const float* r, const float threshold, bool *threshold_reached) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dimProduct; row += blockDim.x * gridDim.x) {
        if (r[row] >= threshold) {
          *threshold_reached = false;
          break;
        }
    }
}

__global__ void initVariablesWithGuess(const int dimProduct, const float *divergence, float* A_times_x_0, float *p, float *r, bool *threshold_reached) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dimProduct) {
        float tmp = divergence[row] - A_times_x_0[row];
        p[row] = tmp;
        r[row] = tmp;

    }
    if(row == 0) *threshold_reached = false;
}

void LaunchPressureKernel(const int* dimensions, const int dimProduct, const int dim_size,
                          const signed char *laplaceMatrix,
                          float* p, float* z, float* r, float* divergence, float* x,
                          const float *oneVector,
                          bool* threshold_reached,
                          const float accuracy,
                          const int max_iterations,
                          const int batch_size,
                          int* iterations_gpu) {
//       printf("Address of laplaceMatrix is %p\n", (void *)laplaceMatrix);
//       printf("Address of oneVector is %p\n", (void *)oneVector);
//       printf("Address of x is %p\n", (void *)x);
//       printf("Address of p is %p\n", (void *)p);
//       printf("Address of z is %p\n", (void *)z);
//       printf("Address of r is %p\n", (void *)r);
//       printf("Address of divergence is %p\n", (void *)divergence);

    cublasHandle_t blasHandle;
    cublasCreate_v2(&blasHandle);
    cublasSetPointerMode_v2(blasHandle, CUBLAS_POINTER_MODE_HOST);

    // CG helper variables variables init
    float *alpha = new float[batch_size], *beta = new float[batch_size];
    const float oneScalar = 1.0f;
    bool *threshold_reached_cpu = new bool[batch_size];
    float *p_r = new float[batch_size], *p_z = new float[batch_size], *r_z = new float[batch_size];

    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;

    // Initialize the helper variables
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4, 0, 0);
    gridSize = (dimProduct + blockSize - 1) / blockSize;

    // First calc A * x_0, save result to z:
    for(int i = 0; i < batch_size; i++) {
        calcZ_v4<<<gridSize, blockSize, dim_size * 2 + 1>>>(dimensions,
                                                            dimProduct,
                                                            dim_size * 2 + 1,
                                                            laplaceMatrix,
                                                            x + i * dimProduct,
                                                            z + i * dimProduct);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess, 0, 0);
    gridSize = (dimProduct + blockSize - 1) / blockSize;

    // Second apply result to the helper variables
    for(int i = 0; i < batch_size; i++) {
        int offset = i * dimProduct;
        initVariablesWithGuess<<<gridSize, blockSize>>>(dimProduct,
                                                        divergence + offset,
                                                        z + offset,
                                                        p + offset,
                                                        r + offset,
                                                        threshold_reached + i);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    // Init residuum checker variables
    CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcZ_v4, 0, 0);
    gridSize = (dimProduct + blockSize - 1) / blockSize;

    // Do CG-Solve
    int checker = 1;
    int iterations = 0;
    for (; iterations < max_iterations; iterations++) {
        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            calcZ_v4<<<gridSize, blockSize, dim_size * 2 + 1>>>(dimensions, dimProduct, dim_size * 2 + 1, laplaceMatrix, p + i * dimProduct, z + i * dimProduct);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            cublasSdot_v2(blasHandle, dimProduct, p + i * dimProduct, 1, r + i * dimProduct, 1, p_r + i);
            cublasSdot_v2(blasHandle, dimProduct, p + i * dimProduct, 1, z + i * dimProduct, 1, p_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            alpha[i] = p_r[i] / p_z[i];
            cublasSaxpy_v2(blasHandle, dimProduct, alpha + i, p + i * dimProduct, 1, x + i * dimProduct, 1);

            alpha[i] = -alpha[i];
            cublasSaxpy_v2(blasHandle, dimProduct, alpha + i, z + i * dimProduct, 1, r + i * dimProduct, 1);

        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Check the residuum every 5 steps to keep memcopys between H&D low
        // Tests have shown, that 5 is a good avg trade-of between memcopys and extra computation and increases the performance
        if (checker % 5 == 0) {
            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                // Use fewer occupancy here, because in most cases residual will be to high and therefore
                checkResiduum<<<8, blockSize>>>(dimProduct, r + i * dimProduct, accuracy, threshold_reached + i);
            }
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            bool done = true;
            for(int i = 0; i < batch_size; i++) {
                if (!threshold_reached_cpu[i]) {
                    done = false;
                    break;
                }
            }
            if(done){
                iterations++;
                break;
            }
            CUDA_CHECK_RETURN(cudaMemset(threshold_reached, 1, sizeof(bool) * batch_size));
        }
        checker++;

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            cublasSdot_v2(blasHandle, dimProduct, r + i * dimProduct, 1, z + i * dimProduct, 1, r_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            beta[i] = -r_z[i] / p_z[i];
            cublasSscal_v2(blasHandle, dimProduct, beta + i, p + i * dimProduct, 1);
            cublasSaxpy_v2(blasHandle, dimProduct, &oneScalar, r + i * dimProduct, 1, p + i * dimProduct, 1);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    delete[] alpha, beta, threshold_reached_cpu, p_r, p_z, r_z;
//    printf("I: %i\n", iterations);

    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}
