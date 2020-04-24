
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

__global__ void calcZ_v4(const int *dimensions, const int dim_product, const int maxDataPerRow, const signed char *laplace_matrix, const float *p, float *z) {
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
    if (row < dim_product) {
        const int diagonal = row * maxDataPerRow;
        float tmp = 0;
        for(int i = diagonal; i < diagonal + maxDataPerRow; i++) {
            // when accessing out of bound memory in p, laplace_matrix[i] is always zero. So no illegal mem-access will be made.
            // If this causes problems add :
            // if(row + offsets[i - diagonalOffsets] >= 0 && row + offsets[i - diagonalOffsets] < dim_product)
            tmp += (signed char)laplace_matrix[i] * p[row + diagonalOffsets[i - diagonal]]; // No modulo here (as the general way in the thesis suggests)
        }
        z[row] = tmp;
    }
}

__global__ void checkResiduum(const int dim_product, const float* r, const float threshold, bool *threshold_reached) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x) {
        if (r[row] >= threshold) {
          *threshold_reached = false;
          break;
        }
    }
}

__global__ void initVariablesWithGuess(const int dim_product, const float *divergence, float* A_times_x_0, float *p, float *r, bool *threshold_reached) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim_product) {
        float tmp = divergence[row] - A_times_x_0[row];
        p[row] = tmp;
        r[row] = tmp;

    }
    if(row == 0) *threshold_reached = false;
}

// global blas handle, initialize only once (warning - currently not free'd!)
bool           initBlasHandle = true;
cublasHandle_t blasHandle;

void LaunchPressureKernel(const int* dimensions, const int dim_product, const int dim_size,
                          const signed char *laplace_matrix,
                          float* p, float* z, float* r, float* divergence, float* x,
                          const float *oneVector,
                          bool* threshold_reached,
                          const float accuracy,
                          const int max_iterations,
                          const int batch_size,
                          int* iterations_gpu) 
{
//       printf("Address of laplace_matrix is %p\n", (void *)laplace_matrix);
//       printf("Address of oneVector is %p\n", (void *)oneVector);
//       printf("Address of x is %p\n", (void *)x);
//       printf("Address of p is %p\n", (void *)p);
//       printf("Address of z is %p\n", (void *)z);
//       printf("Address of r is %p\n", (void *)r);
//       printf("Address of divergence is %p\n", (void *)divergence);

    if(initBlasHandle) {
        cublasCreate_v2(&blasHandle);
        cublasSetPointerMode_v2(blasHandle, CUBLAS_POINTER_MODE_HOST);
        initBlasHandle = false;
    }

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
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // First calc A * x_0, save result to z:
    for(int i = 0; i < batch_size; i++) {
        calcZ_v4<<<gridSize, blockSize, dim_size * 2 + 1>>>(dimensions,
                                                            dim_product,
                                                            dim_size * 2 + 1,
                                                            laplace_matrix,
                                                            x + i * dim_product,
                                                            z + i * dim_product);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Second apply result to the helper variables
    for(int i = 0; i < batch_size; i++) {
        int offset = i * dim_product;
        initVariablesWithGuess<<<gridSize, blockSize>>>(dim_product,
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
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Do CG-Solve
    int checker = 1;
    int iterations = 0;
    for (; iterations < max_iterations; iterations++) {
        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            calcZ_v4<<<gridSize, blockSize, dim_size * 2 + 1>>>(dimensions, dim_product, dim_size * 2 + 1, laplace_matrix, p + i * dim_product, z + i * dim_product);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            cublasSdot_v2(blasHandle, dim_product, p + i * dim_product, 1, r + i * dim_product, 1, p_r + i);
            cublasSdot_v2(blasHandle, dim_product, p + i * dim_product, 1, z + i * dim_product, 1, p_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            alpha[i] = 0.;
            if(fabs(p_z[i])>0.) alpha[i] = p_r[i] / p_z[i];
            cublasSaxpy_v2(blasHandle, dim_product, alpha + i, p + i * dim_product, 1, x + i * dim_product, 1);

            alpha[i] = -alpha[i];
            cublasSaxpy_v2(blasHandle, dim_product, alpha + i, z + i * dim_product, 1, r + i * dim_product, 1);

        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // Check the residuum every 5 steps to keep memcopys between H&D low
        // Tests have shown, that 5 is a good avg trade-of between memcopys and extra computation and increases the performance
        if (checker % 5 == 0) {
            for(int i = 0; i < batch_size; i++) {
                if(threshold_reached_cpu[i]) continue;
                // Use fewer occupancy here, because in most cases residual will be to high and therefore
                checkResiduum<<<8, blockSize>>>(dim_product, r + i * dim_product, accuracy, threshold_reached + i);
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
            cublasSdot_v2(blasHandle, dim_product, r + i * dim_product, 1, z + i * dim_product, 1, r_z + i);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < batch_size; i++) {
            if(threshold_reached_cpu[i]) continue;
            beta[i] = -r_z[i] / p_z[i];
            cublasSscal_v2(blasHandle, dim_product, beta + i, p + i * dim_product, 1);
            cublasSaxpy_v2(blasHandle, dim_product, &oneScalar, r + i * dim_product, 1, p + i * dim_product, 1);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    delete[] alpha, beta, threshold_reached_cpu, p_r, p_z, r_z;
//    printf("I: %i\n", iterations);

    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

}
