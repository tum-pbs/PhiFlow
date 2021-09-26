#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <chrono>

using namespace std;

#define POS(size1, i, j) (i * size2 + j)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void print_matrix(int size1, int size2, float* matrix) {
    cout << endl;
    for(int i = 0; i < size1; i++) {
        for(int j = 0; j < size2; j++) {
            cout << mat[POS(size1, i, j)] << " ";

        }
        cout << endl;
    }
}

float* create_matrix(int size1, int size2, float sparsity) {
    srand(time(NULL));
    float* mat = malloc(sizeof(float) * size1 * size2)
    for(int i = 0; i < size1; i++) {
        for(int j = 0; j < size2; j++) {
            if (rand() % 100 > sparsity) {
                mat[POS(size1, i, j)] = float(rand() % 255);
            }
            else {
                mat[POS(size1, i, j)] = 0.0;
            }

        }
    }
    return mat;
}

void convert_to_csr(float* matrix, int size1, int size2,
                    vector csrOffsets, vector csrcolumns, vector csrvalues) {
        auto elems = 0;
        csrOffsets.push_back(elems);
        for(int i = 0; i < size1; i++) {
            for(int j = 0; j < size2; j++) {
                auto val = matrix[POS(size1, i, j)];
                if(val != 0.0) {
                    elems++;
                    csrcolumns.push_back(j)
                    csrvalues.push_back(val)
                }
            }
            csrOffsets.push_back(elems);
        }
}

void spmv(auto csrA, auto denseVecB, auto denseVecC, auto mvBuffer) {
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
}

void spmm(auto csrA, auto denseMatB, auto denseMatC, auto mmBuffer) {
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, csrA, denseMatB, &beta, denseMatC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, mmBuffer) )
}

void* create_mm_buffer(auto csrA, auto denseMatB, auto denseMatC) {
    size_t bufferSize = 0;
    void* dBuffer;
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, csrA, denseMatB, &beta, denseMatC, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    return dBuffer;
}

void* create_mv_buffer(auto csrA, auto denseVecB, auto denseVecC) {
    size_t bufferSize = 0;
    void* dBuffer;
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, csrA, denseVecB, &beta, denseVecC, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    return dBuffer;
}

void dense_mat(, auto cols, auto rows, auto ld) {
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, cols, rows, ld, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
}

void dense_vec() {

}

void sparse_csr_mat() {

}

cusparseHandle_t handle = NULL;
auto alpha = 1.0;
auto beta = 0.0;
cusparseSpMatDescr_t csrA;
cusparseDnMatDescr_t denseMatB, denseMatC;
cusparseDnVecDescr_t denseVecB, denseVecC;

int main() {
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    auto sparsity = 0.2;
    for(auto size : {10, 100, 1000}) {
        auto memA = create_matrix(size, size, sparsity);
        auto memB = create_matrix(size, 1, 0.0);
        auto memC = malloc(sizeof(float) * size);

        auto csrA = sparse_csr_mat(memA);
        auto denseMatB = dense_mat(memB);
        auto denseVecB = dense_vec(memB);
        auto denseMatC = dense_mat(memC);
        auto denseVecC = dense_vec(memC);

        auto mvBuffer = create_mv_buffer(csrA, denseVecB, denseVecC);
        auto mmBuffer = create_mm_buffer(csrA, denseMatB, denseMatC);

        auto start = chrono::high_resolution_clock::now();
        spmv(csrA, denseVecB, denseVecC, mvBuffer);
        auto stop = chrono::high_resolution_clock::now();
        auto duration_mv = chrono::duration_cast<microseconds>(stop - start);

        auto start = chrono::high_resolution_clock::now();
        spmm(csrA, denseMatB, denseMatC, mmBuffer);
        auto stop = chrono::high_resolution_clock::now();
        auto duration_mm = chrono::duration_cast<microseconds>(stop - start);
    }
    return 0;
}