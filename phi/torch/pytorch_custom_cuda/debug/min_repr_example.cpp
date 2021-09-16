#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h> // SpMM, SpMV

void foo() {
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *gpu_vals;
    cusparseDnMatDescr_t matA;
    cusparseDnVecDescr_t vecX;

    cudaMalloc((void**) &gpu_vals, 4 * sizeof(float));
    cudaMemcpy(gpu_vals, vals, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cusparseCreateDnMat(&matA, 2, 2, 2, gpu_vals, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnVec(&vecX, 4, gpu_vals, CUDA_R_32F);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("foo", &foo, "example function");
}