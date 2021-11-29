#ifndef PYTORCH_CUSTOM_CUDA
#define PYTORCH_CUSTOM_CUDA

#define CHECK_CUDA(call)                                                                \
{                                                                                       \
    cudaError_t err;                                                                    \
    if ((err = (call)) != cudaSuccess) {                                                \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                          \
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err));        \
        exit(1);                                                                        \
    }                                                                                   \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d named %s, %s. At %s %d\n",               \
            err, cusparseGetErrorName(err),                                    \
            cusparseGetErrorString(err),__FILE__, __LINE__);                   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}
#endif