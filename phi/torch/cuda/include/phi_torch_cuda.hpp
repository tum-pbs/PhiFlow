#pragma once

static void 
CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(EXIT_FAILURE);
}


// cuBLAS API errors
static const char* _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:         return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:    return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:   return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:   return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:   return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:  return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


static void
CheckCublasErrorAux(const char* file, unsigned line, const char* statement, cublasStatus_t status) {
    if(status == CUBLAS_STATUS_SUCCESS) return;
    std::cerr << statement << " returned " << _cublasGetErrorEnum(status) << "("
              << status << ") at " << file << ":" << line << std::endl;
    exit(EXIT_FAILURE);
}


// cuSPARSE API errors
static const char* _cuSparseGetErrorEnum(cusparseStatus_t error) {
    switch (error) {
        case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_NOT_SUPPORTED:             return "CUSPARSE_STATUS_NOT_SUPPORTED";
    }
    return "<unknown>";
}


static void
CheckCusparseErrorAux(const char* file, unsigned line, const char* statement, cusparseStatus_t status) {
    if(status == CUSPARSE_STATUS_SUCCESS) return;
    std::cerr << statement << " returned " << _cuSparseGetErrorEnum(status) << "("
              << status << ") at " << file << ":" << line << std::endl;

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cerr << statement << " returned " << cudaGetErrorString(cuda_err) << "("
                  << cuda_err << ") at " << file << ":" << line << std::endl;
    }  
    exit(EXIT_FAILURE);
}


#define CHECK_CUDA(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#define CHECK_CUBLAS(value) CheckCublasErrorAux(__FILE__, __LINE__, #value, value)
#define CHECK_CUSPARSE(value) CheckCusparseErrorAux(__FILE__, __LINE__, #value, value)