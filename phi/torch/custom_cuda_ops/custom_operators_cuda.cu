#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__global__ void vector_addition_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vec1,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vec2,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out,
    int n) {

    // Get our global thread ID
    int first_thread_on_block = blockIdx.x *blockDim.x * blockIdx.y * blockDim.y;
    int id = first_thread_on_block + threadIdx.x + blockDim.x * threadIdx.y;
    // Make sure we do not go out of bounds
    if(id < n) {
        out[id] = vec1[id] + vec2[id];
    }
}
} //namespace

std::vector<torch::Tensor> vector_addition_cuda(
    torch::Tensor vec1,
    torch::Tensor vec2,
    torch::Tensor out,
    int n) {

  out = torch::zeros_like(out);

  dim3 threadsPerBlock(16,16);
  dim3 numBlocks(ceil(n/threadsPerBlock.x),ceil(n/threadsPerBlock.y));

  AT_DISPATCH_ALL_TYPES(vec1.type(), "vector_addition_cuda_kernel", ([&] {
    vector_addition_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        vec1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        vec2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        n);
  }));
  return {out};
}
