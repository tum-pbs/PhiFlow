# Pytorch custom cuda module
This module has been developed with the goal of implementing C++/CUDA functions to extend the functionality of PyTorch
and specially to achieve a better performance in certain cases.


The implementation is based on the following PyTorch tutorial: 
<https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch>
---
## How does it work
### Compilation 
The file _setup.py_ is used to specify the compilation settings.

To compile the module run the following command:

_$ python setup.py install_

Interesting documentation to understand _setup.py_ compilation process:
- Setuptools <https://setuptools.readthedocs.io/en/latest/> 
- CUDAExtension <https://pytorch.org/docs/stable/cpp_extension.html>.

### Implementation
In order to implement a new function we should follow the next steps:

1. Define the function that you will implement at step 2.
   
    _include/pytorch_custom.hpp_
    ```c++
      std::vector<torch::Tensor> vector_addition_cuda(
        torch::Tensor vector1,
        torch::Tensor vector2,
        torch::Tensor out,
        int n);
       ```
   
2. Implement the C++ function that will be called by Python and that has to comunicate with your CUDA implementation.

    _src/pytorch_custom.cpp_
    ```c++
    std::vector<torch::Tensor> vector_addition(
        torch::Tensor vector1,
        torch::Tensor vector2,
        torch::Tensor out,
        int n) {
        CHECK_INPUT(vector1);
        CHECK_INPUT(vector2);
        CHECK_INPUT(out);
      return vector_addition_cuda(vector1, vector2, out, n);
    }
    ```
    Note that there are a some CUDA error checking functions defined in _include/pytorch_custom.hpp_ that could be
    helpful for your implementation.


3. Add your C++ function to the Python/C++ binding in order to allow Python to call your function once the module is
included in a Python file.
    ```c++
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      ...
      m.def("vector_addition", &vector_addition, "Simple vector addition (CUDA)");
    }
    ```
   
4. Implement your CUDA function inside the unnamed namespace.
    _src/pytorch_custom_cuda.cu_
    ```cuda
    namespace {
    ...
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
    ...
    } //namespace
    ```
5. Implement the C++ function that will set up the environment of your CUDA call.
    
    _src/pytorch_custom_cuda.cu_
        
    ```c++
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
    ```