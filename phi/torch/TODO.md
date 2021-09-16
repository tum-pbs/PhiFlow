- [x] Implement cuBLAS matmul as a warmup for more complex functions
- [x] Define sparse named tuple:
- [ ] Implement CUDA Sparse matmul
    - [x] Code general implementation
    - [x] Fix compilation errors
    - [ ] Make sure that the computations are correct (We might find an issue with col major vs row major)
      - When executing sparse_matmul we find the following error:
```
 ** On entry to cusparseCreateDnMat() row-major layout is not currently supported 
 ** On entry to cusparseCreateDnMat() row-major layout is not currently supported 

I have to define the matrices in col-major order and then the multiplications 
have to yield a proper row-major result for pytorch to continue:

Torch:Row_major
CUSPARSE:Col_major

B^T A^T = C^T -> We want C^T in col_major equal to C in row_major
TLDR: We compute B * A instead