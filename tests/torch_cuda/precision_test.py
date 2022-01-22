from json.tool import main

from numpy import dtype
import torch_cuda
from phi.torch.flow import *
import phi.math._functional as functional
from phi.torch._torch_backend import *
TORCH.set_default_device('GPU')

def test_matmul_float(epsilon):
    """
    Multiply double precision torch tensors.
    """
    dim_i = 100000
    dim_j = 100000
    a = torch.rand(dim_i, dim_j, dtype=torch.float32, device='cuda')
    b = torch.rand(dim_j, 1, dtype=torch.float32, device='cuda')
    sp_a = a.to_sparse_csr()
    c = torch.matmul(a, b)
    c_marc = torch_cuda.cusparse_SpMV(sp_a.crow_indices(), sp_a.col_indices(), sp_a.values(), b, dim_i, dim_j)
    diff = c - c_marc
    print(f'Number of errors: {torch.sum(diff>epsilon)} out of {dim_i}')

def test_matmul_double(epsilon):
    """
    Multiply double precision torch tensors.
    """
    dim_i = 1000
    dim_j = 1000
    a = torch.rand(dim_i, dim_j, dtype=torch.float64, device='cuda')
    b = torch.rand(dim_j, 1, dtype=torch.float64, device='cuda')
    sp_a = a.to_sparse_csr()
    c = torch.matmul(a, b)
    c_marc = torch_cuda.cusparse_SpMV_double(sp_a.crow_indices(), sp_a.col_indices(), sp_a.values(), b, dim_i, dim_j)
    diff = c - c_marc
    print(f'Number of errors: {torch.sum(diff>epsilon)} out of {dim_i}')

if __name__ == '__main__':
    for epsilon in [1e-7, 1e-6, 1e-5, 1e-4]:
        print(f'Epsilon = {epsilon}')
        print("Testing double precision")
        test_matmul_double(epsilon)
        print("Testing single precision")
        test_matmul_float(epsilon)
        print("------------------------")
