import torch

from phi.torch.flow import *

TORCH.set_default_device('GPU')
my_gpu = TORCH.get_default_device().ref


def test_matmul():
    settings = ([2, 3, 4], [200, 200, 200], [200, 300, 400], [200, 400, 400])
    for m, k, n in settings:
        a = torch.rand(m, k).to(my_gpu)
        b = torch.rand(k, n).to(my_gpu)
        res = TORCH.matmul(a, b)
        proper_res = torch.matmul(a, b)
        assert(torch.sum(res - proper_res) == 0)


def sparse_mat_gen(shape):
    mask = torch.rand(shape).le(0.2)
    mat = torch.zeros(shape)
    mat[mask] = 1
    return mat


def test_SpMM(verbose=False):
    settings = ([2, 2, 2], [2, 3, 4], [200, 200, 200], [200, 300, 400])
    epsilon = 1e-4
    if verbose:
        print("Result matrix size | elements with an error > ", epsilon, "\n------------------------")
    for m, k, n in settings:
        a = torch.rand(m, k).to(my_gpu)
        a_csr = TORCH.matrix_csr(a)
        b = torch.rand(k, n).to(my_gpu)
        res = TORCH.matmul(a_csr, b)
        proper_res = torch.matmul(a, b)
        errors = torch.sum(torch.where(torch.abs(res - proper_res) > epsilon, 1, 0))
        assert(errors == 0)
        if verbose:
            print(m * n, "|", errors)

def test_SpMV(verbose=False):
    settings = ([2, 2, 2], [2, 3, 4], [200, 200, 200], [200, 300, 400])
    epsilon = 1e-4
    if verbose:
        print("Result matrix size | elements with an error > ", epsilon, "\n------------------------")
    for m, k, n in settings:
        a = torch.rand(m, k).to(my_gpu)
        a_csr = TORCH.matrix_csr(a)
        b = torch.rand(k, n).to(my_gpu)
        res = TORCH.matmul(a_csr, b)
        proper_res = torch.matmul(a, b)
        errors = torch.sum(torch.where(torch.abs(res - proper_res) > epsilon, 1, 0))
        assert(errors == 0)
        if verbose:
            print(m * n, "|", errors)

if __name__ == '__main__':
    test_SpMM()