from phi.torch.flow import *
import pytorch_custom_cuda

TORCH.set_default_device('GPU')
my_gpu = TORCH.get_default_device().ref


def test_matmul():
    settings = ([2, 3, 4], [200, 200, 200], [200, 300, 400], [200, 400, 400])
    for m, k, n in settings:
        a = torch.rand(m, k).to(my_gpu)
        b = torch.rand(k, n).to(my_gpu)
        res = pytorch_custom_cuda.cublas_matmul(a, b)
        proper_res = torch.matmul(a, b)
        assert(torch.sum(res - proper_res) == 0)


if __name__ == '__main__':
    test_matmul()


