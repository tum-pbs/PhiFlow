from functools import partial
from phi.torch.flow import *
TORCH.set_default_device('GPU')

#laplace = partial(math.laplace, padding=math.extrapolation.PERIODIC)
#%lin = math.jit_compile_linear(laplace).sparse_coordinate_matrix(math.zeros(spatial(x=3, y=3))).native().to_dense()

lin = TORCH.as_tensor([[3.0, 2], [2, 6]])
lin_csr = [TORCH.matrix_csr(lin), TORCH.matrix_csr(lin)]
y = TORCH.as_tensor([[2.0, -8], [2.0, -8]]).T
x0 = TORCH.as_tensor([[-2.0, -2], [-2.0, -2]]).T
rtol = 1e-5
atol = 0
res = TORCH.conjugate_gradient(lin=lin_csr, y=y, x0=x0, rtol=rtol, atol=atol, max_iter=1000, trj=True)
#print(TORCH.matmul(lin_csr, res[-1].x) - y)
