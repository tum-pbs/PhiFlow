from os import getcwd
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = [
    CUDAExtension(
        name="phi_torch_cuda",
        sources=[
            "./phi/torch/cuda/src/phi_torch_cuda.cpp",
            "./phi/torch/cuda/src/phi_torch_cuda_kernel.cu"
        ],
        libraries=[
            "cusparse", 
            "cublas"
        ],
        extra_compile_args={
            #"gcc":  ["-g0"],    #"-D_GLIBCXX_USE_CXX11_ABI=1"
            "nvcc": ["-O3", "-gencode=arch=compute_75,code=sm_75", "-lcusparse", "-lcublas"]
        }
    ),
]

setup(
    name="phi_torch_cuda",
    packages=[],
    ext_modules=ext_modules,
    include_dirs=["./phi/torch/cuda/include"],
    # libraries=["cusparse"],
    cmdclass={"build_ext": BuildExtension}
)
