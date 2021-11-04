from os import getcwd
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CXX_ARGS = ["-L/home/marc/TUM/MyPhiFlow/venv/lib/python3.8/site-packages/torch/lib/"]
NVCC_ARGS = ["-L/home/marc/TUM/MyPhiFlow/venv/lib/python3.8/site-packages/torch/lib/"]

ext_modules = [
    CUDAExtension(
        name="pytorch_custom_cuda",
        sources=["src/pytorch_custom.cpp", "src/cuda_code.cu"],
        extra_compile_args={"cxx": CXX_ARGS, "nvcc": NVCC_ARGS},
        libraries=["cusparse"],
    ),
]

setup(
    name="pytorch_custom_cuda",
    packages=[],
    ext_modules=ext_modules,
    include_dirs=["include"],
    cmdclass={"build_ext": BuildExtension}
)
