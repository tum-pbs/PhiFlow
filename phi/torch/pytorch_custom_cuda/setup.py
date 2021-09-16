from os import getcwd
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_custom_cuda',
    ext_modules=[
        CUDAExtension('pytorch_custom_cuda', [
            'src/pytorch_custom.cpp',
            ],
            extra_compile_args={'nvcc': ['-lcublas', '-lcusparse']}
        )
    ],
    include_dirs=[getcwd() + "/include/"],
    cmdclass={
        'build_ext': BuildExtension
    })
