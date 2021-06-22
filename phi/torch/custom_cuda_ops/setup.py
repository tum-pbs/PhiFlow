from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_operators',
    ext_modules=[
        CUDAExtension('custom_operators', [
            'custom_operators.cpp',
            'custom_operators_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })