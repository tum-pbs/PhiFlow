from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_custom_cuda',
    ext_modules=[
        CUDAExtension('pytorch_custom_cuda', [
            'min_repr_example.cpp'],
            extra_compile_args={'nvcc': ['-lcublas', '-lcusparse']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
