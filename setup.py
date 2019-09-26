import setuptools
import distutils.cmd
import distutils.log
from setuptools import setup
from setuptools.command.install import install
import subprocess
import os


class CudaCommand(distutils.cmd.Command):
    description = "Compile CUDA sources"
    user_options = [
        ("gcc=", None, "Path to the gcc compiler."),
        ("nvcc=", None, "Path to the Nvidia nvcc compiler."),
    ]

    def run(self):
        srcPath = os.path.abspath("./phi/solver/cuda/src")
        buildPath = os.path.abspath("./phi/solver/cuda/build")
        print("Source Path:\t" + srcPath)
        print("Build Path:\t" + buildPath)

        # Get TF Compile/Link Flags and write to env
        import tensorflow as tf
        TF_CFLAGS = tf.sysconfig.get_compile_flags()
        TF_LFLAGS = tf.sysconfig.get_link_flags()

        # Remove old build files
        if os.path.isdir(buildPath):
            print("Removing old build files from %s" % buildPath)
            for f in os.listdir(buildPath):
                os.remove(os.path.join(buildPath, f))
        else:
            print("Creating build directory at %s" % buildPath)
            os.mkdir(buildPath)

        print("Compiling CUDA code...")
        # Build the Laplace Matrix Generation CUDA Kernels
        subprocess.check_call([self.nvcc,
                               "-std=c++11",
                               "-c",
                               "-o",
                               os.path.join(buildPath, "laplace_op.cu.o"),
                               os.path.join(srcPath, "laplace_op.cu.cc"),
                               "-x",
                               "cu",
                               "-Xcompiler",
                               "-fPIC"]
                               + TF_CFLAGS)

        # Build the Laplace Matrix Generation Custom Op
        # This is only needed for the Laplace Matrix Generation Benchmark
        subprocess.check_call([self.gcc,
                               "-std=c++11",
                               "-shared",
                               "-o",
                               os.path.join(buildPath, "laplace_op.so"),
                               os.path.join(srcPath, "laplace_op.cc"),
                               os.path.join(buildPath, "laplace_op.cu.o"),
                               "-fPIC"]
                               + TF_CFLAGS + TF_LFLAGS)

        # Build the Pressure Solver CUDA Kernels
        subprocess.check_call([self.nvcc,
                               "-std=c++11",
                               "-c",
                               "-lcublas",
                               "-o",
                               os.path.join(buildPath, "pressure_solve_op.cu.o"),
                               os.path.join(srcPath, "pressure_solve_op.cu.cc"),
                               "-x", "cu",
                               "-Xcompiler",
                               "-fPIC"]
                               + TF_CFLAGS)

        # Build the Pressure Solver Custom Op
        subprocess.check_call([self.gcc,
                               "-std=c++11",
                               "-shared",
                               "-o",
                               os.path.join(buildPath, "pressure_solve_op.so"),
                               os.path.join(srcPath, "pressure_solve_op.cc"),
                               os.path.join(buildPath, "pressure_solve_op.cu.o"),
                               os.path.join(buildPath, "laplace_op.cu.o"),
                               "-fPIC"]
                               + TF_CFLAGS + TF_LFLAGS)


    def initialize_options(self):
        self.gcc = "gcc"
        self.nvcc = "nvcc"

    def finalize_options(self):
        assert os.path.isfile(self.gcc) or self.gcc == "gcc"
        assert os.path.isfile(self.nvcc) or self.nvcc == "nvcc"


extras = {
    'gui': ["dash", "dash-renderer", "dash-html-components", "dash-core-components", "plotly"],
}

setup(
    name='phiflow',
    version='0.3.3',
    packages=['phi', 'phi.data', 'phi.local', 'phi.math', 'phi.physics', 'phi.solver', 'phi.tf', 'phi.viz'],
    cmdclass={
        "cuda": CudaCommand,
    },
    include_package_data=True,
    url='https://bitbucket.org/thunil/mantaflowgit/src/PhiFlow/',
    license='Apache License, Version 2.0',
    author='Philipp Holl',
    author_email='philipp.holl@tum.de',
    description='Fully Differentiable Grid-based Fluid Simulations on the GPU',
    install_requires=['six'],
    extras_require=extras
)
