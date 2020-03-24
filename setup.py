import distutils.cmd
import distutils.log
import subprocess
import os
from setuptools import setup


class CudaCommand(distutils.cmd.Command):
    description = 'Compile CUDA sources'
    user_options = [
        ('gcc=', None, 'Path to the gcc compiler.'),
        ('nvcc=', None, 'Path to the Nvidia nvcc compiler.'),
    ]

    def run(self):
        src_path = os.path.abspath('./phi/tf/cuda/src')
        build_path = os.path.abspath('./phi/tf/cuda/build')
        print('Source Path:\t' + src_path)
        print('Build Path:\t' + build_path)

        # Get TF Compile/Link Flags and write to env
        import tensorflow as tf
        if tf.__version__[0] == '2':
            print('Adjusting for tensorflow 2.0')
            tf = tf.compat.v1
            tf.disable_eager_execution()
        tf_cflags = tf.sysconfig.get_compile_flags()
        tf_lflags = tf.sysconfig.get_link_flags()

        # Remove old build files
        if os.path.isdir(build_path):
            print('Removing old build files from %s' % build_path)
            for file in os.listdir(build_path):
                os.remove(os.path.join(build_path, file))
        else:
            print('Creating build directory at %s' % build_path)
            os.mkdir(build_path)

        print('Compiling CUDA code...')
        # Build the Laplace Matrix Generation CUDA Kernels
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-o',
                os.path.join(build_path, 'laplace_op.cu.o'),
                os.path.join(src_path, 'laplace_op.cu.cc'),
                '-x',
                'cu',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        # Build the Laplace Matrix Generation Custom Op
        # This is only needed for the Laplace Matrix Generation Benchmark
        subprocess.check_call(
            [
                self.gcc,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'laplace_op.so'),
                os.path.join(src_path, 'laplace_op.cc'),
                os.path.join(build_path, 'laplace_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda/lib64/','-lcudart']
        )

        # Build the Pressure Solver CUDA Kernels
        subprocess.check_call(
            [
                self.nvcc,
                '-std=c++11',
                '-c',
                '-lcublas',
                '-o',
                os.path.join(build_path, 'pressure_solve_op.cu.o'),
                os.path.join(src_path, 'pressure_solve_op.cu.cc'),
                '-x', 'cu',
                '-Xcompiler',
                '-fPIC'
            ]
            + tf_cflags
        )

        # Build the Pressure Solver Custom Op
        subprocess.check_call(
            [
                self.gcc,
                '-std=c++11',
                '-shared',
                '-o',
                os.path.join(build_path, 'pressure_solve_op.so'),
                os.path.join(src_path, 'pressure_solve_op.cc'),
                os.path.join(build_path, 'pressure_solve_op.cu.o'),
                os.path.join(build_path, 'laplace_op.cu.o'),
                '-fPIC'
            ]
            + tf_cflags
            + tf_lflags
            + ['-L/usr/local/cuda/lib64/','-lcudart']
        )

    def initialize_options(self):
        self.gcc = 'gcc-4.8'
        self.nvcc = 'nvcc'

    def finalize_options(self):
        assert os.path.isfile(self.gcc) or self.gcc == 'gcc-4.8'
        assert os.path.isfile(self.nvcc) or self.nvcc == 'nvcc'


try:
    with open(os.path.join(os.path.dirname(__file__), 'documentation/Package_Info.md'), 'r') as readme:
        long_description = readme.read()
except FileNotFoundError:
    pass

with open(os.path.join(os.path.dirname(__file__), 'phi', 'VERSION'), 'r') as version_file:
    version = version_file.read()


setup(
    name='phiflow',
    version=version,
    download_url='https://github.com/tum-pbs/PhiFlow/archive/%s.tar.gz' % version,
    packages=['phi',
              'phi.app',
              'phi.backend',
              'phi.data',
              'phi.geom',
              'phi.local',
              'phi.math',
              'phi.physics',
              'phi.physics.field',
              'phi.physics.pressuresolver',
              'phi.struct',
              'phi.tf',
              'phi.viz',
              'phi.viz.dash',
              'webglviewer'],
    cmdclass={
        'tf_cuda': CudaCommand,
    },
    description='Research-oriented differentiable fluid simulation framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['Differentiable', 'Simulation', 'Fluid', 'Machine Learning', 'Deep Learning'],
    license='MIT',
    author='Philipp Holl',
    author_email='philipp.holl@tum.de',
    url='https://github.com/tum-pbs/PhiFlow',
    include_package_data=True,
    install_requires=['six', 'packaging', 'scipy'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    extras_require={
        'gui': ['dash',
                'dash-renderer',
                'dash-html-components',
                'dash-core-components',
                'plotly',
                'imageio'],
    }
)
