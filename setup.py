import distutils.cmd
import distutils.log
import subprocess
import os
import sys
from os.path import join, isfile, abspath, isdir, dirname
from setuptools import setup


def check_tf_cuda_compatibility():
    import tensorflow
    build = tensorflow.sysconfig.get_build_info()  # is_rocm_build, cuda_compute_capabilities
    tf_gcc = build['cpu_compiler']
    is_cuda_build = build['is_cuda_build']
    print(f"TensorFlow compiler: {tf_gcc}.")
    if not is_cuda_build:
        raise AssertionError("Your TensorFlow build does not support CUDA.")
    else:
        cuda_version = build['cuda_version']
        cudnn_version = build['cudnn_version']
        print(f"TensorFlow was compiled against CUDA {cuda_version} and cuDNN {cudnn_version}.")
        return tf_gcc


def compile_cuda(file_names, nvcc, source_dir, target_dir, logfile):
    import tensorflow
    tf_cflags = tensorflow.sysconfig.get_compile_flags()
    command = [
            nvcc,
            join(source_dir, f'{file_names}.cu.cc'),
            '-o', join(target_dir, f'{file_names}.cu.o'),
            '-std=c++11',
            '-c',
            '-D GOOGLE_CUDA=1',
            '-x', 'cu',
            '-Xcompiler',
            '-fPIC',
            '--expt-relaxed-constexpr',
            '-DNDEBUG',
            '-O3'
        ] + tf_cflags
    print(f"nvcc {file_names}")
    logfile.writelines(["\n", " ".join(command), "\n"])
    subprocess.check_call(command, stdout=logfile, stderr=logfile)


def compile_gcc(file_names, gcc, source_dir, target_dir, cuda_lib, logfile):
    import tensorflow
    from packaging import version
    if version.parse(tensorflow.__version__) >= version.parse('2.5.0'):
        cpp_version, gcc_version = '14', '7.5'
    else:
        cpp_version, gcc_version = '11', '4.8'
    tf_cflags = tensorflow.sysconfig.get_compile_flags()
    tf_lflags = tensorflow.sysconfig.get_link_flags()
    link_cuda_lib = '-L' + cuda_lib
    command = [
                gcc,
                join(source_dir, f'{file_names}.cc'),
                join(target_dir, f'{file_names}.cu.o'),
                '-o', join(target_dir, f'{file_names}.so'),
                f'-std=c++{cpp_version}',
                '-shared',
                '-fPIC',
                '-lcudart',
                '-O3',
                link_cuda_lib
            ] + tf_cflags + tf_lflags
    print(f"gcc {file_names}")
    logfile.writelines(["\n", " ".join(command), "\n"])
    subprocess.check_call(command, stdout=logfile, stderr=logfile)


class CudaCommand(distutils.cmd.Command):
    description = 'Compile CUDA sources'
    user_options = [
        ('gcc=', None, 'Path to the gcc compiler.'),
        ('nvcc=', None, 'Path to the Nvidia nvcc compiler.'),
        ('cuda-lib=', None, 'Path to the CUDA libraries.'),
    ]

    def initialize_options(self):
        tf_gcc = check_tf_cuda_compatibility()
        self.gcc = tf_gcc if isfile(tf_gcc) else 'gcc'
        self.nvcc = '/usr/local/cuda/bin/nvcc' if isfile('/usr/local/cuda/bin/nvcc') else 'nvcc'
        self.cuda_lib = '/usr/local/cuda/lib64/'

    def finalize_options(self) -> None:
        pass

    def run(self):
        src_path = abspath('./phi/tf/cuda/src')
        build_path = abspath('./phi/tf/cuda/build')
        logfile_path = abspath('./phi/tf/cuda/log.txt')
        print("Source Path:\t" + src_path)
        print("Build Path:\t" + build_path)
        print("GCC:\t\t" + self.gcc)
        print("NVCC:\t\t" + self.nvcc)
        print("CUDA lib:\t" + self.cuda_lib)
        print("----------------------------")
        # Remove old build files
        if isdir(build_path):
            print('Removing old build files from %s' % build_path)
            for file in os.listdir(build_path):
                os.remove(join(build_path, file))
        else:
            print('Creating build directory at %s' % build_path)
            os.mkdir(build_path)
        print('Compiling CUDA code...')
        with open(logfile_path, "w") as logfile:
            try:
                compile_cuda('resample', self.nvcc, src_path, build_path, logfile=logfile)
                compile_gcc('resample', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
                compile_cuda('resample_gradient', self.nvcc, src_path, build_path, logfile=logfile)
                compile_gcc('resample_gradient', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
                # compile_cuda('bicgstab_ilu_linear_solve_op', self.nvcc, src_path, build_path, logfile=logfile)
                # compile_gcc('bicgstab_ilu_linear_solve_op', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
            except BaseException as err:
                print(f"Compilation failed. See {logfile_path} for details.")
                raise err
        print(f"Compilation complete. See {logfile_path} for details.")


try:
    with open(join(dirname(__file__), 'docs/Package_Info.md'), 'r') as readme:
        long_description = readme.read()
except FileNotFoundError:
    long_description = ""
    pass

with open(join(dirname(__file__), 'phi', 'VERSION'), 'r') as version_file:
    version = version_file.read()

setup(
    name='phiflow',
    version=version,
    download_url='https://github.com/tum-pbs/PhiFlow/archive/%s.tar.gz' % version,
    packages=['phi',
              'phi.field',
              'phi.geom',
              'phi.jax',
              'phi.jax.stax',
              'phi.math',
              'phi.math.backend',
              'phi.physics',
              'phi.tf',
              'phi.torch',
              'phi.vis',
              'phi.vis._console',
              'phi.vis._dash',
              'phi.vis._matplotlib',
          ],
    cmdclass={
        'tf_cuda': CudaCommand,
    },
    description='Differentiable PDE solving framework for machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['Differentiable', 'Simulation', 'Fluid', 'Machine Learning', 'Deep Learning'],
    license='MIT',
    author='Philipp Holl',
    author_email='philipp.holl@tum.de',
    url='https://github.com/tum-pbs/PhiFlow',
    include_package_data=True,
    install_requires=[
        'numpy',  # 1.20 causes TensorFlow tracing errors: NotImplementedError: Cannot convert a symbolic Tensor to a numpy array.
        'scipy>=1.5.4',
        'matplotlib'  # also required by dash for color maps
    ],
    # Optional packages:
    # - dash + plotly (included in dash)
    # - torch
    # - tensorflow
    # - jax
    #
    # phi.verify() should detect missing packages.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
