import os
from os.path import isfile, isdir, abspath, join, dirname
import subprocess


def compile_cuda_ops(gcc: str = None,
                     nvcc: str = None,
                     cuda_lib: str = None):
    tf_gcc = check_tf_cuda_compatibility()
    if gcc is None:
        gcc = tf_gcc if isfile(tf_gcc) else 'gcc'
    if nvcc is None:
        nvcc = '/usr/local/cuda/bin/nvcc' if isfile('/usr/local/cuda/bin/nvcc') else 'nvcc'
    if cuda_lib is None:
        cuda_lib = '/usr/local/cuda/lib64/'

    phi_tf_path = abspath(dirname(__file__))
    src_path = join(phi_tf_path, 'cuda', 'src')
    build_path = join(phi_tf_path, 'cuda', 'build')
    logfile_path = join(phi_tf_path, 'cuda', 'log.txt')
    print("Source Path:\t" + src_path)
    print("Build Path:\t" + build_path)
    print("GCC:\t\t" + gcc)
    print("NVCC:\t\t" + nvcc)
    print("CUDA lib:\t" + cuda_lib)
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
            compile_cuda('resample', nvcc, src_path, build_path, logfile=logfile)
            compile_gcc('resample', gcc, src_path, build_path, cuda_lib, logfile=logfile)
            compile_cuda('resample_gradient', nvcc, src_path, build_path, logfile=logfile)
            compile_gcc('resample_gradient', gcc, src_path, build_path, cuda_lib, logfile=logfile)
            # compile_cuda('bicgstab_ilu_linear_solve_op', self.nvcc, src_path, build_path, logfile=logfile)
            # compile_gcc('bicgstab_ilu_linear_solve_op', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
        except BaseException as err:
            print(f"Compilation failed. See {logfile_path} for details.")
            raise err
    print(f"Compilation complete. See {logfile_path} for details.")


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
    tf_cflags = tensorflow.sysconfig.get_compile_flags()
    tf_lflags = tensorflow.sysconfig.get_link_flags()
    link_cuda_lib = '-L' + cuda_lib
    command = [
                gcc,
                join(source_dir, f'{file_names}.cc'),
                join(target_dir, f'{file_names}.cu.o'),
                '-o', join(target_dir, f'{file_names}.so'),
                '-std=c++11',
                '-shared',
                '-fPIC',
                '-lcudart',
                '-O3',
                link_cuda_lib
            ] + tf_cflags + tf_lflags
    print(f"gcc {file_names}")
    logfile.writelines(["\n", " ".join(command), "\n"])
    subprocess.check_call(command, stdout=logfile, stderr=logfile)
