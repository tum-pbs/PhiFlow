# Φ<sub>Flow</sub> Installation

## Requirements

* [Python](https://www.python.org/downloads/) 3.6 or newer (e.g. [Anaconda](https://www.anaconda.com/products/individual))
* [pip](https://pip.pypa.io/en/stable/) (included in many Python distributions)

For GPU acceleration, deep learning and optimization,
[TensorFlow 2.x](https://www.tensorflow.org/install/) or [PyTorch](https://pytorch.org/)
must be registered with your Python installation.
Note that these also require a CUDA installation with *cuDNN* libraries for GPU execution.

## Installing Φ<sub>Flow</sub> using pip

*Note*: If you want to use the Φ<sub>Flow</sub> CUDA operations with TensorFlow, you have to build Φ<sub>Flow</sub> from source instead (see below).

The following command installs the latest stable version of Φ<sub>Flow</sub> using pip.
```bash
$ pip install phiflow
```
To install the latest developer version, run
```bash
$ pip install --upgrade git+https://github.com/tum-pbs/PhiFlow@develop
```

## Installing Φ<sub>Flow</sub> from Source
The Φ<sub>Flow</sub> source additionally contains demo scripts and tests.
In particular, it includes [`tests/verify.py`](https://github.com/tum-pbs/PhiFlow/blob/develop/tests/verify.py),
which tests your configuration.

Clone the git repository by running
```bash
$ git clone https://github.com/tum-pbs/PhiFlow.git <target directory>
```
This will create the folder \<target directory\> and copy all Φ<sub>Flow</sub> source files into it.

With this done, you may compile CUDA kernels for better performance, see below.

Finally, Φ<sub>Flow</sub> needs to be added to the Python path.
This can be done in one of multiple ways:

* Marking \<target directory\> as a source directory in your Python IDE.
* Manually adding the cloned directory to the Python path.
* Installing Φ<sub>Flow</sub> using pip: `$ pip install <target directory>/`. This command needs to be rerun after you make changes to the source code.


## Compiling the CUDA Kernels for TensorFlow

The Φ<sub>Flow</sub> source includes several custom CUDA kernels to speed up certain operations when using TensorFlow.
To use these, you must have a [TensorFlow compatible CUDA SDK with cuDNN](https://www.tensorflow.org/install/gpu#software_requirements) as well as a compatible C++ compiler installed.
We strongly recommend using Linux with GCC 7.3.1 (`apt-get install gcc-4.8`) for this.
See the [tested TensorFlow build configurations](https://www.tensorflow.org/install/source#tested_build_configurations).

To compile the CUDA kernels for TensorFlow, clone the repository as described above, then run `$ python <target directory>/setup.py tf_cuda`.
This will add the compiled CUDA binaries to the \<target directory\>.


## Verifying the installation
If you have installed Φ<sub>Flow</sub> from source, execute the included `verify.py` script.
```bash
$ python <Φ-Flow directory>/tests/verify.py
```
Otherwise, run the following Python code.
```python
import phi
phi.verify()
```
If Φ<sub>Flow</sub> and dependencies are installed correctly, you should see the text `Installation verified.`, followed by additional information on the components at the end of the console output.


### Unit tests

PyTest is required to run the unit tests. To install it, run `$ pip install pytest`.

Some unit tests require TensorFlow and PyTorch.
Make sure, both are installed before running the tests.

Execute `$ pytest <target directory>/tests/commit` to run the normal tests.
The result of this should match the automated tests run by Travis CI which are linked on the main GitHub page.

You can run the additional tests
```bash
$ pytest <target directory>/tests/gpu
$ pytest <target directory>/tests/release
```
