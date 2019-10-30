# Φ<sub>*Flow*</sub> Installation

## Dependencies

You need to have Python 3.5.X, 3.6.X or 2.7.X with pip (or an alternative package manager) installed.
NumPy and SciPy are required in order to run Φ<sub>*Flow*</sub>. These are likely included in your Python distribution.

If not, you can install them by running:

```bash
$ pip install scipy
```

For GPU acceleration, deep learning and optimization, [TensorFlow](https://www.tensorflow.org/install/) must be registered with your Python installation.
TensorFlow can be compiled from sources or installed as the CPU-enabled version using:

```bash
$ pip install tensorflow==1.14.0
```

and as the GPU-enabled version using:

```bash
$ pip install tensorflow_gpu==1.14.0
```

The browser-based GUI depends on [Plotly / Dash](https://dash.plot.ly/installation).
These packages can be installed together with Φ<sub>*Flow*</sub> (see next section).

## Installing Φ<sub>*Flow*</sub> from sources

Clone the git repository by running

```bash
$ git clone https://gitlab.lrz.de/ga67fet/phiflow.git
```

See the section *Optional features* below on how to configure the installation to add CUDA operators.

Φ<sub>*Flow*</sub> is built as a Python package.
If you run a program that uses Φ<sub>*Flow*</sub> from the command line, it is recommended to install Φ<sub>*Flow*</sub> using pip.
For installing Φ<sub>*Flow*</sub> with GUI dependencies, run:

```bash
$ pip install phiflow/[gui]
```

To update the Φ<sub>*Flow*</sub> installation (because the sources changed), simply run
`$ pip install phiflow/` (or `$ pip install .` inside the phiflow directory).

Installing Φ<sub>*Flow*</sub> as a package is not required if your Python PATH points to it (as is the case when executing code from within most IDEs).

## Verifying the installation

### Testing with GUI

To verify your Φ<sub>*Flow*</sub> installation (including TensorFlow), run `smokedemo_tf.py` using the following command:

```bash
$ python phiflow/demos/smokedemo.py tf
```

To test Φ<sub>*Flow*</sub> without TensorFlow, leave out the `tf` at the end.

The application should output a URL which you can open in your browser to use the GUI.
At the top of the page, the app displays if it is using TensorFlow or NumPy.
Simply press the Play button and watch the simulation run.

### Testing without GUI

If you do not wish to use the GUI, you can verify the installation by running the test scripts.
Inside the phiflow directory, run:

```bash
$ pip install pytest
$ pytest
```

This will run all tests (including some depending on TensorFlow).
If everything works correctly, all test should pass.

## Optional features

### CUDA operations

There are some custom TensorFlow operations, written in CUDA.
To use these, you must have a TensorFlow compatible CUDA SDK installed.

To install Φ<sub>*Flow*</sub> with CUDA operators, run:

```bash
$ python phiflow/setup.py cuda
$ pip install phiflow/
```
