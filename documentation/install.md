# Φ-*Flow* Installation

## Dependencies

You need to have Python 3.5.X, 3.6.X or 2.7.X with pip (or an alternative package manager) installed.
NumPy and SciPy are required in order to run Φ-*Flow* and come with many Python distributions.

For GPU acceleration, deep learning and optimization, [TensorFlow](https://www.tensorflow.org/install/) must be registered with your Python installation.
TensorFlow can be compiled from sources or installed using

```
$ pip install tensorflow
```

The GUI requires [Plotly / Dash](https://dash.plot.ly/installation) which can be installed together with Φ-*Flow*.


## Installing Φ-*Flow* from sources

Clone the git repository and checkout the branch `phiflow/master`.

```
$ git clone https://bitbucket.org/thunil/mantaflowgit.git
$ cd mantaflowgit/
$ git checkout phiflow/master
```

See the section *Optional features* below on how to configure the installation to add CUDA operators.

Φ-*Flow* is built as a Python package.
If you run a program that uses Φ-*Flow* from the command line, it is recommended to install Φ-*Flow* using pip.
For installing Φ-*Flow* with GUI dependencies, run
```
$ pip install .[gui]
```

To update the Φ-*Flow* installation (because the sources changed), run
```
$ pip install .
```

Installing Φ-*Flow* as a package is not required if your Python PATH points to it (as is the case when executing code from within most IDEs).


## Verifying the installation

To verify your Φ-*Flow* installation (including TensorFlow), run simpleplume_tf.py like so:
```
$ cd apps
$ python simpleplume_np.py
```

The application should output a URL which you can open in your browser to use the GUI.
Simply press the Play button and watch the simulation run.

If you don't have TensorFlow installed, you can test the installation using
```
$ cd apps
$ python simpleplume_np.py
```


## Optional features

### CUDA operations

There are some custom TensorFlow operations, written in CUDA.
To use these, you must have a TensorFlow compatible CUDA SDK installed.

To install Φ-*Flow* with CUDA operators, run
```
$ python setup.py cuda
$ pip install .
```

### MantaFlow integration

Deprecated.

Φ-*Flow* can be used in conjunction with MantaFlow.
If you would like to use MantaFlow functions, see the installation instructions of the branch [mantatensor](https://bitbucket.org/thunil/mantaflowgit/src/mantatensor/).
