# Φ<sub>Flow</sub>

[![Build Status](https://travis-ci.com/tum-pbs/PhiFlow.svg?token=8vG2QPsZzeswTApmkekH&branch=master)](https://travis-ci.com/tum-pbs/PhiFlow)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/phiflow.svg)](https://pypi.org/project/phiflow/)
[![PyPI license](https://img.shields.io/pypi/l/phiflow.svg)](https://pypi.org/project/phiflow/)
[![Code Coverage](https://codecov.io/gh/tum-pbs/PhiFlow/branch/develop/graph/badge.svg)](https://codecov.io/gh/tum-pbs/PhiFlow/branch/develop/)
[![Google Collab Book](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S21OY8hzh1oZK2wQyL3BNXvSlrMTtRbV#offline=true&sandboxMode=true)

![Gui](documentation/figures/WebInterface.png)

Φ<sub>Flow</sub> is an open-source simulation toolkit built for optimization and machine learning applications.
It is written mostly in Python and can be used with NumPy, TensorFlow or PyTorch.
The close integration with machine learning frameworks allows it to leverage their automatic differentiation functionality,
making it easy to build end-to-end differentiable functions involving both learning models and physics simulations.

## Features

* Variety of built-in PDE operations with focus on fluid phenomena, allowing for concise formulation of simulations.
* Tight integration with [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for straightforward neural network training with fully differentiable simulations that [run on the GPU](documentation/GPU_Execution.md).
* Flexible, easy-to-use [web interface](documentation/Web_Interface.md) featuring live visualizations and interactive controls that can affect simulations or network training on the fly.
* Object-oriented, vectorized design for expressive code, ease of use, flexibility and extensibility.
* Reusable simulation code, independent of backend and dimensionality, i.e. the exact same code can run a 2D fluid sim using NumPy and a 3D fluid sim on the GPU using TensorFlow or PyTorch.
* High-level linear equation solver with automated sparse matrix generation.

## Publications

* [Learning to Control PDEs with Differentiable Physics](https://ge.in.tum.de/publications/2020-iclr-holl/), *Philipp Holl, Vladlen Koltun, Nils Thuerey*, ICLR 2020.
* [Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers](https://arxiv.org/abs/2007.00016), *Kiwon Um, Raymond Fei, Philipp Holl, Robert Brand, Nils Thuerey*, NeurIPS 2020.
* [Φ<sub>Flow</sub>: A Differentiable PDE Solving Framework for Deep Learning via Physical Simulations](https://montrealrobotics.ca/diffcvgp/), *Nils Thuerey, Kiwon Um, Philipp Holl*, DiffCVGP workshop at NeurIPS 2020.

## Installation

Installation with pip on Python 3.7 or newer:
``` bash
$ pip install phiflow
```
Install TensorFlow or PyTorch in addition to Φ<sub>Flow</sub> to enable machine learning capabilities and GPU execution.
See the [detailed installation instructions](documentation/Installation_Instructions.md) on how to compile the custom CUDA operators and verify your installation.

## Documentation and Guides

| [Index](documentation) | [Demos](demos) | [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> Fluids Tutorial](https://colab.research.google.com/drive/1S21OY8hzh1oZK2wQyL3BNXvSlrMTtRbV#offline=true&sandboxMode=true) / [Playground](https://colab.research.google.com/drive/1zBlQbmNguRt-Vt332YvdTqlV4DBcus2S#offline=true&sandboxMode=true) | [Source](phi) |
|------------------------|---------------------------------|---------------| -----------------------------|

If you would like to get right into it and have a look at some code, check out the
[tutorial notebook on Google Colab](https://colab.research.google.com/drive/1S21OY8hzh1oZK2wQyL3BNXvSlrMTtRbV#offline=true&sandboxMode=true).
It lets you run fluid simulations with Φ<sub>Flow</sub> in the browser.

The following introductory demos are also helpful to get started with writing your own app using Φ<sub>Flow</sub>:

* [simpleplume.py](./demos/simpleplume.py) runs a fluid simulation and displays it in the web interface.
* [optimize_pressure.py](demos/differentiate_pressure.py) uses TensorFlow to optimize a velocity field.

## Module Overview

| Module      | Documentation                                        |
|-------------|------------------------------------------------------|
| [phi.app](phi/app)     | [Interactive application development, web interface](documentation/Web_Interface.md)   |
| [phi.physics](phi/physics) | [Domains, built-in physics functions](documentation/Physics.md)         |
| [phi.field](phi/field)   | [Grids, particles, analytic representations](documentation/Fields.md)           |
| [phi.geom](phi/geom)    | [Differentiable Geometry](documentation/Geometry.md)                              |
| [phi.math](phi/math)    | [Vectorized operations, tensors with named dimensions](documentation/Math.md) |

## Other Documentation

* [Fluids](documentation/Fluid_Simulation.md)
* [Staggered grids](documentation/Staggered_Grids.md)

Not yet updated:

* [GPU execution](documentation/GPU_Execution.md)
* [NumPy or TensorFlow](documentation/NumPy_and_TensorFlow_Execution.md)
* [Data](documentation/Reading_and_Writing_Data.md)
* [Pressure solvers](documentation/Pressure_Solvers.md)

## Version History

The [Version history](https://github.com/tum-pbs/PhiFlow/releases) lists all major changes since release.

## Contributions

Contributions are welcome! Check out [this document](CONTRIBUTING.md) for guidelines.

## Acknowledgements

This work is supported by the ERC Starting Grant realFlow (StG-2015-637014) and the Intel Intelligent Systems Lab.
