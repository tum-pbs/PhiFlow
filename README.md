# Φ<sub>*Flow*</sub>

[![Build Status](https://travis-ci.com/tum-pbs/PhiFlow.svg?token=8vG2QPsZzeswTApmkekH&branch=master)](https://travis-ci.com/tum-pbs/PhiFlow)

![Gui](documentation/figures/Gui.png)

Φ<sub>*Flow*</sub> is a research-oriented, open-source fluid simulation toolkit.
It is written mostly in Python and can use both NumPy and TensorFlow for execution.

Having all functionality of a fluid simulation running in TensorFlow opens up the possibility of back-propagating gradients through the simulation as well as running the simulation on GPUs.

## Features

- Support for a variety of differentiable simulation types, from Burgers over Navier-Stokes to the Schrödinger equation.
- Tight integration with [TensorFlow](https://www.tensorflow.org/) allowing for straightforward network training with fully differentiable simulations that run on the GPU.
- Object-oriented architecture enabling concise and expressive code, designed for ease of use and extensibility.
- Reusable simulation code, independent of backend and dimensionality, i.e. the exact same code can run a 2D fluid sim using NumPy and a 3D fluid sim on the GPU using TensorFlow.
- Flexible, easy-to-use web interface featuring live visualizations and interactive controls that can affect simulations or network training on the fly.

## Installation

The following commands will get you Φ<sub>*Flow*</sub> + browser-GUI + NumPy execution:

```bash
$ git clone https://github.com/tum-pbs/PhiFlow.git
$ pip install phiflow/[gui]
```

See the [detailed installation instructions](documentation/Installation_Instructions.md) on how to install Φ<sub>*Flow*</sub>
with TensorFlow support.

## Documentation and Guides

| [Index](documentation) | [Demos](demos) / [Tests](tests) | [Source](phi) |
|------------------------|---------------------------------|---------------|

If you would like to get right into it and have a look at some example code, check out the following demos:

- [simpleplume.py](./demos/simpleplume.py): Runs a fluid simulation and displays it in the browser
- [optimize_pressure.py](./demos/optimize_pressure.py): Uses TensorFlow to optimize a velocity channel. TensorBoard can be started from the GUI and displays the loss.


### Running simulations

The [simulation overview](documentation/Simulation_Overview.md) explains how to run predefined simulations using either the [NumPy or TensorFlow](documentation/NumPy_and_TensorFlow_Execution.md) backend. It also introduces the GUI.

To learn how specific simulations are implemented, check out the documentation for [Fluids](documentation/Fluid_Simulation.md) or read about [staggered grids](documentation/Staggered_Grids.md) or [pressure solvers](documentation/Pressure_Solvers.md).

[Writing a Φ<sub>*Flow*</sub> Application](documentation/Browser_GUI.md) introduces the high-level classes and explains how to use the Φ<sub>*Flow*</sub> GUI for displaying a simulation.

For I/O and data management, see the [data documentation](documentation/Reading_and_Writing_Data.md) or the [scene format specification](documentation/Scene_Format_Specification.md).


### Optimization and Learning

For training machine learning models, [this document](documentation/Interactive_Training_Apps.md) gives an introduction into writing a GUI-enabled application.


### Architecture

The [simulation code design documentation](documentation/Simulation_Architecture.md) provides a deeper look into the object-oriented code design of simulations.

All simulations of continuous systems are based on the [Field API](documentation/Fields.md) and underlying all states is the [struct API](documentation/Structs.ipynb).

The [software architecture documentation](documentation/Software_Architecture.md) shows the building blocks of Φ<sub>*Flow*</sub> and the module dependencies.


## Known Issues

GUI: Message not updating correctly on some Chrome installations on Windows.

TensorBoard: Live supervision does not work when running a local app that writes to a remote directory.

Resampling / Advection: NumPy interpolation handles the boundaries slightly differently than TensorFlow.

## Acknowledgements

This work is supported by the ERC Starting Grant realFlow (StG-2015-637014) and the Intel Intelligent Systems Lab.

