# Φ<sub>*Flow*</sub>

[![pipeline status](https://gitlab.lrz.de/ga67fet/phiflow/badges/master/pipeline.svg)](https://gitlab.lrz.de/ga67fet/phiflow/commits/master)


![Gui](documentation/figures/Gui.png)

Φ<sub>*Flow*</sub> is a research-oriented, open-source fluid simulation toolkit.
It is written mostly in Python and can use both NumPy and TensorFlow for execution.

Having all functionality of a fluid simulation running in TensorFlow opens up the possibility of back-propagating gradients through the simulation as well as running the simulation on GPUs.


## Features
- Tight integration with [TensorFlow](https://www.tensorflow.org/) making network training easy
- All simulation steps are differentiable
- Fluid simulations can be run completely on the GPU
- Easy visualization of live data in the browser with a powerful interactive GUI
- Tweaking of network hyperparameters during training through the GUI
- Native support for n-dimensional fluid simulations
- Weak dependency on TensorFlow, allowing for execution of simulations with NumPy and SciPy
- Option of application deployment as Flask web-service


## Installation

The following commands will get you Φ<sub>*Flow*</sub> + browser-GUI + NumPy execution:

```
$ git clone https://gitlab.lrz.de/ga67fet/phiflow.git
$ pip install phiflow/[gui]
```

See the [detailed installation instructions](documentation/Installation_Instructions.md) on how to install Φ<sub>*Flow*</sub>
with TensorFlow support.


## Documentation and Guides

| [Index](documentation) | [Demos](demos) / [Tests](tests) | [Source](phi) |
|-------|---------------|--------|

If you would like to get right into it and have a look at some example code, check out the following files in the `demos` directory:

- [simpleplume.py](./demos/simpleplume.py): Runs a smoke simulation and displays it in the browser
- [optimize_pressure.py](./demos/optimize_pressure.py): Uses TensorFlow to optimize a velocity channel. TensorBoard can be started from the GUI and displays the loss.

The [simulation overview](documentation/Simulation_Overview.md) explains how to run predefined simulations using either the [NumPy or TensorFlow](documentation/NumPy_and_TensorFlow_Execution.md) backend. It also introduces the GUI.
The [simulation code design documentation](documentation/Simulation_Architecture.md) provides a deeper look into the object-oriented code design of simulations.

To learn how specific simulations are implemented, check out the documentation for [Smoke](documentation/Smoke_Simulation.md) or read about [staggered grids](documentation/Staggered_Grids.md) or [pressure solvers](documentation/Pressure_Solvers.md). 

[Writing a Φ<sub>*Flow*</sub> Application](documentation/Browser_GUI.md) introduces the high-level classes and expalins how to use the Φ<sub>*Flow*</sub> GUI for displaying a simulation.
For training machine learning models, [this document](documentation/Interactive_Training_Apps.md) gives an introduction int o writing a GUI-enabled application.


For I/O and data management, see the [data documentation](documentation/Reading_and_Writing_Data.md) or the [scene format specification](documentation/Scene_Format_Specification.md).

## Known Issues

GUI: Message not updating correctly on some Chrome installations on Windows.

TensorBoard: Live supervision does not work when running a local app that writes to a remote directory.