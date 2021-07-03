# Reading and Writing Simulation Data
This document describes how simulation data can be written and read using Φ<sub>Flow</sub>.
The values format itself is described in the [scene format specification](Scene_Format_Specification.md).

## Referencing a Scene Object
Scenes are represented by instances of [`Scene`](phi/app/index.html#phi.app.Scene).
There are two possibilities to reference existing scenes:

```python
from phi.flow import *


scene = Scene.at('~/data/sim_000000')  # reference an existing scene by full path
scene = Scene.at('~/data', 0)  # reference an existing scene by directory and id
scenes = Scene.list('~/data')  # list all scenes in directory
```

New scenes can be created using the function [`Scene.create`](phi/app/index.html#phi.app.Scene.create)
which creates a new folder in the directory.
By default, it also copies the python script that created that scene into the included `src` folder.

## Writing to a Scene
The simulation data are stored as in individual files, one for each quantity and frame.
Frames are integers (up to 1000000) that typically represent time steps of a simulation.
They can, however, be used for any purpose.

Instead of writing tensors directly, the scene writes instances of [`SampledField`](phi/field/index.html#phi.field.SampledField).
These also encode physical size and extrapolation.
The method [`write()`](phi/app/index.html#phi.app.Scene.write) stores a dictionary containing fields and their names.
```python
scene.write({'smoke': smoke, 'velocity': staggered_velocity}, frame=0)
```

Subdirectories in the scene can be created using the method [`subpath()`](phi/app/index.html#phi.app.Scene.subpath).

The methods `copy_calling_script()` and `copy_src` can be used to copy source files into the `src` folder of a scene.

## Reading from a Scene
Similar to `write()`, the method [`read()`](phi/app/index.html#phi.app.Scene.read) loads `Field` objects that were previously stored.
It can be used to read single fields or multiple fields.
```python
smoke = scene.read('smoke', frame=0, convert_to_backend=False)
smoke, velocity_staggered = scene.read(['density', 'velocity'], frame=0)
```
The `convert_to_backend` argument determines how the loaded data is stored.
If `True`, the default backend is used to create the tensors, e.g. TensorFlow or PyTorch.
If `False`, the loaded data will be held as NumPy arrays and future operations will also use NumPy functions unless converted manually.

## Properties
Scenes may contain a property file named `description.json` that stores information about the simulation in an easy-to-parse format.
For further documentation, see the [scene format specification](Scene_Format_Specification.md).

Properties can be set using `put_property` and read from `scene.properties`.


## Batches of Scenes
As discussed in the [performance optimization guide](GPU_Execution.md), it is recommended to combine data into large tensors whenever possible.
For example, multiple smoke simulations can be simulated in a data-parallel way by stacking the tensors of the different simulation, assuming all have the same resolution.

To facilitate reading and writing batched data, scenes have a batch mode.
A batch of scenes can be created using [`create`](phi/app/index.html#phi.app.Scene.create) with `count>1`.
```python
scenes = Scene.create('~/data', count=batch_size, batch_dim='batch')
```
When writing data using `scenes.write()`, the fields are unstacked along *batch_dim* and written to the corresponding scenes.

Similarly, reading from a batch of scenes stacks the loaded fields along the batch dimension.
