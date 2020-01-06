# Reading and writing simulation data

This document describes how simulation data can be written and read using Φ<sub>*Flow*</sub>.
The data format itself is described in the [data format specification](Scene_Format_Specification.md).

## Referencing a scene object

The fluid I/O functionality of Φ<sub>*Flow*</sub> is located in [phi.data.fluidformat](../phi/data/fluidformat.py).

There are two possibilities to reference existing scenes:

```python
from phi.flow import *

# reference a specific scene
scene = Scene.at('~/phi/data/simpleplume/sim_000000')

# list all scenes in a category
scenes = Scene.list('~/phi/data/simpleplume')
```

New scenes can be created using the function `Scene.create` which appends a new scene to a category.
It also copies the python script that created that scene into the `src` folder unless otherwise specified.

## Reading from a specific scene

The main simulation data is stored in individual files, one for each field and frame.
Other properties are listed in the accompanying `description.json` (see the [data format specification](Scene_Format_Specification.md)).

The following table gives an overview of what information can be obtained from a `scene` object.

| Property            | Description                                            |
|---------------------|--------------------------------------------------------|
| `scene.path`        | file path to the scene                                 |
| `scene.category`    | name of the category                                   |
| `scene.dir`         | directory containing the category                      |
| `scene.index`       | index within the category                              |
| `scene.properties`  | dict containing the values stored in description.json  |
| `scene.frames`      | list of all frames contained in the scene              |
| `scene.fieldnames`  | list of all fields contained in the scene              |

The scene provides a couple of methods to read simulation data from a scene.

```python
from phi.flow import *

# Create Scene
scene = Scene.at('~/phi/data/simpleplume/sim_000000')

# Read a single NumPy array from the associated file
density = scene.read_array(fieldname='density', frame=0)  

# Read multiple fieldnames, concatenating the frames in the batch dimension
densities, velocities = scene.read_sim_frames(fieldnames=['density', 'velocity'], frames=range(16))

# Read a state or struct
fluid = scene.read(Fluid(...), frame=0)
```

The last call makes use of Φ<sub>*Flow*</sub>'s [`struct` system](Structs.ipynb).

## Writing to a scene

The following methods can be used to store simulation data in a scene.

```python
from phi.flow import *

# Create Scene
scene = Scene.create('~/phi/data/simpleplume')

# Write one frame with multiple fields
scene.write_sim_frame([density, velocity], ['density', 'velocity'], frame=0)

# Write a state
scene.write(Fluid(...), frame=0)
```

Subdirectories in the scene can be created using the method `subpath(name, create)`.

To copy source files into the `src` folder of a scene:

```python
scene.copy_calling_script()
scene.copy_src(path)
```

## Reading from a set of scenes

Machine learning applications usually iterate over a large number of scenes while training models.
Φ<sub>*Flow*</sub> provides a data management system to simplify data handling.

```python
from phi.flow import *

whole_dataset = Dataset.load('~/phi/data/simpleplume')
training_data = Dataset.load('~/phi/data/simpleplume', range(1000), name='train')
```

Classes that extend [`LearningApp`](../phi/tf/app.py) only need to call `self.set_data`, passing a training and validation dataset as well as a struct containing TensorFlow placeholders (see the [documentation](Interactive_Training_Apps.md)).

### Channels

Registering and processing fields

### Iterating over data

BatchReader

## Data augmentation

Data augmentation is implemented like any other transform -- using channels.
