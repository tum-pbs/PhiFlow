
# Data

## Scene data format

Φ<sub>*Flow*</sub> uses a NumPy based format which is also being implemented into MantaFlow.

It specifies the following elements:

- Arrays hold n-dimensional fields at one point in time (frame)
- Scenes contain arrays for different properties and at different frames
- Categories enumerate scenes with similar properties

### Arrays

Arrays hold the spatial distribution of one property of the simulation at a certain frame.
Each array is stored as one compressed NumPy (.npz) file. The contained NumPy array has the shape `(height, width, components)` in 2D and `(depth, height, width, components)` in 3D where `components` refers to number of dimensions stored per cell, 1 for scalar fields such as density, 2 for vector fields in 2D, 3 for vector fields in 3D.

The spatial size of the arrays is not the same as the number of valid voxels in the simulation.
In the following, `x`, `y` and `z` refer to the shape of valid entries in centered fields like state density.
There are two conventions for how the array sizes can be derived from these:

- Mantaflow: The outer voxels of centered fields are invalid (depth=z+2, height=y+2, widht=x+2). Staggered grids store the component at the lower face of the cell with same index (depth=z+2, height=y+2, widht=x+2), the top rows are invalid.
- Φ<sub>*Flow*</sub>: The arrays of centered fields store only valid values (depth=z, height=y, width=x). Staggered grids store the component at the lower face of the cell with same index, the top-most rows hold partly invalid values (depth=z+1, height=y+1, width=x+1).

The filename of an array at a certain frame is:

```
<Property>_<frame>.npz
```

The property name starts with an upper case letter and the frame index is padded by zeros forming a 6-digit string.

Examples: `Density_000000.npz`,  `Velocity_000042.npz`


### Scenes

A scene is a directory that directly contains all arrays associated with the simulation.
Its name starts with `sim_` followed by a six-digit scene index.

In addition to arrays, scenes store properties of the simulation in a JSON file called `description.json`.
This file describes both simulation properties and the origin of the data. It can store any number of properties but the excerpt below can be used as a reference.

```json
{
  "instigator": "FieldSequenceModel",
  "app": "smokesm_refine.py",
  "app_path": "/home/holl/phiflow/apps/smokesm_refine.py",
  "name": "Refine-SmokeSM 32",
  "description": "Slow motion density generation CNN training",
  "summary": "Refine-SmokeSM 32",
  "time_of_writing": 1,
  "dimensions": [
    128,
    128
  ],
  "rank": 2,
  "batch_size": null,
  "solver": "Sparse Conjugate Gradient",
  "open_boundary": true,
  "gravity": [
    -9.81,
    0.0
  ],
  "buoyancy_factor": 0.01,
  "parameter_count": 71157
}
```

Here, `dimensions` is equal to shape of centered fields, stored as described above.

Source files can be added to an optional `src` directory, images to an optional `images` directory.
Optionally, a log file can be added to the scene directory as `info.log`.

Scenes can contain any number of additional subdirectories for specific information.

### Categories

Categories are used to organize scenes that belong together. Categories are realized simply as a directory that contains scene directories. It can have any name.
Scenes within that directory are assumed to have the same properties and the same number of frames but these restrictions are optional.


## Handling fluid data I/O in Φ<sub>*Flow*</sub>

The fluid I/O functionality of Φ<sub>*Flow*</sub> is located in [phi.fluidformat](../phi/fluidformat.py).

There are two possibilities to reference existing scenes:

```python
from phi.fluidformat import *

scene = scene_at(scene_directory)  # reference a specific scene
all_in_category = scenes(category_directory)  # list all scenes in category
```

New scenes can be created using the function

```python
new_scene(category_directory)  # Uses the next free index
```

Once a scene object is obtained, fluid data and other properties can be written and read from the scene.
Reading data can be achieved with one of the following:

```python
array = scene.read_array(fieldname, index)  # Locates the desired npz file and returns the contents as a numpy array

scene.read_sim_frames(fieldnames, frames)  # Reads multiple fieldnames, concatenating the frames in the batch dimension
densities, velocities = scene.read_sim_frames(["Density, Velocity"], range(10))  # Reads the first 10 densities and velocities
```

Writing a frame to a scene can be done as follows:

```python
write_sim_frame(scene.path, arrays, fieldnames, index)  # Writes multiple properties at one frame
```

To obtain information about a scene:

| Property  | Description  |
|---|---|
| scene.path  | file path to the scene  |
| scene.category  | name of the category  |
| scene.dir  | directory containing the category  |
| scene.index  | index within the category  |
|  scene.properties | dict containing the values stored in description.json  |
| scene.frames  | list of all frames contained in the scene  |
| scene.fieldnames  | list of all fields contained in the scene  |

Subdirectories in the scene can be created using

```python
scene.subpath(name, create=True)  # Create a folder within the scene of given name
```

To copy source files into the scene:

```python
scene.copy_calling_script()
scene.copy_src(path)
```

This automatically creates a `src` folder within the scene and places the specified source files within.


## Managing the data

Φ<sub>*Flow*</sub>'s built-in [Database](../phi/data/data.py) class is the primary way to organize large quantities of data. It supports on-the-fly loading of an arbitrary amount of scenes, organized into multiple datasets.

Subclasses of [TFModel](../phi/tf/model.py) are supplied with an instance of database, referenced as `self.database`. This database is set up with a training and validation dataset.

### Adding data

Scenes are the primary way to add data to a database.

```python
database.put_scene(scene, frames=None, dataset=None, logf=None)  # Add a single scene
database.put_scenes(scenes, per_scene_indices=None, dataset=None, allow_scene_split=False, logf=None)  # Add multiple scenes
```

In both cases, the frames to be loaded from the scenes can be specified. If None, all available frames are used. This may be slow for large quantities of scenes since all scenes need to be browsed to compute the total number of frames.

If a dataset is specified, all frames are added to that dataset.
If the dataset is not specified (and `allow_scene_split=True`), the frames of a scene are split and distributed among all datasets.
If the dataset is not specified for `put_scenes` and `allow_scene_split=False`, frames of one scene are kept together and scenes are distributed among the datasets.

The `logf` parameter is used to log the actions. For subclasses of [FieldSequenceModel](../phi/model.py) or [TFModel](../phi/tf/model.py), passing `self.info` ensures the results are recorded.

Data can also be supplied as a generator using `database.put_generated(...)`.


### Channels

Registering and processing fields

### Iterating over data

BatchIterator...

## Data augmentation

Data augmentation is implemented like any other transform -- using channels.