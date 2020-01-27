# Data format specification

Φ<sub>*Flow*</sub> and [MantaFlow](http://mantaflow.com/) use a similar format to store simulation data.
This document explains the file structure of this format. If you only want to read an write data without knowing the specifics, check out the [data documentation](Reading_and_Writing_Data.md).

The data format specifies the following elements:

- Arrays hold n-dimensional fields at one point in time (frame)
- Scenes contain arrays for different properties and at different frames
- Categories enumerate scenes with similar properties

## Arrays

Arrays hold the spatial distribution of one property of the simulation at a certain frame.
Each array is stored as one compressed NumPy (`.npz`) file.
The contained NumPy array has the shape `(height, width, components)` in 2D and `(depth, height, width, components)` in 3D where `components` refers to number of dimensions stored per cell,
1 for scalar fields such as density,
2 for vector fields in 2D,
3 for vector fields in 3D.
I.e., the formats are [ZYXC] in 3D, and [YXC] in 2D. As `npz` files can contain multiple arrays, the last entry with a file has to contain the array data to be loaded.

The spatial size of the arrays is not the same as the number of valid voxels in the simulation.
In the following, `x`, `y` and `z` refer to the shape of valid entries in centered fields like state density.
There are two conventions for how the array sizes can be derived from these:

- Mantaflow: The outer voxels of centered fields are invalid (depth=z+2, height=y+2, width=x+2). Staggered grids store the component at the lower face of the cell with same index (depth=z+2, height=y+2, width=x+2), the top rows are invalid.
- Φ<sub>*Flow*</sub>: The arrays of centered fields store only valid values (depth=z, height=y, width=x). Staggered grids store the component at the lower face of the cell with same index, the top-most rows hold partly invalid values (depth=z+1, height=y+1, width=x+1).

The filename of an array at a certain frame is:

```bash
<Property>_<frame>.npz
```

The property name starts with an upper case letter and the frame index is padded by zeros forming a 6-digit string.

Examples: `Density_000000.npz`,  `Velocity_000042.npz`

## Scenes

A scene is a directory that directly contains all arrays associated with the simulation.
Its name starts with `sim_` followed by a six-digit scene index.

In addition to arrays, scenes store properties of the simulation in a JSON file called `description.json`.
This file describes both simulation properties and the origin of the data.
It can store any number of properties, depending on the application.

The following content was created by running the [simpleplume.py](../demos/simpleplume.py) demo and can be used as a reference.

```json
{
  "instigator": "App",
  "traits": [],
  "app": "simpleplume.py",
  "app_path": "~/phiflow/demos/simpleplume.py",
  "name": "Simpleplume",
  "description": "",
  "all_fields": [
    "Density",
    "Velocity"
  ],
  "actions": [],
  "controls": [],
  "summary": "Simpleplume",
  "time_of_writing": 0,
  "world": {
    "age": 0.0,
    "states": [
      {
        "age": 0.0,
        "domain": {
          "grid": {
            "dimensions": [
              80,
              64
            ],
            "box": {
              "origin": [
                0,
                0
              ],
              "size": [
                80,
                64
              ],
              "type": "Box",
              "module": "phi.physics.geom"
            },
            "type": "Grid",
            "module": "phi.physics.geom"
          },
          "boundaries": {
            "solid": true,
            "friction": 0.0,
            "extrapolate_fluid": true,
            "global_velocity": 0.0,
            "local_velocity": 0.0,
            "type": "Material",
            "module": "phi.physics.material"
          },
          "type": "Domain",
          "module": "phi.physics.domain"
        },
        "gravity": -9.81,
        "buoyancy_factor": 0.1,
        "conserve_density": false,
        "type": "Smoke",
        "module": "phi.physics.smoke"
      },
      {
        "age": 0.0,
        "geometry": {
          "center": [
            10,
            32
          ],
          "radius": 5,
          "type": "Sphere",
          "module": "phi.physics.geom"
        },
        "velocity": 0,
        "rate": 0.2,
        "type": "Inflow",
        "module": "phi.physics.objects"
      }
    ],
    "type": "StateCollection",
    "module": "phi.physics.collective"
  }
}
```

Φ<sub>*Flow*</sub> writes all simulation properties into the category `world`.

Source files can be added to an optional `src` directory, images to an optional `images` directory.
Optionally, a log file can be added to the scene directory as `info.log`.

Scenes can contain any number of additional subdirectories for specific information.

## Categories

Categories are used to organize scenes that belong together. Categories are realized simply as a directory that contains scene directories. It can have any name.
Scenes within that directory are assumed to have the same properties and the same number of frames but these restrictions are optional.
