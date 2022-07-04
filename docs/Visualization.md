# Visualization


## Interactive User Interface
Φ<sub>Flow</sub> provides interactive user interfaces for live visualization and control.
They allow you to see neural network predictions as the network is being trained or view training data while it is being generated.

The following interfaces are currently available:

- [`dash`](Web_Interface.md) web interface hosts a web server
- [`console`](ConsoleUI.md) for command line applications

### Viewing Data
The primary way to launch an interactive user interface is through
[`view()`](phi/vis/index.html#phi.vis.view).
This function takes a number of `SampledField` instances or variable names and shows them in a new UI.
```python
from phi.flow import *

data = Domain(x=32, y=32).scalar_grid(Noise())
view(data, gui=None)
```
With `gui=None` (default) an appropriate interface is automatically selected based on the environment and installed packages.
Otherwise, the type of interface can be specified with `gui='dash'` or `gui='console'`.

### Batch Selection
When the viewed data are batched, the GUI will try to plot all examples at once which may result in crowded plots.
This can be avoided using the `select` keyword argument.
It specifies dimensions along which a single slice is displayed, e.g.
```python
data = Domain(x=32, y=32).scalar_grid(Noise(batch=10, time=33))
view(batched_data, select='batch,time')
```
For each dimension, a slider will be added to the UI for the user to select which slice to display.

### Controlling Loop Execution
The GUI can be used to let the user control loop execution.
To do this, the user code needs to iterate over 
[`Viewer.range()`](phi/vis/index.html#phi.vis.Viewer.range).
```python
data = Domain(x=32, y=32).scalar_grid(Noise())
for _ in view(data, play=False).range(10):
    data = physics(data)
```
Once the loop is encountered, the execution controls of the GUI can be used to pause execution, run single iterations or break the loop.
With `play=False`, execution stops immediately once the loop is hit.

The GUI will update the displayed values either after each iteration, or at a configurable refresh rate.

### Loop Recording
The GUI can also be used to record the values of the viewed variables during loop execution.
Recording is enabled if the stop argument in
[`Viewer.range()`](phi/vis/index.html#phi.vis.Viewer.range)
is named, e.g. `viewer.range(frames=10)`.
Then, the values for each viewed variable are accumulated and stacked along a new batch dimension with this name.

The recorded values can later be accessed through `Viewer.rec.<variable>`.
```python
data = Domain(x=32, y=32).scalar_grid(Noise())
viewer = view(data)
for _ in viewer.range(frames=10):
    data = physics(data)
all_data = viewer.rec.data  # CenteredGrid (frames=11, x=32, y=32)
```


### Custom Controls
It is often useful to modify parameters while a script is running,
e.g. adjusting the learning rate to see which values work best.
This can be easily achieved using [`control()`](phi/vis/#phi.vis.control).
```python
learning_rate = control(0.001)
checkpoint_interval = control(100, (1, 200))
```
`control()` returns the first argument which specifies the initial value.
Consequently, `learning_rate` is a `float` and can be used as such.
The value of the variable will be modified each time the user edits the value.
This can happen while the user code is running.

The following control types are supported:

- Linear sliders for `int` and `float` with a suitable value range
- Logarithmic sliders for `float` values with a large range or no range specified.
- Checkboxes for `bool` values.
- Text fields for `str` values.
  
Additionally, buttons are generated for functions declared in the user script that take no parameters.


## Plotting
See [`plot()`](phi/vis/index.html#phi.vis.plot) and
[`plot_scalars()`](phi/vis/index.html#phi.vis.plot_scalars).