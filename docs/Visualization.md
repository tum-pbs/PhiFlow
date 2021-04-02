# Visualization


## Interactive User Interface
Φ<sub>Flow</sub> provides interactive user interfaces for live visualization and control.
They allow you to see neural network predictions as the network is being trained or view training data while it is being generated.

The following interfaces are currently available:

- [`dash`](Web_Interface.md) web interface hosts a web server
- [`widgets`](Widgets.md) for Jupyter notebooks
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
With `gui=None` an appropriate interface is automatically selected based on the environment and installed packages.
Otherwise, the type of interface can be specified with `gui='dash'`, `gui='widgets'` or `gui='console'`.

### Viewing Sequences
```python
from phi.flow import *

data = Domain(x=32, y=32).scalar_grid(Noise())
for _ in view(data, gui=None).range(10):
    data = field.laplace(field)
```

Recording / Slider
`range(frames=10)`


## Plotting
See [`plot()`](phi/vis/index.html#phi.vis.plot) and
[`plot_scalars()`](phi/vis/index.html#phi.vis.plot_scalars).