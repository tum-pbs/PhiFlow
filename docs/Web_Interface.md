# Φ<sub>Flow</sub> Web Interface

![Gui](figures/WebInterface.png)

Φ<sub>Flow</sub> provides an interactive web interface that can display 1D, 2D and 3D data.
The interface is displayed in the browser so it can be viewed remotely.

## Tabs & Features

The web interface consists of multiple tabs (web pages) which can be accessed at the upper left corner of each page.

- **Home** shows the title and description of the app. It allows the user to choose one field to view and to start/pause the app or step a single frame. Additionally all app-specific controls are shown at the bottom.
- **Side-by-Side** is similar to Home but shows two fields at a time instead of one.
- **Info** displays additional information about the current session such as file paths and run time.
- **Log** displays the complete application log.
- **Φ Board** contains benchmarking functionality. For TensorFlow apps it also allows the user to launch TensorBoard and run the TensorFlow profiler.
- **Help** refers to this page.

Tips & Tricks:

- You can run a specified number of frames by entering the number in the text box next to the 'Step' button. If you put a '*' before the number, it is multiplied by the app's `stride` value which can be found in the `info` tab.


## Displaying Module-level Variables

The class `ModuleViewer` is the simplest way to start the web interface but provides less flexibility than using the `App` class directly.
Since `ModuleViewer` is a subclass of `App`, much of its functionality still caries over, however.
Instantiating a `ModuleViewer` immediately launches the interface, displaying all module variables of type `Field`.

Example:
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80)
velocity = DOMAIN.staggered_grid(Noise())
pressure = DOMAIN.grid(0)

ModuleViewer('Module Viewer Example')
```
To have something happen when pressing *Step* or *Play*, write a loop using `ModuleViewer.range()`.
Each loop iteration is then executed as a step.
A sequence of such loops can also be used. Theses are then executed one after the other with the step index continuously increasing.

Example:
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise())
pressure = DOMAIN.grid(0)
for _ in ModuleViewer().range(100):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    velocity, pressure, iterations, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
```

## Frame Rate and Refresh Rate

The web interface provides a *Refresh rate* control above the field viewer.
This value describes how often the diagrams in the browser are refreshed.
It is independent of the frame rate of the app.

The frame rate of the app can be set in the constructor, `App.__init__(self, framerate=10)`.
To allow the user to control the frame rate, set it to `framerate=EditableInt('Framerate', 10, minmax=(1, 30))`.

The `stride` parameter is used to segment frames of the app.
For machine learning applications, this is equal to the number of steps in one epoch and influences how often validations are performed.
It has no effect on the interface, except that it is multiplied to the frame count when entering `*` in front of the number into the text box.

## Custom Controls

To create an interactive application, it is useful to add control element which the user can use to manipulate the simulation or training process.

Currently, the following control types are supported:

- Floating point controls (sliders)
- Integer controls (sliders)
- Boolean controls (checkboxes)
- Text controls (text fields)
- Action controls (buttons)

The easiest way to add controls is to create an attribute with the prefix `value_`.

```python
from phi.flow import *

app = App()
app.value_temperature = 39.5
app.value_windows_open = False
app.value_message = "It's too hot!"
app.value_limit = 42
show(app)
```

The type of controls is derived from the assigned value and the corresponding control element is automatically added to the GUI.

Analogously, actions are created from methods with the prefix `action_`.

```python
from phi.flow import *

app = App()
app.action_click_here = lambda: app.info('Thanks!')
```

Or for subclasses of `App`:
```python
    def action_click_here(self):
        self.info('Thanks!')
```

Controls can also be configured by creating an instance of `EditableValue`. In this case, the prefix `value_` is not needed.

```python
from phi.flow import *

app = App()
app.temperature = EditableFloat("Temperature", 40.2)
app.windows_open = EditableBool("Windows Open?", False)
app.message = EditableString("Message", "It's too hot!")
app.limit = EditableInt("Limit", 42, (20, 50))
```


## Configuration

The `show` method supports additional keyword arguments to configure how the App contents are displayed.

The `display` parameter defines which fields are displayed initially, e.g. `display='Density'` or `display=('Density', 'Velocity')`.

### Further configuration parameters

| Parameter            | Description                                                                                       | Default |
|----------------------|---------------------------------------------------------------------------------------------------|---------|
| external_web_server  | Whether an external tool is used to host the Dash server. If False, launches a new web server.    | False   |
| arrow_origin         | Which part of the arrow is centered at the position of the field. One of 'base', 'center', 'tip'. | 'tip'   |
| max_arrow_resolution | Downscale grids to have no more than resolution**2 cells before drawing arrows.                   | 40      |
| min_arrow_length     | Fraction of figure size. Smaller arrows are not drawn.                                            | 0.005   |
| max_arrows           | Maximum number of arrows to draw. Arrows are sorted by length.                                    | 2000    |
| draw_full_arrows     | Whether to draw the tips of arrows. If False, draws arrows as lines.                              | False   |
| colormap             | Built-in: 'viridisx', 'OrWhBl'. Additional colormaps can be used if matplotlib is installed.      | 'viridisx'|
| slow_colorbar        | If True, keeps colorbar similar to previous frame if possible.                                    | False   |
