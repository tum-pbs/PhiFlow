# Φ<sub>*Flow*</sub> Web Interface

![Gui](figures/WebInterface.png)

Φ<sub>*Flow*</sub> contains an interactive web interface that can display 1D, 2D and 3D data.
The interface is displayed in the browser so it can be used remotely.

*Note*:
If you intend to use the interface for interactive training of a TensorFlow model, make sure to read the 
[TensorFlow specific section](Interactive_Training_Apps.md) as well. Else, you can simply create a subclass of [App](../phi/app/app.py) as described below.

## Tabs & Features

The web interface consists of multiple tabs (web pages) which can be accessed from the top left.

- **Home** shows the title and description of the app. It allows the user to choose one field to view and to start/pause the app or step a single frame. Additionally all app-specific controls are shown at the bottom.
- **Side-by-Side** is similar to Home but shows two fields at a time instead of one.
- **Info** displays additional information about the current session such as file paths and run time.
- **Log** displays the complete application log.
- **Φ Board** contains benchmarking functionality. For TensorFlow apps it also allows the user to launch TensorBoard and run the TensorFlow profiler.
- **Help** refers to this page.

Tips & Tricks:

- You can run a specified number of frames by entering the number in the text box next to the 'Step' button. If you put a '*' before the number, it is multiplied by the app's `stride` value which can be found in the `info` tab.


## Launching the Web Interface

The web interface is based on [Dash by Plotly](https://plot.ly/dash/) which uses the popular framework [Flask](https://www.palletsprojects.com/p/flask/).

To launch the web interface, you have to create an instance of [`App`](../phi/app/app.py) and pass it to the `show` method.

While a direct instance of `App` can be used (with `force_show=True`), the preferred way is to create a subclass of `App`.
The following code snippet shows how to do this and launch the interface.

```python
from phi.flow import *

class HelloWorld(App):
    def __init__(self):
        App.__init__(self)

show()
```

When run, the application prints a URL to the console.
Enter this URL into a browser to view the interface.
The URL always stays the same, so you can simply refresh the web page when restarting your app.

You should see *Hello World*, the name of our app, at the top, followed by a diagram and the controls. None of the controls will do anything useful at this point so let's focus on the diagram.

A key part of any [App](../phi/app/app.py) is that it exposes a set of fields (NumPy arrays) which can change over time. How these fields are generated and how they evolve is up to the application. They could change as part of an evolving fluid simulation, they could be the predictions of a neural network that is being optimized or they could simply be a sequence read from disc.

In any case, these fields must be exposed to the GUI. This is done by calling the inherited `add_field()` method in the constructor. Its first argument is the name (can contain unicode characters) and the second is either a `numpy.ndarray` or a generator function without arguments that returns a `numpy.array`.
If a NumPy array is passed directly, the field cannot change over time.
A generator, typically a `lambda` expression, will be called regularly by the GUI for displaying the latest data.

A simple example, displaying randomly generated fields could look like this:

```python
from phi.flow import *

class GuiTest(App):
    def __init__(self):
        App.__init__(self, "Random", "Display random NumPy arrays!")
        self.add_field("Random Scalar", np.random.rand(1, 16, 16, 1))
        self.add_field("Random Vector", np.random.rand(1, 16, 16, 3))
        self.add_field("Evolving Scalar", lambda: np.random.rand(1, 16, 16, 1))

show()
```

On startup, the GUI will automatically display the fields.

## The `step()` Method

The `step()` method is the core part of the model. It defines how the next update is calculated. This could mean running one simulation step to advance the state of the physical system in time, loading data from disc or running a training pass for a neural network.
If not overridden, `step()` calls `world.step()` on the global `world` object.

Your subclass of [App](../phi/app/app.py) automatically inherits the variable `time` which holds the current frame as an integer (`time=0` for the first call). It is automatically incremented after step is called and is displayed in the GUI, below the diagrams.

After `step()` finishes, the GUI is updated to reflect the change in the data. Consequently, the channel generators (`numpy,ndarray`'s and `lambda` expressions in the above example) can be called after each step. In practice, however, steps can often be performed at a higher framerate than the GUI update rate.

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
    self.value_temperature = 39.5
    self.value_windows_open = False
    self.value_message = "It's too hot!"
    self.value_limit = 42
```

The type of controls is derived from the assigned value and the corresponding control element is automatically added to the GUI.

Analogously, actions are created from methods with the prefix `action_`.

```python
    def action_click_here(self):
        self.info('Thanks!')
```

Controls can also be configured by creating an instance of `EditableValue`. In this case, the prefix `value_` is not needed.

```python
    self.temperature = EditableFloat("Temperature", 40.2)
    self.windows_open = EditableBool("Windows Open?", False)
    self.message = EditableString("Message", "It's too hot!")
    self.limit = EditableInt("Limit", 42, (20, 50))
```


## Configuration

The `show` method supports additional keyword arguments to configure how the App contents are displayed.

The `display` parameter defines which fields are displayed initially, e.g. `display='Density'` or `display=('Density', 'Velocity')`.

### Further configuration parameters

| Parameter            | Description                                                                                       | Default |
|----------------------|---------------------------------------------------------------------------------------------------|---------|
| arrow_origin         | Which part of the arrow is centered at the position of the field. One of 'base', 'center', 'tip'. | 'tip'   |
| max_arrow_resolution | Downscale grids to have no more than resolution**2 cells before drawing arrows.                   | 40      |
| max_arrows           | Maximum number of arrows to draw. Arrows are sorted by length.                                    | 300     |
| draw_full_arrows     | Whether to draw the tips of arrows. If False, draws arrows as lines.                              | False   |

