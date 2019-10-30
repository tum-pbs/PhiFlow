# Φ<sub>*Flow*</sub> Browser GUI

![Gui](./figures/Gui.png)

Φ<sub>*Flow*</sub> contains an interactive GUI that can display any kind of two-dimensional or three-dimensional fields. The interface is displayed in the browser and is very easy to set up.

The default GUI will display your application in the browser.
If you intend to use the GUI for interactive training of a TensorFlow model, make sure to read the 
[TensorFlow specific section](Interactive_Training_Apps.md) as well. Else, you can simply create a subclass of [App](../phi/app/app.py) as described below.

## Launching the GUI

The following code defines a custom [App](../phi/app/app.py) and uses it to launch the GUI.

```python
from phi.flow import *

class GuiTest(App):
    def __init__(self):
        App.__init__(self, 'Hello World!')

show()
```

When run, the application prints a URL to the console. Enter this URL into a browser to view the GUI.

You should see the _Hello World_ text at the top, followed by two empty diagrams and a bunch of controls. None of the controls will do anything useful at this point so let's focus on the diagrams.

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

The `step()` method is the core part of the model. It defines how the next step is calculated. This could mean running one simulation step, loading data from disc or running a training pass for a neural network.
If not overridden, `step()` calls `world.step()` on the global `world` object.

Your subclass of [App](../phi/app/app.py) automatically inherits the variable `time` which holds the current frame as an integer (`time=0` for the first call). It is automatically incremented after step is called and is displayed in the GUI, below the diagrams.

After `step()` finishes, the GUI is updated to reflect the change in the data. Consequently, the channel generators (`numpy,ndarray`'s and `lambda` expressions in the above example) can be called after each step. In practice, however, steps can often be performed at a higher framerate than the GUI update rate.

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
