# Interactive training or optimization


If the purpose of your application is to train a TensorFlow model, you can extend from [TFModel](../phi/tf/model.py) instead. This has a couple of benefits:

- Model parameters can be saved and loaded from the GUI
- Summaries including the loss value are created and can easily be extended using `add_scalar`
- [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) can be launched from the GUI
- Profiling tools can be used in the browser
- A database is set up (see [the guide on data handling](data.md))
- The `step` method is implemented by default
- Tensor nodes and database channel names can be passed to `add_field`
- Properties and the application source file are written to the output directory
- A learning rate control is available by default and more controls can be created easily using `editable_float`, `editable_int`

### Simple Example

The following example trains a neural network, referenced as `network` to predict a force channel from two velocity fields.

```python
from phi.tf.model import *
from phi.data import *

class TrainingTest(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Training")
        sim = self.sim = TFFluidSimulation([128] * 2, "open")

        initial_velocity = sim.placeholder("velocity", "InitialVelocity")
        target_velocity = sim.placeholder("velocity", "TargetVelocity")
        true_force = sim.placeholder("velocity", "Force") * self.editable_float("Scale", 1.0)

        with self.model_scope():
            pred_force = network(initial_velocity, target_velocity)
        loss = l2_loss(pred_force - true_force)
        self.add_objective("Supervised_Loss", loss)

        self.database.add(["InitialVelocity", "TargetVelocity", "Force"])
        self.database.put_scenes(scenes("SmokeIK/forces"), logf=self.info)
        self.finalize_setup([initial_velocity, target_velocity, true_force])

        self.add_field("Force (Ground Truth)", "Force")
        self.add_field("Force (Model)", pred_force)

app = TrainingTest().show(production=__name__!="__main__")
```

Let's go over what's happening here in detail.
First, the app calls the super constructor, passing only the app's name.
Next, the fluid simulation is created and assigned to `self.sim`. This variable is inherited from `TFModel` and must be initialized with a `TFFluidSimulation` in the constructor.

The following three lines create input fields for TensorFlow's graph. We allow the true_force tensor to be scaled by a user-defined value which can be set in the GUI.

Now that the network inputs are set up, the network can be built. The use of `with self.model_scope()` ensures that the network parameters can be saved and loaded automatically and from the GUI.
The `l2_loss` is part of Φ<sub>*Flow*</sub>'s n-d math package but a regular TensorFlow loss can also be used.
The inherited method `add_objective` sets up the optimizer. This optimizer will be used in the default `step` implementation.

The following block sets up the database by registering the required fields and adding all scenes from one category (see [the data documentation](data.md) for more).
The call to `finalize_setup` is mandatory in the constructor and sets up the TensorFlow summary as well as database iterators.

Finally, the viewable fields are exposed to the GUI. The first line exposes the channel `Force` which was registered with the database while the second line exposes the graph output `pred_force` which will be recalculated each time the GUI is updated.

Lastly, the app is instantiated and the GUI created in the same way as with a [FieldSequenceModel](../phi/model.py).

