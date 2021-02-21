
[**Homepage**](https://github.com/tum-pbs/PhiFlow)
&nbsp;&nbsp;&nbsp; [**API**](phi)
&nbsp;&nbsp;&nbsp; [**Demos**](https://github.com/tum-pbs/PhiFlow/tree/develop/demos)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Fluids Tutorial**](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Playground**](https://colab.research.google.com/drive/1zBlQbmNguRt-Vt332YvdTqlV4DBcus2S#offline=true&sandboxMode=true)

If you would like to get right into it and have a look at some code, check out the
[tutorial notebook on Google Colab](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true).
It lets you run fluid simulations with Φ<sub>Flow</sub> in the browser.
Also check out the [explanation of common fluid simulation operations](https://tum-pbs.github.io/PhiFlow/Fluid_Simulation.html).

The following introductory demos are also helpful to get started with writing your own scripts using Φ<sub>Flow</sub>:

* [smoke_plume.py](https://github.com/tum-pbs/PhiFlow/tree/develop/demos/smoke_plume.py) runs a smoke simulation and displays it in the web interface.
* [optimize_pressure.py](https://github.com/tum-pbs/PhiFlow/tree/develop/demos/differentiate_pressure.py) uses TensorFlow to optimize a velocity field and displays it in the web interface.

### Module Overview

| Module      | Documentation                                        |
|-------------|------------------------------------------------------|
| [phi.app](phi/app)     | [Web interface](Web_Interface.md): Interactive application development <br /> [Reading and Writing Simulation Data](Reading_and_Writing_Data.md)   |
| [phi.physics](phi/physics) <br /> [phi.physics.advect](phi/physics/advect.html) <br /> [phi.physics.fluid](phi/physics/fluid.html) <br /> [phi.physics.diffuse](phi/physics/diffuse.html) | [Overview](Physics.md): Domains, built-in physics functions <br /> [Writing Fluid Simulations](Fluid_Simulation.md): Advection, projection, diffusion        |
| [phi.field](phi/field)   | [Overview](Fields.md): Grids, particles <br /> [Staggered Grids](Staggered_Grids.md): Data layout, usage           |
| [phi.geom](phi/geom)    | [Differentiable Geometry](Geometry.md)                              |
| [phi.math](phi/math) <br /> [phi.math.backend](phi/math/backend) <br /> [phi.math.extrapolation](phi/math/extrapolation.html)  | [Overview](Math.md): Named dimensions, backends, indexing, non-uniform tensors, precision <br /> [Optimizing Performance](GPU_Execution.md): GPU, graph mode, profiler |

### API Documentation

The [API documentation](phi) is generated using [pdoc](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force phi
```

### Other Documentation

* [Installation Instructions](Installation_Instructions.md)
* [Scene Format Specification](Scene_Format_Specification.md)
