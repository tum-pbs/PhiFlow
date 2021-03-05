
[**Homepage**](https://github.com/tum-pbs/PhiFlow)
&nbsp;&nbsp;&nbsp; [**API**](phi)
&nbsp;&nbsp;&nbsp; [**Demos**](https://github.com/tum-pbs/PhiFlow/tree/develop/demos)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Fluids Tutorial**](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Playground**](https://colab.research.google.com/drive/1zBlQbmNguRt-Vt332YvdTqlV4DBcus2S#offline=true&sandboxMode=true)

The
[Colab notebook](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true)
on fluid simulations is a great place to get started with Φ<sub>Flow</sub>.
Also check out the [demos](https://github.com/tum-pbs/PhiFlow/tree/develop/demos/).

### Module Overview

| Module API  | Documentation                                        |
|-------------|------------------------------------------------------|
| [phi.app](phi/app)     | [Web interface](Web_Interface.md): Interactive application development   |
| [phi.physics](phi/physics) <br /> [phi.physics.advect](phi/physics/advect.html) <br /> [phi.physics.fluid](phi/physics/fluid.html) <br /> [phi.physics.diffuse](phi/physics/diffuse.html) | [Overview](Physics.md): Domains, built-in physics functions <br /> [Writing Fluid Simulations](Fluid_Simulation.md): Advection, projection, diffusion        |
| [phi.field](phi/field)   | [Overview](Fields.md): Grids, particles <br /> [Staggered Grids](Staggered_Grids.md): Data layout, usage  <br /> [Reading and Writing Simulation Data](Reading_and_Writing_Data.md) |
| [phi.geom](phi/geom)    | [Overview](Geometry.md): Differentiable Geometry        |
| [phi.math](phi/math) <br /> [phi.math.backend](phi/math/backend) <br /> [phi.math.extrapolation](phi/math/extrapolation.html)  | [Overview](Math.md): Named dimensions, backends, indexing, non-uniform tensors, precision <br /> [Optimizing Performance](GPU_Execution.md): GPU, graph mode, profiler <br /> [Dictionary](Math_Translations.md): NumPy / TensorFlow / PyTorch / Jax cheat sheet|

### API Documentation

The [API documentation](phi) is generated using [pdoc](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force phi
```

### Other Documentation

* [Installation Instructions](Installation_Instructions.md):
  Requirements, installation, CUDA compilation
* [Contributing to Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow/blob/develop/CONTRIBUTING.md):
  Style guide, docstrings, commit tags
* [Scene Format Specification](Scene_Format_Specification.md):
  Directory layout, file format
