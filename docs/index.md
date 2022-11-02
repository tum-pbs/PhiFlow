# ![PhiFlow](figures/Logo_DallE2_3_layout.png)

[**Homepage**](https://github.com/tum-pbs/PhiFlow)
&nbsp;&nbsp;&nbsp; [**API**](phi)
&nbsp;&nbsp;&nbsp; [**Demos**](https://github.com/tum-pbs/PhiFlow/tree/develop/demos)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Playground**](https://colab.research.google.com/drive/1zBlQbmNguRt-Vt332YvdTqlV4DBcus2S#offline=true&sandboxMode=true)


### Beginner Guides

[Installation Instructions](Installation_Instructions.md):
  Requirements, installation, CUDA compilation

[Cookbook](Cookbook.html): Example code snippets for various things

#### Tensors and Dimensions

* [▶️ Introduction Video](https://youtu.be/4nYwL8ZZDK8)
* [Introduction Notebook](Math_Introduction.html)

#### Differentiation

* [Learning to Throw](Learn_to_Throw_Tutorial.html)
* [Billiards](Billiards.html)
* [Differentiable fluid simulations](Fluids_Tutorial.html)

#### Fluids

* [▶️ Introduction Video](https://youtu.be/YRi_c0v3HKs)
* [Differentiable fluid simulations](Fluids_Tutorial.html)
* [Batched Obstacles](Batched_Obstacles.html)

#### I/O

* [Introduction to Scenes](IO_with_Scenes.html)

#### Visualization

* [Solar System](Planets_Tutorial.html)
* [Animation Gallery](Animations.html)

#### Neural Networks

* [▶️ Introduction Video](https://youtu.be/aNigTqklCBc)
* [Learning to Throw](https://tum-pbs.github.io/PhiFlow/Learn_to_Throw_Tutorial.html)


#### Advanced

* [What to Avoid](Known_Issues.html): How to keep your code compatible with PyTorch, TensorFlow and Jax


### Module Documentation

| Module API                                                                                                                                                                                                                                 | Documentation                                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [phi.vis](phi/vis)                                                                                                                                                                                                                         | [Visualization](Visualization.md): Plotting, interactive user interfaces <br /> [Dash](Web_Interface.md): Web interface <br /> [Console](ConsoleUI.md): Command line interface                                                                                                         |
| [phi.physics](phi/physics) <br /> [phi.physics.advect](phi/physics/advect.html) <br /> [phi.physics.fluid](phi/physics/fluid.html) <br /> [phi.physics.diffuse](phi/physics/diffuse.html) <br /> [phi.physics.flip](phi/physics/flip.html) | [Fluids Tutorial](Fluids_Tutorial.html): Introduction to core classes and fluid-related functions. <br /> [Overview](Physics.md): Domains, built-in physics functions <br /> [Functions for Fluid Simulations](Fluid_Simulation.md): Advection, projection, diffusion                  |
| [phi.field](phi/field)                                                                                                                                                                                                                     | [Overview](Fields.md): Grids, particles <br /> [Staggered Grids](Staggered_Grids.html): Data layout, usage <br /> [Reading and Writing Simulation Data](Reading_and_Writing_Data.md) <br /> [Scene Format Specification](Scene_Format_Specification.md): Directory layout, file format |
| [phi.geom](phi/geom)                                                                                                                                                                                                                       | [Overview](Geometry.md): Differentiable Geometry                                                                                                                                                                                                                                       |
| [phi.math](phi/math) <br /> [phi.math.backend](phi/math/backend) <br /> [phi.math.extrapolation](phi/math/extrapolation.html) <br /> [phi.math.magic](phi/math/magic.html)                                                                 | [Overview](Math.html): Named dimensions, backends, indexing, non-uniform tensors, precision <br /> [Optimization and Training](Optimization.md): Automatic differentiation, neural network training <br /> [Performance](GPU_Execution.md): GPU, JIT compilation, profiler             |
| [phi.torch.nets](phi/torch/nets) <br> [phi.tf.nets](phi/tf/nets) <br> [phi.jax.stax.nets](phi/jax/stax/nets)                                                                                                                               | [Built-in Neural Networks](Network_API): Architectures, convenience functions                                                                                                                                                                                                          |

### Core Classes

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;lightbox&quot;:false,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;app.diagrams.net\&quot; modified=\&quot;2021-07-02T20:16:56.988Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36\&quot; etag=\&quot;_2Yp1tYnDWZnCvK0xIgF\&quot; version=\&quot;14.8.0\&quot; type=\&quot;onedrive\&quot;&gt;&lt;diagram name=\&quot;Page-1\&quot; id=\&quot;c4acf3e9-155e-7222-9cf6-157b1a14988f\&quot;&gt;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&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

### API Documentation

The [API documentation](phi) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force phi
```
This requires PyTorch, TensorFlow and Jax to be installed, in addition to the standard Φ<sub>Flow</sub> requirements.


### Contributing to Φ<sub>Flow</sub>

Contributions are welcome!

If you have changes to merge, check out our [style guide](https://github.com/tum-pbs/PhiFlow/blob/develop/CONTRIBUTING.md) before opening a pull request.
