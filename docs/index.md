
[**Homepage**](https://github.com/tum-pbs/PhiFlow)
&nbsp;&nbsp;&nbsp; [**API**](phi)
&nbsp;&nbsp;&nbsp; [**Demos**](https://github.com/tum-pbs/PhiFlow/tree/develop/demos)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Fluids Tutorial**](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true)
&nbsp;&nbsp;&nbsp; [<img src="https://www.tensorflow.org/images/colab_logo_32px.png" height=16> **Playground**](https://colab.research.google.com/drive/1zBlQbmNguRt-Vt332YvdTqlV4DBcus2S#offline=true&sandboxMode=true)

### Guides

* [Colab notebook](https://colab.research.google.com/drive/1LNPpHoZSTNN1L1Jt9MjLZ0r3Ejg0u7hY#offline=true&sandboxMode=true)
on fluid simulations. This is a great place to get started with Φ<sub>Flow</sub>.
* [Optimization and Training](Optimization.md): Automatic differentiation, neural network training
* [Reading and Writing Simulation Data](Reading_and_Writing_Data.md)
* [Performance](GPU_Execution.md): GPU, JIT compilation, profiler 

### Module Overview

| Module API  | Documentation                                        |
|-------------|------------------------------------------------------|
| [phi.vis](phi/vis) | [Visualization](Visualization.md): Plotting, interactive user interfaces <br /> [Dash](Web_Interface.md): Web interface <br /> [Widgets](Widgets.md): Notebook interface  <br /> [Console](ConsoleUI.md): Command line interface   |
| [phi.physics](phi/physics) <br /> [phi.physics.advect](phi/physics/advect.html) <br /> [phi.physics.fluid](phi/physics/fluid.html) <br /> [phi.physics.diffuse](phi/physics/diffuse.html) <br /> [phi.physics.flip](phi/physics/flip.html) | [Overview](Physics.md): Domains, built-in physics functions <br /> [Writing Fluid Simulations](Fluid_Simulation.md): Advection, projection, diffusion        |
| [phi.field](phi/field)   | [Overview](Fields.md): Grids, particles <br /> [Staggered Grids](Staggered_Grids.md): Data layout, usage  |
| [phi.geom](phi/geom)    | [Overview](Geometry.md): Differentiable Geometry        |
| [phi.math](phi/math) <br /> [phi.math.backend](phi/math/backend) <br /> [phi.math.extrapolation](phi/math/extrapolation.html)  | [Overview](Math.md): Named dimensions, backends, indexing, non-uniform tensors, precision <br /> |

### Core Classes

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;app.diagrams.net\&quot; modified=\&quot;2021-06-20T11:36:59.331Z\&quot; agent=\&quot;5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36\&quot; etag=\&quot;RvXB7JJeA-QN1fo_TdIm\&quot; version=\&quot;14.8.0\&quot; type=\&quot;onedrive\&quot;&gt;&lt;diagram name=\&quot;Page-1\&quot; id=\&quot;c4acf3e9-155e-7222-9cf6-157b1a14988f\&quot;&gt;7VzZbuM6Ev0aA+kLRNBieXmMk7jnoS8m00lP33kKaIu2OK3FkKiOfb9+ihQpUdRi2bGTDGIjSESyuIh1TrFIljNwbsPt1wRt/D9jDwcD2/S2A+duYNvW0LYH7Mf0dnnOeCwy1gnxhFCZ8Uj+xiLTFLkZ8XBaEaRxHFCyqWYu4yjCS1rJQ0kSv1TFVnFQ7XWD1riW8bhEQT33J/Gon+dOXLPM/wcma1/2bJmiZIGWv9ZJnEWivyiOcF4SItmMEE195MUvSpZzP3Bukzim+VO4vcUBm1Y5Y3m9eUtpMeQER7Shwo8UJ/9c/JfNlm0GaAEa40KD2/vBzcjgw+MtBCT6lRf5lLL5vmFt2HOahdebRWqsCfWzhUFiyHvwyTyAl7DnG5/A77yVednQE0rWWAzneREgaFrVZl7wAJPGFAKj94kYyWhgO4AGZ6Y+3tuDiTm4sRVJA28pYDAOECVxtE+YqQeDbvgY5Mzdp89+GCbjHw8r8nRNveQh2V0PR4VKiqlO6U4CBJS3YY9ZGMwTFMLj7MUnFD9u0JLlv8CYIM+nIZtmCx45KDDrko2uUD5LLOOQLMUz18ysQNFtHMQJFHEcQTWaxL8KSLJmV3FE5ygkAWPav3HioQiJbEErPhsoIOsIEksAB07YaEUbY1bqSyzbLLUiQSD7hdn0EJ6slkXnSsloOcGLFX+BIECblCz49OSvFFFEItYVHybKaJyK8bDJwMssSclv/B2L3FzVOoIFqH/jhOKtkiX08hXHIabJDkREqSMpKgyPLYn5UtJ4KK2Mr1DYGYlMJEzHumi75BBDeA4GmVRY1cmyJxylMG2nI5hTYFpvuhfllgFKU5UZaiudxJjY3cR4ISH0iSUGRYnVhEGmVgJG90YU0JhxZumTwPuGdnHGXiKlwAWZmvlxQv5m0JKsguJEAh24r0o8spoCWQmH2YNElqVl/Ym2FcFvKKUFkFVos4ohzC+JZjGlcdjE5vQXpktfJNgcKIwx4TOfNxK5Dv4Ou3QIJaDXCiWsOiOscQMj7C5GiN6+A9BRtA7wQd0191btDAWAkQhRPGNmMK2xsHjPnsSsrZfN4HbE7KIgE4Ae2DPmyNAds/fAVPPuiT2qC9QsX82lwCN/1AU8vEJZQJ/lEsRFZ3I90oQTxlouQQANeili78kKoSRgtIB5AlvKOC5YPFpTbenMK0a74yqGGEXH1Uyz8MguyZE9hozMvSraVdMFfKK1ZU4svOqaKLKkOQvwirYasxT8ARKtv3GZu2GZ811An2XFUHcVcAPiE8/DEXcXKKKoXFA3MSCBc8OdwQ+83K1puAMXBn4LaatMww8TT8DuRPAuiHB7gsGkveCU9rc0E7u3pdlVObzPsljD4cFrbU/+Dhv4q6kZ1sgWV+pgHYegrQCXSn1iOr+7tmqKd+qKdxqUzB2GB3CUuC/r3CW5rKb8d9Nvocx9Cp6cS71ui3nOKX715TiLEWXhZicrRxsj8mAPiXZ1E05CaAhUcwVPQjz3m+5IWJNeBWTDBNOKZE0si7iLozRJsw0gwp2JCu7dxVKdHsmTnki2D4Zy2xZAuAVn2AFoLR+3AVAa6ebgno3xxf//YP6/pfnj8ihOhf6oaZUed5jxdv+/R3cf2P8ftyww3Q757zjIQtxeTtLnLCKrOBG+8AKYWF+FUIjT6goAKOF2U/epAaKaJO/VZMuLqSwbWj22h2np4bLAdC4w/UnX4Qo3kWxyNldpcvGEz6deyxr21O+51DttMVQk8vBWcSabzBF4B8/MhuhiFfvRVIkZEKUSMx4Xy3FqaNn1BbMRWu7oYGy13wHZgxu2hsAUsvsLSM5mche1UDdOeY7MuNfuXoJcar6onrGcwN2t3PIY/Frllf5utUXtTfZ6wVNTUWLdC1bufT60b+i4FWfNbdjfN12WuB0nw/3vSmqWbm0u11M/yX6Sv378Xj/9HP2LPFw3eWSazcGRd8MufdnegqmZXabNPJT6/LqNTaCij36zDnM2g8lOdn+xaoZpjmTGf5j2DHPoyow7uXfIUzuZ8tZY7n7AjvjxOo5QcF/mgrw3J4EcF6/9gBMCM8mvzXgjW0LZCK5NOQTIyEdgDaciXQ6AJXZKQm8unzk2MgUcBwArjbNkibvMnTg/pwolOxztOi4V5LkdwEtwkJ/tqK/RsU95iPMFTt5RaLeEgs5lE/mLilolpusNyT2SaMhxbMOcWlP5kfsJ2W4+L7V2b/LjpkJMrFPt49c2We6ke5SauG3aGlPzAfTcXfXjrbRF78DbY5gnWWZNNaKPh/uI3kVcq8JZYTVOzVmrJzultd/PTvs92VlcxUt2Do9k53Ci4d7RGjoRHR2z2s/Y7hyWLi6HdSQd9wT2rAgOvNf7YaKZU4T2qM6VNUbLBXLHwwl2neloca36K5ewm+6wm4o36SXxplCJVTu5bArKwdsNiqSpPVtIjn4gWPcx7UmDMSlYcp6AnFuuEex9ZQGJ7MoIkG0+UrRel7mn4ww/q+dJo9p0LxKpoWzqPiZvUH2TK/brS0sUXL1q5X1l3b17H2fYzdHPeAPQ5p008KfR4r2CUE4DoZrOC46M5+nRm9nYW7Wz9znPd9oujHlG2nkxu8gHmsfpxNtaeTXwlIvp5yEHhggB9OIgK5trlko30DwKnruvJLxt58vhAIeYn4UxoRJjmpg8MFNbuhz4dbnOzrA3nduvChr5aznnippxeuzVLncFx+q3OLzdp+Bz3RU49SO0P/iCL4wgUEG3ZjxLGED+XJqmL5/WAvRf40+GnHqUSvPSbp8LOk23iNVQKHZtLe6D5sypbAqFegEn+TlHmwCdqPFH7g03iVcAeVVJfWEO+776OXqv8j/V7j4pfs+H02lPCzc+fEu353SDtfD6jVreyuvONsqRdF8cOd2bp8sBx//dAYf+naOmL1g0nXBYR9DhgBMOvmc5GTMciXGj0u7Bd6+yiasZSmEvuO1z4GCZdjdpPuOJwxveK0+dQymx/9CgOeTQ6XBkXnNE0XweUu3rfU4oLLPtK0f99vlq6GHLXl9iv0OkfjRx8ZG6GCEs0jkiAt1zbQMt8/LlmHMqePLOIYGWWT/sZAstkAfYz3fx2WaDE/V6gK3l7mzLdlPb51xmxxI7kTAM4zOHB79yx38EiKZ2PxDpV+LHoKgxbMOuYaim/XOFbciQCcM0J4NKqNNoOOgMm1ADOPRILavMOFWkVkOURmMQSD5zbxu6UfWV3zx2w9ViGxxbA2rf2A1X/0cO+mapJXbjVOFLPdbKN+eBPdoT8id5YFYpcPJQxQ8SplQJwn37IMKpbZjjafmp4HU6NqZjbYfRF/vOUD9QcN8U+02Xpm+P/WEF+4476Yf9pjUA1PSZ1oDKVzTfnBeWOzWm6seqrggTFgc5MouPhu3eobeObUhvSYbBOs6b8qTHV6zOzxPLqfBksi/E9FRAbwuMH433Ue3MNClOVD44T2oezohZrjJ2fHokMVz7OM/p0KhXV1+lLLd7XJq8IzbMR4a9QrL8X3m5ePm/CJ37/wE=&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

### API Documentation

The [API documentation](phi) is generated using [pdoc3](https://pdoc3.github.io/pdoc/).
To manually generate the documentation, add the PhiFlow directory to your Python path and run
```bash
$ pdoc --html --output-dir docs --force phi
```
To simply view the documentation, you can host a local web server using
```bash
$ pdoc --http phi
```

### Other Documentation

* [Installation Instructions](Installation_Instructions.md):
  Requirements, installation, CUDA compilation
* [Contributing to Φ<sub>Flow</sub>](https://github.com/tum-pbs/PhiFlow/blob/develop/CONTRIBUTING.md):
  Style guide, docstrings, commit tags
* [Scene Format Specification](Scene_Format_Specification.md):
  Directory layout, file format
