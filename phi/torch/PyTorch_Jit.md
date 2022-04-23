# Making PyTorch compatible with Φ<sub>Flow</sub>'s `jit`


## PyTorch Limitations

* PyTorch does not support nested tracing
* `autograd.Function` implementations are not traced but will be called in Python.
* Tracing  `autograd.Function`s results in [double execution](https://github.com/pytorch/pytorch/issues/75655)
* Modules cannot be called within traced functions
* Gradient functions may return `None` to indicate no dependency but `None` is not supported in traces.


## Solutions

Overview - PyTorch Tracing Procedure:

1. Call function non-tracing, record all called nn.Modules and autograd.Function forward calls with their args
2. Compile autograd.Functions forward and backward passes
3. Add nn.Modules to JIT Module
4. Trace JIT Module

Nested jit calls are ignored.


### Internal Modules for `jit`
All called modules can be added as submodules and can therefore be traced.
This requires a call to `._torch_backend.register_module_call()` inside  `nn.Module.forward()`

Jit-functions are represented by `phi.torch._torch_backend.JITFunction` which instantiates a `JITFunction.__call__.JitModule`


### Separate traces for `autograd.Function`
All custom functions are traced separately.
In the first non-tracing run, all functions are recorded and added to `JITFunction.autograd_function_calls` with their arguments.
Then, their forward passes are compiled individually.
The backward passes are compiled on demand.

Custom functions are reprsented by `construct_torch_custom_function.TorchCustomFunction`.
This class must be declared inside the function to have access to the given forward and backward function (`autograd.Function` methods are static).

When compiling, a new `TorchCustomFunction` is constructed for the jit version (`TorchCustomFunction.compile`, called by `JITFunction`).

### None filter
`TorchCustomFunction.backward` filters `None` gradients from the result during tracing and reconstructs them afterward.


### Prevent double execution
To prevent double execution, custom functions must not perform any operations during tracing.
PyTorch does not drop these from the graph even though their output is not used.

Instead, custom functions check the tracing state and return the cached outputs when tracing.
This allows later operations to work correctly during tracing without adding extra computations to the graph.

