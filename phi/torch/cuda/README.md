# Compilation:

`~/PhiFlow/setup.py install phi_torch_cuda`

It will create a library file like so:

`~/PhiFlow/build/phi_torch_cuda.so'`

Then this file has to be imported by `_torch_backend.py` like so:

`torch.ops.load_library('../build/phi_torch_cuda.so')`