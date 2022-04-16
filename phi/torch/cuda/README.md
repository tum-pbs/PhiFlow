# Compilation:

`~/PhiFlow/setup.py install torch_cuda`

It will create a library file like so:

`~/PhiFlow/venv/lib/python3.8/site-packages/torch_cuda-0.0.0-py3.8-linux-x86_64.egg/torch_cuda.cpython-38-x86_64-linux-gnu.so'
`

Then this file has to be imported by `torch_backend.py` like so:

`torch.ops.load_library('../venv/lib/python3.8/site-packages/torch_cuda-0.0.0-py3.8-linux-x86_64.egg/torch_cuda.cpython-38-x86_64-linux-gnu.so')
`