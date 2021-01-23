"""
Open-source simulation toolkit built for optimization and machine learning applications.

Use one of the following imports:

* `from phi.flow import *`  for NumPy mode
* `from phi.tf.flow import *`  for TensorFlow mode
* `from phi.torch.flow import *` for PyTorch mode

Project homepage: https://github.com/tum-pbs/PhiFlow

PyPI: https://pypi.org/project/phiflow/
"""

import os


with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()
