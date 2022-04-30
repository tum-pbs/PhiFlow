import sys
import traceback

import numpy as np


class AxisOrder:
    """Define the order of spatial axes. Default: x first"""

    def __init__(self):
        self.is_x_first = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.x_first()

    def x_first(self):
        self.is_x_first = True
        self.x, self.y, self.z = 0, 1, 2

    def x_last(self):
        self.is_x_first = False
        self.x, self.y, self.z = -1, -2, -3

    def up_vector(self, rank):
        if self.is_x_first:
            return np.array([0] * (rank - 1) + [1])
        else:
            return np.array([1] + [0] * (rank - 1))

    def axis_name(self, index, spatial_rank):
        if self.is_x_first:
            return ['x', 'y', 'z', 'w'][:spatial_rank][index]
        else:
            return ['x', 'y', 'z', 'w'][:spatial_rank][::-1][index]


GLOBAL_AXIS_ORDER = AxisOrder()


# AUTO_REVERSE_ORDER = False

def should_use_color():
    for line in traceback.format_stack():
        if "_pydevd_bundle\pydevd_xml.py" in line:
            return False
    return True

# COLOR_PRINT = 'ipykernel' in sys.modules
