import numpy as np


class AxisOrder:

    def __init__(self):
        self.is_x_first = False
        self.x = 0
        self.y = 0
        self.z = 0
        self.x_last()

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


GLOBAL_AXIS_ORDER = AxisOrder()
