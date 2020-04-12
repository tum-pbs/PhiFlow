import numpy as np


X = 0
Y = 0
Z = 0
X_FIRST = False


def x_first():
    global X, Y, Z, X_FIRST
    X_FIRST = True
    X, Y, Z = 0, 1, 2


def x_last():
    global X, Y, Z, X_FIRST
    X_FIRST = False
    X, Y, Z = -1, -2, -3


x_last()


def up_vector(rank):
    if X_FIRST:
        return np.array([0] * (rank - 1) + [1])
    else:
        return np.array([1] + [0] * (rank - 1))



