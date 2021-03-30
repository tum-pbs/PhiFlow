import numpy as np

from ._value import EditableValue
from typing import Callable


class Control(object):

    def __init__(self, model, attribute_name: str, editable_value: int or float):
        assert isinstance(editable_value, EditableValue)
        self.model = model
        self.attribute_name = attribute_name
        self.editable_value = editable_value

    @property
    def value(self):
        val = getattr(self.model, self.attribute_name)
        if isinstance(val, np.float32):
            return float(val)
        if isinstance(val, np.float64):
            return float(val)
        return val

    @value.setter
    def value(self, value):
        setattr(self.model, self.attribute_name, value)
        self.model.invalidate()

    @property
    def name(self):
        return self.editable_value.name

    @property
    def type(self):
        return self.editable_value.type

    @property
    def id(self):
        return self.attribute_name

    def __str__(self):
        return self.name + '_' + str(self.value)

    @property
    def range(self):
        return self.editable_value.minmax


class Action(object):

    def __init__(self, name: str, method: Callable, id: str):
        self.name = name
        self.method = method
        self.method_name = id

    @property
    def id(self):
        return self.method_name
