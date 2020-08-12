from numpy import log10


class EditableValue(object):

    def __init__(self, name, type, initial_value, category, minmax, is_linear):
        self.name = name
        self.type = type
        self.initial_value = initial_value
        self.category = category
        self.minmax = minmax
        self.is_linear = is_linear

    @property
    def min_value(self):
        return self.minmax[0]

    @property
    def max_value(self):
        return self.minmax[1]

    @staticmethod
    def value(value_or_editable_value):
        if isinstance(value_or_editable_value, EditableValue):
            return value_or_editable_value.initial_value
        else:
            return value_or_editable_value


class EditableFloat(EditableValue):

    def __init__(self, name, initial_value, minmax=None, category=None, log_scale=None):
        if minmax is not None:
            assert len(minmax) == 2, 'minmax must be pair (min, max)'

        if log_scale is None:
            if minmax is None:
                log_scale = True
            else:
                log_scale = minmax[1] / float(minmax[0]) > 10

        if not minmax:
            if log_scale:
                magn = log10(initial_value)
                minmax = (10.0**(magn - 3.2), 10.0**(magn + 2.2))
            else:
                if initial_value == 0.0:
                    minmax = (-10.0, 10.0)
                elif initial_value > 0:
                    minmax = (0., 4. * initial_value)
                else:
                    minmax = (2. * initial_value, -2. * initial_value)
        else:
            minmax = (float(minmax[0]), float(minmax[1]))
        EditableValue.__init__(self, name, 'float', initial_value, category, minmax, not log_scale)

    @property
    def use_log_scale(self):
        return not self.is_linear


class EditableInt(EditableValue):

    def __init__(self, name, initial_value, minmax=None, category=None):
        if not minmax:
            if initial_value == 0:
                minmax = (-10, 10)
            elif initial_value > 0:
                minmax = (0, 4 * initial_value)
            else:
                minmax = (2 * initial_value, -2 * initial_value)
        EditableValue.__init__(self, name, 'int', initial_value, category, minmax, True)


class EditableBool(EditableValue):

    def __init__(self, name, initial_value, category=None):
        EditableValue.__init__(self, name, 'bool', initial_value, category, (False, True), True)


class EditableString(EditableValue):

    def __init__(self, name, initial_value, category=None, rows=20):
        EditableValue.__init__(self, name, 'text', initial_value, category, ('', 'A' * rows), True)

    @property
    def rows(self):
        return len(self.max_value)
