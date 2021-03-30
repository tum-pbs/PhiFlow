from numpy import log10


class EditableValue(object):
    """
    Editable values are used to specify controls of an `App` that can be manipulated at runtime by the user.
    EditableValues only specify the initial value of the control.
    During `App.prepare()`, they are replaced by `Control` objects that hold the actual value.
    """

    def __init__(self, name: str, type: str, initial_value, category: str, minmax, is_linear):
        """
        This constructor should not be used directly. Instantiate a subclass instead.

        Args:
            name: Human-readable name to be displayed.
            type: Type identifier
            initial_value:
            category: Used to sort values. Not currently used.
            minmax: tuple (min, max). Determines the range of allowed values. Determines the maximum expected text length for text fields.
            is_linear: Whether to use a linear slider where applicable.
        """
        self.name = name
        """ Human-readable name to be displayed. """
        self.type = type
        """ Type identifier: 'int', 'float', 'bool', 'text'. """
        self.initial_value = initial_value
        """ Set manually or value of the variable when `App.prepare()` was called. """
        self.category = category
        """ Used to sort values. Not currently used. """
        self.minmax = minmax
        self.is_linear = is_linear
        """ Whether to use a linear slider where applicable. """

    @property
    def min_value(self):
        """ Determines the range of allowed values. """
        return self.minmax[0]

    @property
    def max_value(self):
        """ Determines the range of allowed values. Determines the maximum expected text length for text fields. """
        return self.minmax[1]

    @staticmethod
    def value(value_or_editable_value):
        """
        Retrieves the initial value if the argument is an `EditableValue`, else returns the given value.
        This is useful for consistently accessing an `App` variable before and after `prepare()` is called.
        """
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
