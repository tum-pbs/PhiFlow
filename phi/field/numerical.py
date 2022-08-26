"""
This module contains the `Scheme` class for specifying properties of the numerical scheme of a simulation or individual operations.
"""
from ..math import Solve


# volume_sampling='scatter', 'exact'
# NaN to zero: (velocity - prev_velocity) @ particles  for outside particles


class Scheme:
    """
    Numerical scheme, specifying details about the numerical method being used.

    Numerical schemes are used, among others, for

    * Field resampling, such as `phi.field.resample()`, `phi.field.Field.at()`
    * Finite difference operations, such as `phi.field.spatial_gradient()`, `phi.field.laplace()`.

    Schemes generally do not affect the sample point locations, extrapolations or other properties.
    Consequently, simulation code should run with various schemes without additional modification.
    """

    def __init__(self, order: int = None, solve: Solve = None, outside_points: str = 'discard'):
        """
        Args:
            order: Minimum spatial order of the scheme. If not supported, functions may choose the next higher order.
            solve: Specifies the accuracy for implicit schemes, `None` for explicit schemes.
            outside_points: How to handle points lying outside the valid bounds.
                Either `'discard'` to ignore them or `'clamp'` to treat them as if they lied on the domain boundary.
        """
        self.order = order
        """ Minimum spatial order of the scheme. If not supported, functions may choose the next higher order. """
        self.solve = solve
        """ Specifies the accuracy for implicit schemes, `None` for explicit schemes. """
        self.outside_points = outside_points
        """
        How to handle points lying outside the valid bounds.
        Either `'discard'` to ignore them or `'clamp'` to treat them as if they lied on the domain boundary.
        """

    @property
    def is_implicit(self):
        """ Implicit schemes define a valid `solve` object specifying the accuracy. """
        return self.solve is not None
