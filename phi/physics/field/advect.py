from .field import StaggeredSamplePoints


def semi_lagrangian(field, velocity_field, dt):
    """
    Semi-Lagrangian advection with simple backward lookup.
        :param field: Field to be advected
        :param velocity_field: Field, need not be compatible with field.
        :param dt: time step
        :return: Field compatible with input field
    """
    try:
        x0 = field.points
        v = velocity_field.at(x0)
        x = x0 - v * dt
        data = field.sample_at(x.data)
        return field.with_data(data)
    except StaggeredSamplePoints:
        advected = [semi_lagrangian(component, velocity_field, dt) for component in field.unstack()]
        return field.with_data(advected)
