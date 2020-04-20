""" High-level world utility and convenience functions. """
from phi.geom import union
from .world import StateProxy, World
from .field.mask import mask


def obstacle_mask(world_or_proxy, **mask_kwargs):
    """
Builds a binary Field, masking all obstacles in the world.
    :param world_or_proxy: World or StateProxy object
    :return: Field
    """
    world = world_or_proxy.world if isinstance(world_or_proxy, StateProxy) else world_or_proxy
    assert isinstance(world, World)
    geometries = [obstacle.geometry for obstacle in world.state.all_with_tag('obstacle')]
    return mask(union(geometries), **mask_kwargs)
