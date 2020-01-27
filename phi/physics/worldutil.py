""" High-level world utility and convenience functions. """

from .world import StateProxy, World
from .field.mask import union_mask


def obstacle_mask(world_or_proxy):
    """
Builds a binary Field, masking all obstacles in the world.
    :param world_or_proxy: World or StateProxy object
    :return: Field
    """
    world = world_or_proxy.world if isinstance(world_or_proxy, StateProxy) else world_or_proxy
    assert isinstance(world, World)
    geometries = [obstacle.geometry for obstacle in world.state.all_with_tag('obstacle')]
    return union_mask(geometries)
