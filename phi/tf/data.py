from phi import struct
from phi.data.fluidformat import _transform_for_writing
from phi.physics.physics import State
from phi.physics.world import StateProxy
from phi.struct.context import _unsafe

from .util import placeholder


def load_state(state):
    """
Create placeholders for tensors in the supplied state.
    :param state: State or StateProxy
    :return:
      1. Valid state containing or derived from created placeholders
      2. dict mapping from placeholders to their default names (using struct.names)
    """
    if isinstance(state, StateProxy):
        state = state.state
    assert isinstance(state, State)
    state = _transform_for_writing(state)
    names = struct.names(state)
    with _unsafe():
        placeholders = placeholder(state.staticshape)
    state_in = struct.map(lambda x: x, placeholders)  # validates fields, splits staggered tensors
    return state_in, {placeholders: names}
