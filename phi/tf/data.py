from phi.data.fluidformat import _transform_for_writing
from .util import *
from phi.physics.physics import State
from phi.physics.world import StateProxy


def load_state(state):
    if isinstance(state, StateProxy):
        state = state.state
    state = _transform_for_writing(state)
    names = struct.names(state)
    with struct.anytype():
        placeholders = placeholder(state.shape)
    state_in = struct.map(lambda x: x, placeholders)  # validates fields, splits staggered tensors
    return state_in, {placeholders: names}