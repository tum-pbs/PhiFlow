from phi import struct
from phi.data.fluidformat import _transform_for_writing
from phi.physics.physics import State
from phi.physics.world import StateProxy
from phi.struct.context import _unsafe

from .util import placeholder


def load_state(state):
    if isinstance(state, StateProxy):
        state = state.state
    assert isinstance(state, State)
    state = _transform_for_writing(state)
    names = struct.names(state)
    with _unsafe():
        placeholders = placeholder(state.shape)
    state_in = struct.map(lambda x: x, placeholders)  # validates fields, splits staggered tensors
    return state_in, {placeholders: names}
