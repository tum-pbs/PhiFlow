"""
This script is a history of various ways we considered for declaring dimension names and types
"""

from phi.flow import *
from phi.math import zeros
from phi.math._shape import spatial, channel

array = np.zeros([5, 4, 2])


# Implicit names
zeros(x=5, y=4, vector=2)
tensor(array, 'x, y, vector')

# Explicit names
zeros(x_spatial=5, y_spatial=4, vector_channel=2)
tensor(array, 'x: spatial, y: spatial, vector: channel')

# Mixed
zeros(x=5, y=4, vector_channel=2)
tensor(array, 'x, y, vector: channel')

# Explicit types, automatic order
zeros(spatial=dict(x=5, y=4), channel=dict(vector=2))
zeros(spatial=[('x', 5), ('y', 4)], channel=('vector', 2))
zeros(spatial(x=5), channel(vector=2))
tensor(array, 'x, y, vector', [spatial, spatial, channel])
tensor(array, [('x', spatial), ('y', spatial), ('vector', channel)])

# Explicit types, manual order
zeros(x=(5, spatial), y=(4, spatial), vector=(2, channel))
tensor(array, x=spatial, y=spatial, vector=channel)

# Python types
zeros(spatial('x', 5), spatial('y', 4), channel('vector', 2))
zeros(x=spatial(5), y=spatial(4), vector=channel(2))
tensor(array, x=spatial, y=spatial, vector=channel)

# Icons
zeros(batch_ᵇ=10, particles_ᵛ=34, x_ˢ=32, y_ˢ=32, vector_ᵛ=2)
ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ = 5

# New proposal: special names x, y, z, vector, points, particles, batch, mini_batch
zeros(inflow_loc=(10, batch), x=32, y=16, vector=(2, channel))
tensor(array, 'inflow_loc: batch, x, y, vector')
tensor(array, inflow_loc=batch, x=spatial, y=spatial, vector=channel)
tensor(array, [('inflow_loc', batch), 'x, y', 'vector'])

# Final decision
s = spatial('x, y')
z = math.zeros(spatial(x=5, y=4) & channel(vector=2))
t = tensor(array, spatial('x, y'), channel(vector=2))