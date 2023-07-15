"""
Stax implementation of the unified machine learning API.
Equivalent functions also exist for the other frameworks.

For API documentation, see https://tum-pbs.github.io/PhiFlow/Network_API .
"""
from phiml.backend.jax.stax_nets import *
from phiml.backend.jax.stax_nets import mlp as dense_net
from phiml.nn import parameter_count
