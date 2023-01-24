"""
Contains built-in physics functions, mainly for partial differential equations, such as incompressible fluids.
The actual physics functions are located in the submodules of `phi.physics`.

Some physics functions have built-in time advancement while others return the PDE term, i.e. the derivative.
The time-advancing functions always take a time increment argument called `dt`.

See the `phi.physics` module documentation at https://tum-pbs.github.io/PhiFlow/Physics.html
"""
