"""
Wave equation demo using PhiFlow.

Simulates a 2D wave equation ∂²u/∂t² = c² ∇²u on a square domain
starting from a Gaussian pulse, using the leapfrog scheme from
phi.physics.wave.
"""
from phi.flow import *
from phi.physics import wave

# --- Parameters ---
N = 128
DT = 0.002
C = 1.0
STEPS = 500

# --- Initial condition: Gaussian pulse centered in domain, at rest ---
u = CenteredGrid(
    lambda x: math.exp(-100 * math.vec_squared(x - 0.5)),
    extrapolation.ZERO_GRADIENT,
    x=N, y=N,
    bounds=Box(x=1, y=1)
)
u_prev = u  # zero initial velocity (du/dt = 0)

# --- Time integration with leapfrog ---
for step_i in range(STEPS):
    u, u_prev = wave.step(u, u_prev, c=C, dt=DT)

# --- Report ---
print(f"Wave equation simulation completed: {STEPS} steps, grid {N}x{N}")
print(f"  dt={DT}, c={C}, CFL = {C * DT / (1.0 / N):.3f}")
print(f"  Final field: min={float(math.min(u.values)):.6f}, max={float(math.max(u.values)):.6f}")
