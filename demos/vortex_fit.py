from functools import partial
from phi.tf.flow import *

VORTEX_COUNT = 80
DESCRIPTION = """
Each vortex is parameterized by its location, strength and radius.
Vortexes produce divergence-free velocities swirling around the center with the fluid speed falling off with distance.

You can adjust the number of vortexes in the demo script by tweaking VORTEX_COUNT (currently %d).
""" % VORTEX_COUNT


def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return math.exp(- sq_distance / sigma ** 2) / math.sqrt(sq_distance)


# --- Prepare reference state ---
fluid = world.add(Fluid(Domain([80, 64], boundaries=CLOSED), velocity=Noise()), physics=IncompressibleFlow())
for _ in range(10): world.step()

# --- Set up optimization ---
opt_velocity = variable(AngularVelocity(location=math.random_uniform((1, VORTEX_COUNT, 2)) * fluid.domain.resolution,
                                        strength=(math.random_uniform((VORTEX_COUNT,)) - 0.5) * 0.1,
                                        falloff=partial(gaussian_falloff, sigma=variable(math.random_uniform((1, VORTEX_COUNT, 1)) + 5))))
sampled_velocity = opt_velocity.at(fluid.velocity)
loss = math.l2_loss(sampled_velocity - fluid.velocity)
reg = math.l1_loss(opt_velocity.strength)

app = LearningApp('Fit Fluid with Vortexes', DESCRIPTION, learning_rate=0.1)
app.add_objective(loss, reg=reg)
app.add_field('Fit Velocity', sampled_velocity)
app.add_field('True Velocity', fluid.velocity)
app.add_field('Vortex Strength', SampledField(opt_velocity.location, math.expand_dims(math.expand_dims(opt_velocity.strength, -1), 0)).at(fluid.density))
show(app, display=('True Velocity', 'Fit Velocity'))
