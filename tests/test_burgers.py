from unittest import TestCase

import numpy

from phi import math
from phi.physics import Physics
from phi.physics.domain import Domain
from phi.physics.field.effect import FieldEffect
from phi.physics.schroedinger import SinPotential
from phi.physics.world import World


class ForcingPhysics(Physics):

    def __init__(self, omega):
        Physics.__init__(self)
        self.omega = omega

    def step(self, fieldeffect, dt=1.0, **dependent_states):
        # pylint: disable-msg = arguments-differ
        field = fieldeffect.field
        field = field.copied_with(phase_offset=field.phase_offset + dt * self.omega)
        return fieldeffect.copied_with(field=field)


class TestBurgers(TestCase):

    def test_batched_forced_burgers_1d(self):
        world = World(batch_size=3)
        burgers = world.add(Domain([4]).centered_grid(0, batch_size=world.batch_size, name='velocity'))
        k = math.to_float(numpy.random.uniform(3, 6, [world.batch_size, 1]))
        amplitude = numpy.random.uniform(-0.5, 0.5, [world.batch_size, 1])
        force = SinPotential(k, phase_offset=numpy.random.uniform(0, 2 * numpy.pi, [world.batch_size]), data=amplitude)
        physics = ForcingPhysics(numpy.random.uniform(-0.4, 0.4, [world.batch_size]))
        effect = FieldEffect(force, ['velocity'])
        world.add(effect, physics=physics)
        burgers.step()
        burgers.step()

    def test_batched_forced_burgers_2d(self):
        world = World(batch_size=3)
        burgers = world.add(Domain([4, 4]).centered_grid(0, batch_size=world.batch_size, name='velocity'))
        k = math.to_float(numpy.random.uniform(3, 6, [world.batch_size, 2]))
        amplitude = numpy.random.uniform(-0.5, 0.5, [world.batch_size])
        force = SinPotential(k, phase_offset=numpy.random.uniform(0, 2 * numpy.pi, [world.batch_size]), data=amplitude)
        physics = ForcingPhysics(numpy.random.uniform(-0.4, 0.4, [world.batch_size]))
        effect = FieldEffect(force, ['velocity'])
        world.add(effect, physics=physics)
        burgers.step()
        burgers.step()
