from unittest import TestCase

from phi.flow import *
from phi.math.backend import Backend
from phi.physics._boundaries import STICKY, Domain

BACKENDS = phi.detect_backends()


class ColabNotebookTest(TestCase):

    def test_graph_gradients(self):
        for backend in BACKENDS:
            if backend.supports(Backend.record_gradients):
                with backend:
                    DOMAIN = Domain(x=32, y=40, boundaries=STICKY, bounds=Box[0:32, 0:40])
                    INFLOW_LOCATION = math.tensor([(4., 5), (8., 5), (12., 5), (16., 5)], batch('inflow_loc'), channel('vector'))
                    INFLOW = DOMAIN.grid(Sphere(center=INFLOW_LOCATION, radius=3)) * 0.6

                    smoke = DOMAIN.scalar_grid(math.zeros(batch(inflow_loc=4)))
                    velocity = initial_velocity = DOMAIN.staggered_grid(0) * math.ones(batch(inflow_loc=4))

                    with math.record_gradients(velocity.values):
                        for _ in range(3):
                            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
                            buoyancy_force = smoke * (0, 0.5) @ velocity
                            velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
                            velocity, _ = fluid.make_incompressible(velocity)
                        loss = math.mean(field.l2_loss(smoke - field.stop_gradient(smoke.inflow_loc[-1])))
                        grad = math.gradients(loss, initial_velocity.values)

    def test_functional_gradient(self):
        for backend in BACKENDS:
            if backend.supports(Backend.functional_gradient):
                with backend:
                    DOMAIN = Domain(x=32, y=40, boundaries=STICKY, bounds=Box[0:32, 0:40])
                    INFLOW_LOCATION = math.tensor([(4., 5), (8., 5), (12., 5), (16., 5)], batch('inflow_loc'), channel('vector'))
                    INFLOW = DOMAIN.scalar_grid(Sphere(center=INFLOW_LOCATION, radius=3)) * 0.6

                    def simulate(velocity: StaggeredGrid, smoke: CenteredGrid):
                        for _ in range(3):
                            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
                            buoyancy_force = smoke * (0, 0.5) @ velocity
                            velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
                            velocity, _ = fluid.make_incompressible(velocity)
                        loss = field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10))
                        return loss, smoke, velocity

                    initial_smoke = DOMAIN.scalar_grid(math.zeros(batch(inflow_loc=4)))
                    initial_velocity = DOMAIN.staggered_grid(0) * math.ones(batch(inflow_loc=4))

                    sim_grad = field.functional_gradient(simulate, wrt=[0], get_output=False)

                    for _ in range(2):
                        velocity_grad, = sim_grad(initial_velocity, initial_smoke)
                        initial_velocity = initial_velocity - 0.01 * velocity_grad

    def test_functional_gradient_jit(self):
        for backend in BACKENDS:
            if backend.supports(Backend.functional_gradient):
                with backend:
                    DOMAIN = Domain(x=32, y=40, boundaries=STICKY, bounds=Box[0:32, 0:40])
                    INFLOW_LOCATION = math.tensor([(4., 5), (8., 5), (12., 5), (16., 5)], batch('inflow_loc'), channel('vector'))
                    INFLOW = DOMAIN.scalar_grid(Sphere(center=INFLOW_LOCATION, radius=3)) * 0.6

                    @math.jit_compile
                    def simulate(velocity: StaggeredGrid, smoke: CenteredGrid):
                        for _ in range(3):
                            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
                            buoyancy_force = smoke * (0, 0.5) @ velocity
                            velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
                            velocity, _ = fluid.make_incompressible(velocity)
                        loss = field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10))
                        return loss, smoke, velocity

                    initial_smoke = DOMAIN.scalar_grid(math.zeros(batch(inflow_loc=4)))
                    initial_velocity = DOMAIN.staggered_grid(0) * math.ones(batch(inflow_loc=4))

                    sim_grad = field.functional_gradient(simulate, wrt=[0], get_output=False)

                    for _ in range(2):
                        velocity_grad, = sim_grad(initial_velocity, initial_smoke)
                        initial_velocity = initial_velocity - 0.01 * velocity_grad
