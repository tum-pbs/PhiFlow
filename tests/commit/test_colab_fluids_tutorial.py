from unittest import TestCase

from phi.flow import *
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class ColabNotebookTest(TestCase):

    def test_functional_gradient(self):
        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    INFLOW_LOCATION = math.tensor([(4., 5), (8., 5), (12., 5), (16., 5)], batch('inflow_loc'), channel(vector='x,y'))
                    INFLOW = CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=3), extrapolation.BOUNDARY, x=32, y=40) * 0.6

                    def simulate(velocity: StaggeredGrid, smoke: CenteredGrid):
                        for _ in range(3):
                            smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
                            buoyancy_force = smoke * (0, 0.5) @ velocity
                            velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
                            velocity, _ = fluid.make_incompressible(velocity, (), Solve(abs_tol=1e-6))
                        loss = field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10))
                        return loss, smoke, velocity

                    initial_smoke = CenteredGrid(math.zeros(batch(inflow_loc=4)), extrapolation.BOUNDARY, x=32, y=40)
                    initial_velocity = StaggeredGrid(0, 0, x=32, y=40) * math.ones(batch(inflow_loc=4))

                    sim_grad = field.functional_gradient(simulate, wrt=[0], get_output=False)

                    for _ in range(2):
                        velocity_grad, = sim_grad(initial_velocity, initial_smoke)
                        initial_velocity = initial_velocity - 0.01 * velocity_grad

    # def test_functional_gradient_jit(self):
    #     for backend in BACKENDS:
    #         if backend.supports(Backend.jacobian):
    #             with backend:
    #                 INFLOW_LOCATION = math.tensor([(4., 5), (8., 5), (12., 5), (16., 5)], batch('inflow_loc'), channel(vector='x,y'))
    #                 INFLOW = CenteredGrid(Sphere(center=INFLOW_LOCATION, radius=3), extrapolation.BOUNDARY, x=32, y=40) * 0.6
    #
    #                 @math.jit_compile
    #                 def simulate(velocity: StaggeredGrid, smoke: CenteredGrid):
    #                     for _ in range(3):
    #                         smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    #                         buoyancy_force = smoke * (0, 0.5) @ velocity
    #                         velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
    #                         velocity, _ = fluid.make_incompressible(velocity)
    #                     loss = field.l2_loss(diffuse.explicit(smoke - field.stop_gradient(smoke.inflow_loc[-1]), 1, 1, 10))
    #                     return loss, smoke, velocity
    #
    #                 initial_smoke = CenteredGrid(math.zeros(batch(inflow_loc=4)), extrapolation.BOUNDARY, x=32, y=40)
    #                 initial_velocity = StaggeredGrid(0, 0, x=32, y=40) * math.ones(batch(inflow_loc=4))
    #
    #                 sim_grad = field.functional_gradient(simulate, wrt=[0], get_output=False)
    #
    #                 for _ in range(2):
    #                     velocity_grad, = sim_grad(initial_velocity, initial_smoke)
    #                     initial_velocity = initial_velocity - 0.01 * velocity_grad
