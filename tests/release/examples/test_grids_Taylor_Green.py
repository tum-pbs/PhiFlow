from tqdm import trange
from phi.jax.flow import *
from functools import partial


def test_taylor_green():
    math.set_global_precision(64)

    domain = dict(x=25, y=25, boundary=PERIODIC, bounds=Box(x=2*PI, y=2*PI))
    viscosity = 0.1

    def taylor_green_velocity(x, t):
        sin_x, sin_y = math.sin(x).vector
        cos_x, cos_y = math.cos(x).vector
        return vec(x=cos_x*sin_y, y=-sin_x*cos_y) * math.exp(-2 * viscosity * t)

    def taylor_green_pressure(x, t):
        return -1 / 4 * (math.sum(math.cos(2 * x), 'vector')) * math.exp(-4 * viscosity * t)

    time = math.linspace(0, 10., batch(time=200))
    analytic_v = StaggeredGrid(partial(taylor_green_velocity, t=time), **domain)
    analytic_p = CenteredGrid(partial(taylor_green_pressure, t=time), **domain)

    plot({"Velocity": analytic_v.time[::4], "Pressure": analytic_p.time[::4]}, animate='time', same_scale=False)
    vis.savefig('vis/taylor_green_analytic.mp4')

    @jit_compile
    def step(velocity, pressure, dt):
        velocity = diffuse.explicit(velocity, viscosity, dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt)
        return fluid.make_incompressible(velocity, (), Solve('CG', 1e-12, 1e-12, x0=pressure))

    dt = time.time[1] - time.time[0]
    v0, p0 = analytic_v.time[0], analytic_p.time[0]
    sim_v, sim_p = iterate(step, time.shape-1, v0, p0, dt=dt, range=trange)

    rmse = math.sqrt(math.mean((analytic_v - sim_v).values**2))
    relative_err = rmse / math.mean(abs(analytic_v.values))
    plot({"RMSE": rmse.time.as_spatial(), "Relative Error": relative_err.time.as_spatial()})
    vis.savefig('vis/taylor_green_error.jpg')


if __name__ == '__main__':
    test_taylor_green()
