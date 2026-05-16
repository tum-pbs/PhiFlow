from phi.flow import *


def test_variable_boundaries():
    domain = Box(x=10, y=10)
    inflow = CenteredGrid(lambda x, y: vec(x=math.tanh((y-5)/2), y=0), ZERO_GRADIENT, domain, x=1, y=32)

    plot(inflow.as_points(), size=(3, 4))
    vis.savefig('vis/variable_boundaries_inflow.jpg')

    boundary = {'x-': inflow.as_boundary(), 'x+': ZERO_GRADIENT, 'y': 0}

    plot(CenteredGrid(Noise(vector='x,y'), boundary, domain, x=60, y=60).pad(10)['x'])
    vis.savefig('vis/variable_boundaries_padded.jpg')

    def step(v: Field, pressure, dt=1.):
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, 0.01, dt)
        v, pressure = fluid.make_incompressible(v, solve=Solve(x0=pressure))
        return v, pressure

    v0 = StaggeredGrid(0, boundary, domain, x=50, y=32)
    v, p = step(v0, None)
    plot({'velocity': v, 'pressure': p})
    vis.savefig('vis/variable_boundaries_result.jpg')


if __name__ == '__main__':
    test_variable_boundaries()
