import os
import subprocess
from tqdm import trange
from phi.torch.flow import *


def test_fvm_cylinder_gmsh():
    mesh_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'mesh', 'cylinder.msh')
    if not os.path.exists(mesh_path):
        subprocess.run(['wget', 'https://raw.githubusercontent.com/tum-pbs/PhiFlow/master/examples/mesh/cylinder.msh', '-O', mesh_path], check=True)

    mesh = geom.load_gmsh(mesh_path, ('y-', 'x+', 'y+', 'x-', 'cyl+', 'cyl-'))
    plot(Box(x=6, y=6), mesh, overlay='args', size=(4, 3), title='cylinder.msh')
    vis.savefig('vis/fvm_cylinder_gmsh_mesh.jpg')

    @jit_compile_linear
    def momentum_eq(u, u_prev, dt, diffusivity=0.01):
        diffusion_term = dt * diffuse.differential(u, diffusivity, correct_skew=False)
        advection_term = dt * advect.differential(u, u_prev, order=1)
        return u + advection_term + diffusion_term

    @jit_compile
    def implicit_time_step(v, dt):
        v = math.solve_linear(momentum_eq, v, Solve(x0=v), u_prev=v, dt=-dt)
        v, p = fluid.make_incompressible(v, (), Solve('scipy-direct'))
        return v

    boundary = {'x-': vec(x=1, y=0), 'x+': ZERO_GRADIENT, 'y': 0, 'cyl': 0}
    velocity = Field(mesh, tensor(vec(x=0, y=0)), boundary)
    v_trj = math.iterate(implicit_time_step, batch(time=100), velocity, dt=0.001, range=trange)

    plot(v_trj * .1, v_trj.to_grid(), animate='time')
    vis.savefig('vis/fvm_cylinder_gmsh.mp4')


if __name__ == '__main__':
    test_fvm_cylinder_gmsh()
