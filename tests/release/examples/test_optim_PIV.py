from phi.jax.flow import *


def test_piv():
    v0 = StaggeredGrid(Noise(batch(seed=2)), x=64, y=64, bounds=Box(x=20, y=20))
    v0, _ = fluid.make_incompressible(v0)
    marker_count = vec(batch('count'), 128, 256, 512, 1024, 2048, 4096)
    initial_markers = v0.bounds.sample_uniform(instance(markers=marker_count))
    plot([Sphere(initial_markers.count['128'], radius=.4), v0], overlay='list', color=['orange', None], size=(5, 3))
    vis.savefig('vis/piv_setup.jpg')

    @jit_compile
    def simulate(v):
        return advect.points(initial_markers, v, dt=.1, integrator=advect.rk4)

    final_markers = simulate(v0)

    with math.SolveTape(record_trajectories=True) as solves:
        fit1 = minimize(lambda x: math.l2_loss(final_markers - simulate(x)), Solve('L-BFGS-B', x0=0 * v0.downsample(4))).at(v0)
        fit2 = minimize(lambda x: math.l2_loss(final_markers - simulate(x+fit1)), Solve('L-BFGS-B', x0=0 * v0))
    v_estimate = fit1 + fit2

    plot((v_estimate - v0).seed[0].curl(), row_dims='seed', size=(10, 2.5))
    vis.savefig('vis/piv_residual.jpg')

    v_mse = math.l2_loss(v_estimate - v0).count.as_instance()
    plot(math.mean(v_mse, 'seed'), err=math.std(v_mse, 'seed'), title="Velocity MSE", log_dims='_', size=(4, 3))
    vis.savefig('vis/piv_mse.jpg')

    plot(solves[0].residual.count.as_channel().trajectory.as_spatial(), size=(5, 3), log_dims='_')
    vis.savefig('vis/piv_convergence.jpg')

    v_trj = solves[0].x.at(v0)
    plot((v_trj - v0).count['4096'].curl(), size=(5, 3), animate='trajectory')
    vis.savefig('vis/piv_optimization.mp4')


if __name__ == '__main__':
    test_piv()
