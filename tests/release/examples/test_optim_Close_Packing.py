from phi.jax.flow import *


def test_close_packing():
    R = wrap([1]*64 + [vec(batch('d'), 1, .5)]*64, instance('spheres'))
    size = (math.sum(Sphere(vec(x=0, y=0), R).volume, 'spheres') * 1.05) ** .5
    x0 = math.random_uniform(instance(R), channel(vector='x,y'), high=size)
    plot(Sphere(x0, R), size=(6, 3))
    vis.savefig('vis/close_packing_initial.jpg')

    def loss(x: Tensor, boundary=PERIODIC):
        dx = boundary.shortest_distance(x, rename_dims(x, 'spheres', 'o'), size)
        dr = math.vec_length(dx, eps=1e-8) / (R + rename_dims(R, 'spheres', 'o'))
        return math.l2_loss(math.where((dr < 2e-4) | (dr > 1), 0, 1 - dr))

    x_packed = minimize(loss, Solve('L-BFGS-B', x0=x0)) % size
    plot(Sphere(x_packed, R), size=(6, 3))
    vis.savefig('vis/close_packing_result.jpg')

    with math.SolveTape(record_trajectories=True) as solves:
        minimize(loss, Solve('L-BFGS-B', x0=x0))
    x_trj = solves[0].x % size

    plot(Sphere(x_trj, R), size=(8, 4), animate='trajectory', frame_time=40)
    vis.savefig('vis/close_packing.mp4')


if __name__ == '__main__':
    test_close_packing()
