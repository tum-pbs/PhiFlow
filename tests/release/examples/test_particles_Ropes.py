from tqdm import trange
from phi.jax.flow import *


def test_ropes():
    grid = CenteredGrid(0, x=20, y=20, bounds=Box(x=1, y=1))
    x = pack_dims(grid.points, 'x,y', instance('nodes'))
    x += math.random_uniform(x.shape) * .01
    fixed_indices = vec(nodes=[399, 19, 199, 239, 0])
    fixed = math.scatter(expand(False, instance(x)), fixed_indices, True)
    plot(PointCloud(x, fixed), size=(4, 3))
    vis.savefig('vis/ropes_setup.jpg')

    deltas = math.pairwise_differences(x, max_distance=grid.dx.mean * 1.1, format='coo')
    distances = math.vec_length(deltas)
    graph = geom.graph(x, distances, {})
    plot(graph)
    vis.savefig('vis/ropes_graph.jpg')

    @jit_compile
    def step(graph: geom.Graph, v, dt=1., gravity=vec(x=0, y=-0.01), relaxation_steps=50):
        v += gravity * dt
        x = graph.center + math.where(fixed, 0, dt * v)
        for _ in range(relaxation_steps):
            deltas = math.pairwise_differences(x, format=graph.edges)
            stick_centers = x + .5 * deltas
            dist = math.norm(deltas)
            stick_directions = deltas / (dist + 1e-5)
            next_x = stick_centers - stick_directions * .5 * graph.edges
            next_x = math.mean(next_x, dual)
            x = math.where(fixed, x, next_x)
        v = (x - graph.center) / dt
        return geom.graph(x, graph.edges), v

    v0 = math.zeros_like(x)
    graph_trj, v_trj = iterate(step, batch(time=50), graph, v0, substeps=2, range=trange)

    show(graph_trj, animate='time')
    vis.savefig('vis/ropes.mp4')


if __name__ == '__main__':
    test_ropes()
