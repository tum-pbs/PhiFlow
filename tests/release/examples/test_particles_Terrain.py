import os
from tqdm import trange
from phi.torch.flow import *
from PIL import Image


def test_terrain():
    terrain_img = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'particles', 'Terrain003_2K.png')
    if not os.path.exists(terrain_img):
        import pytest
        pytest.skip("Terrain image not found")

    im_frame = Image.open(terrain_img)
    height_2k = tensor(np.array(im_frame), spatial('y,x'))
    height = math.downsample2x(math.downsample2x(height_2k)) / height_2k.max * 50
    bounds = Box(x=100, y=100, z=50)
    terrain = geom.Heightmap(height, bounds, max_dist=.5)
    plot(terrain)
    vis.savefig('vis/terrain_heightmap.jpg')

    x0 = CenteredGrid(0, x=10, y=10, z=1, bounds=Cuboid(vec(x=60, y=30, z=50), x=10, y=10, z=1).corner_representation()).as_points().points
    balls = Sphere(x0, radius=1)
    v0 = math.zeros_like(balls.center)
    plot([terrain, balls], overlay='list', color=[0, 1], plt_params={'z-order': 'as-provided'})
    vis.savefig('vis/terrain_setup.jpg')

    @jit_compile
    def step(balls: Field, dt, elasticity=.4, gravity=vec(x=0, y=0, z=-9.81)):
        v = balls.values + dt * gravity
        dist, _, normal, *_ = terrain.approximate_closest_surface(balls.points)
        bounce = (dist < balls.geometry.bounding_radius()) & (v.vector @ normal < 0)
        impact = -(1+elasticity) * (v.vector @ normal.vector) * normal
        v = math.where(bounce, v + impact, v)
        x = math.clip(balls.points + dt * v, bounds.lower, bounds.upper)
        return balls.shifted_to(x).with_values(v)

    ball_trj = iterate(step, batch(time=50), PointCloud(balls, v0), dt=.1, substeps=2, range=trange)

    plot([terrain, ball_trj.geometry], overlay='list', color=[0, 1], animate='time', plt_params={'z-order': 'as-provided'})
    vis.savefig('vis/terrain.mp4')


if __name__ == '__main__':
    test_terrain()
