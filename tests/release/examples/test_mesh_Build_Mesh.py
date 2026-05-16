from phi.flow import *


def test_build_mesh():
    domain = Box(x=2, y=1)
    sphere_x = math.sin(math.linspace(0, 1.5, batch(time=30))) * .5
    sphere = Sphere(x=sphere_x, y=.5, radius=.3)
    box = Box(x=(1, 3), y=(-1, .5))
    mesh = geom.build_mesh(domain, x=30, y=10, obstacles=union(sphere, box))
    plot(mesh, animate='time')
    vis.savefig('vis/build_mesh.mp4')


if __name__ == '__main__':
    test_build_mesh()
