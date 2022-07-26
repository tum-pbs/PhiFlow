""" Pendulum Chain
A number of points connected via (invisible) sticks of fixed length.

Set refresh rate to high, pause before resetting.
"""
from phi.flow import *


positions = tensor([(i*0.4, 0) for i in range(10)] + [(5, 0), (4.5, -0.5), (4.5, 0.5)], instance('positions'), channel('vector'))
fixed = tensor([True] + [False] * 12, instance('positions'))
sticks = tensor([(i, i+1) for i in range(9)] + [(9, 11), (9, 12), (10, 11), (10, 12), (11, 12)], instance('sticks, ab'))

id_a, id_b = unstack(sticks, 'ab')
id_ab = concat([id_a, id_b], id_a.shape)
lengths = math.vec_abs(positions[id_b] - positions[id_a])
previous_positions = positions

cloud = PointCloud(Sphere(positions, radius=0.1), bounds=Box(x=(-5, 5), y=(-10, 1)))  # only for viewing
viewer = view(cloud, positions, previous_positions, namespace=globals(), play=False, framerate=20)
for _ in viewer.range():
    moved_positions = positions + (positions - previous_positions)
    moved_positions += (0, -0.02)
    previous_positions = positions
    positions = math.where(fixed, positions, moved_positions)
    for _relaxation_iter in range(10):
        pos_a, pos_b = positions[id_a], positions[id_b]
        stick_centers = 0.5 * (pos_a + pos_b)
        stick_directions = math.vec_normalize(pos_a - pos_b)
        new_pos_a = stick_centers + stick_directions * lengths * 0.5
        new_pos_b = stick_centers - stick_directions * lengths * 0.5
        new_pos = concat([new_pos_a, new_pos_b], new_pos_a.shape.instance)
        new_points = math.scatter(positions, id_ab, new_pos, mode='mean', outside_handling='undefined')
        positions = math.where(fixed, positions, new_points)
    cloud = PointCloud(Sphere(positions, radius=0.1), bounds=Box(x=(-5, 5), y=(-10, 1)))  # only for viewing
