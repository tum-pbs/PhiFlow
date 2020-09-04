# coding=utf-8
import itertools

import numpy as np
from numpy import pi
from phi import math, struct
from phi.geom import AABox
from phi.physics.field import ConstantField, StaggeredGrid

from .field import Field, StaggeredSamplePoints
from .grid import CenteredGrid


def diffuse(field, amount, substeps=1):
    u"""
Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` F with diffusion coefficient α.

If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
Otherwise, finite differencing is used to approximate the
    :param field: CenteredGrid, StaggeredGrid or ConstantField
    :param amount: number of Field, typically α · dt
    :param substeps: number of iterations to use
    :return: Field of same type as `field`
    :rtype: Field
    """
    if isinstance(field, ConstantField):
        return field
    if isinstance(field, StaggeredGrid):
        return struct.map(lambda grid: diffuse(grid, amount, substeps=substeps), field, leaf_condition=lambda x: isinstance(x, CenteredGrid))
    assert isinstance(field, CenteredGrid), "Cannot diffuse field of type '%s'" % type(field)
    if field.extrapolation == 'periodic' and not isinstance(amount, Field):
        fft_laplace = -(2 * pi) ** 2 * field.squared_frequencies
        diffuse_kernel = math.exp(fft_laplace * amount)
        return math.real(math.ifft(field.fft() * math.to_complex(diffuse_kernel)))
    else:
        data = field.data
        if isinstance(amount, Field):
            amount = amount.at(field).data
        else:
            amount = math.batch_align(amount, 0, data)
        for i in range(substeps):
            data += amount / substeps * field.laplace().data
    return field.with_data(data)


def data_bounds(field):
    assert field.has_points
    try:
        data = field.points.data
        min_vec = math.min(data, axis=tuple(range(len(data.shape) - 1)))
        max_vec = math.max(data, axis=tuple(range(len(data.shape) - 1)))
    except StaggeredSamplePoints:
        boxes = [data_bounds(c) for c in field.unstack()]
        min_vec = math.min([b.lower for b in boxes], axis=0)
        max_vec = math.max([b.upper for b in boxes], axis=0)
    return AABox(min_vec, max_vec)


def staggered_curl_2d(grid, pad_width=(1, 2)):
    assert isinstance(grid, CenteredGrid)
    kernel = math.zeros((3, 3, 1, 2))
    kernel[1, :, 0, 0] = [0, 1, -1]  # y-component: - dz/dx
    kernel[:, 1, 0, 1] = [0, -1, 1]  # x-component: dz/dy
    scalar_potential = grid.padded([pad_width, pad_width]).data
    vector_field = math.conv(scalar_potential, kernel, padding='valid')
    return StaggeredGrid(vector_field, box=grid.box)


def extrapolate(input_field, valid_mask, voxel_distance=10):
    """
    Create a signed distance field for the grid, where negative signs are fluid cells and positive signs are empty cells. The fluid surface is located at the points where the interpolated value is zero. Then extrapolate the input field into the air cells.
        :param domain: Domain that can create new Fields
        :param input_field: Field to be extrapolated
        :param valid_mask: One dimensional binary mask indicating where fluid is present
        :param voxel_distance: Optional maximal distance (in number of grid cells) where signed distance should still be calculated / how far should be extrapolated.
        :return: ext_field: a new Field with extrapolated values, s_distance: tensor containing signed distance field, depending only on the valid_mask
    """
    ext_data = input_field.data
    dx = input_field.dx
    if isinstance(input_field, StaggeredGrid):
        ext_data = input_field.staggered_tensor()
        valid_mask = math.pad(valid_mask, [[0, 0]] + [[0, 1]] * input_field.rank + [[0, 0]], "constant")

    dims = range(input_field.rank)
    # Larger than voxel_distance to be safe. It could start extrapolating velocities from outside voxel_distance into the field.
    signs = -1 * (2 * valid_mask - 1)
    s_distance = 2.0 * (voxel_distance + 1) * signs
    surface_mask = create_surface_mask(valid_mask)

    # surface_mask == 1 doesn't output a tensor, just a scalar, but >= works.
    # Initialize the voxel_distance with 0 at the surface
    # Previously initialized with -0.5*dx, i.e. the cell is completely full (center is 0.5*dx inside the fluid surface). For stability and looks this was changed to 0 * dx, i.e. the cell is only half full. This way small changes to the SDF won't directly change neighbouring empty cells to fluid cells.
    s_distance = math.where((surface_mask >= 1), -0.0 * math.ones_like(s_distance), s_distance)

    directions = np.array(list(itertools.product(
        *np.tile((-1, 0, 1), (len(dims), 1))
    )))

    # First make a move in every positive direction (StaggeredGrid velocities there are correct, we want to extrapolate these)
    if isinstance(input_field, StaggeredGrid):
        for d in directions:
            if (d <= 0).all():
                continue

            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple(
                [(slice(1, None) if d[i] == -1 else slice(0, -1) if d[i] == 1 else slice(None)) for i in dims])

            d_field = math.pad(ext_data,
                               [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in
                                           dims] + [[0, 0]], "symmetric")
            d_field = d_field[(slice(None),) + d_slice + (slice(None),)]

            d_dist = math.pad(s_distance,
                              [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in dims] + [
                                  [0, 0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += np.sqrt((dx * d).dot(dx * d)) * signs

            if (d.dot(d) == 1) and (d >= 0).all():
                # Pure axis direction (1,0,0), (0,1,0), (0,0,1)
                updates = (math.abs(d_dist) < math.abs(s_distance)) & (surface_mask <= 0)
                updates_velocity = updates & (signs > 0)
                ext_data = math.where(
                    math.concat([(math.zeros_like(updates_velocity) if d[i] == 1 else updates_velocity) for i in dims],
                                axis=-1), d_field, ext_data)
                s_distance = math.where(updates, d_dist, s_distance)
            else:
                # Mixed axis direction (1,1,0), (1,1,-1), etc.
                continue

    for _ in range(voxel_distance):
        buffered_distance = 1.0 * s_distance  # Create a copy of current voxel_distance. This should not be necessary...
        for d in directions:
            if (d == 0).all():
                continue

            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple(
                [(slice(1, None) if d[i] == -1 else slice(0, -1) if d[i] == 1 else slice(None)) for i in dims])

            d_field = math.pad(ext_data,
                               [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in
                                           dims] + [[0, 0]], "symmetric")
            d_field = d_field[(slice(None),) + d_slice + (slice(None),)]

            d_dist = math.pad(s_distance, [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in dims] + [[0, 0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += np.sqrt((dx * d).dot(dx * d)) * signs

            # We only want to update velocity that is outside of fluid
            updates = (math.abs(d_dist) < math.abs(buffered_distance)) & (surface_mask <= 0)
            updates_velocity = updates & (signs > 0)
            ext_data = math.where(math.concat([updates_velocity] * math.spatial_rank(ext_data), axis=-1), d_field, ext_data)
            buffered_distance = math.where(updates, d_dist, buffered_distance)

        s_distance = buffered_distance

    # Cut off inaccurate values
    distance_limit = -voxel_distance * (2 * valid_mask - 1)
    s_distance = math.where(math.abs(s_distance) < voxel_distance, s_distance, distance_limit)

    if isinstance(input_field, StaggeredGrid):
        ext_field = input_field.with_data(ext_data)
        stagger_slice = tuple([slice(0, -1) for i in dims])
        s_distance = s_distance[(slice(None),) + stagger_slice + (slice(None),)]
    else:
        ext_field = input_field.copied_with(data=ext_data)

    return ext_field, s_distance


def create_surface_mask(liquid_mask):
    """
Computes inner contours of the liquid_mask.
A cell i is flagged 1 if liquid_mask[i] = 1 and it has a non-liquid neighbour.
    :param liquid_mask: binary tensor
    :return: tensor
    """
    # When we create inner contour, we don't want the fluid-wall boundaries to show up as surface, so we should pad with symmetric edge values.
    mask = math.pad(liquid_mask, [[0, 0]] + [[1, 1]] * math.spatial_rank(liquid_mask) + [[0, 0]], "constant")
    dims = range(math.spatial_rank(mask))
    bcs = math.zeros_like(liquid_mask)

    # Move in every possible direction to assure corners are properly set.
    directions = np.array(list(itertools.product(
        *np.tile((-1, 0, 1), (len(dims), 1))
    )))

    for d in directions:
        d_slice = tuple([(slice(2, None) if d[i] == -1 else slice(0, -2) if d[i] == 1 else slice(1, -1)) for i in dims])
        center_slice = tuple([slice(1, -1) for _ in dims])

        # Create inner contour of particles
        bc_d = math.maximum(mask[(slice(None),) + d_slice + (slice(None),)],
                            mask[(slice(None),) + center_slice + (slice(None),)]) - \
            mask[(slice(None),) + d_slice + (slice(None),)]
        bcs = math.maximum(bcs, bc_d)
    return bcs
