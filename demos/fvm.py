import phiml.math
from phi.flow import *
from phi.physics.fvm import diffusion, convection

phiml.math.enable_debug_checks()

mesh = geom.load_su2("mesh_wedge_inv.su2", face_format='csc')
# print(mesh.face_centers)

v = Field(mesh, vec(x=0, y=0), {'inlet': 0, 'upper': ZERO_GRADIENT, 'lower': ZERO_GRADIENT, 'outlet': ZERO_GRADIENT})
# v = v.at_faces(interpolation='linear')
# show(v)
# div = field.divergence(v)
# grad = field.spatial_gradient(v, scheme='green-gauss', stack_dim='~grad')
conv = convection(v, v)
# diff = diffusion(v, grad)
# show(div, grad)

# field.sample(v, v.elements, 'face', v.extrapolation, interpolation='linear')
# v = v.with_values(math.random_uniform(non_channel(v.values)))
# show(v)
print("done")
