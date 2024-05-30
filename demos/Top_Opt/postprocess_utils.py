import numpy as np
import sys
from phi.torch.flow import *

np.set_printoptions(threshold=sys.maxsize)

def write_vtk_darcy3d(u, v ,w, index, cell_locs, additional_args=None, params=None):
    f = open(f'{params["sim_name"]}_{index}.vtk', mode="w")
    
    new_line = "\n"
    f.write("# vtk DataFile Version 2.0\n")
    f.write("TopOpt Flow Data\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")

    f.write(f"DIMENSIONS {u.shape[0]+1} {u.shape[1]+1} {u.shape[2]+1}{new_line}")
    f.write(f"POINTS {(u.shape[0]+1)*(u.shape[1]+1)*(u.shape[2]+1)} float{new_line}")

    for k in range(u.shape[2]+1):
        for j in range(u.shape[1]+1):
            for i in range(u.shape[0]+1):
                f.write(f"{i} {j} {k}{new_line}")

    f.write(f"CELL_DATA {(u.shape[0])*(u.shape[1])*(u.shape[2])}{new_line}")

    f.write("VECTORS actual_vel float\n")

    for k in range(u.shape[2]):
        for j in range(u.shape[1]):
            for i in range(u.shape[0]):
                f.write(f"{u[i,j,k]} {v[i,j,k]} {w[i,j,k]}{new_line}")

    if additional_args!=None:
        X, Y,Z, start_x, start_y, start_z, end_x, end_y, end_z, obs_list, darcy_params = additional_args

        f.write("SCALARS darcy_params float\n")
        f.write("LOOKUP_TABLE default\n")
    
        for k in range(darcy_params.shape[2]):
            for j in range(darcy_params.shape[1]):
                for i in range(darcy_params.shape[0]):
                    f.write(f"{darcy_params[i,j,k]}{new_line}")

    f.write(f"{params}")
    f.close()
    

