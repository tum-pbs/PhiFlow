"""Topology Optimization
Fluid flow in a cubic container with inlets and outlets set at random.
After every few time steps non optimal regions are filled with obstacles and optimization simulation is restarted
"""
# from phi.flow import *  # minimal dependencies
import os, argparse, subprocess, pickle, logging, time
from datetime import datetime


parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(  '--topopt_domain_res', default=60, type=int, help='Topology Optimization Domain')
parser.add_argument(  '--convergence_norm', default=1e-4, type=float)
parser.add_argument(  '--network_type', default='None', type=str)
parser.add_argument(  '--t', default=2, type=int)
parser.add_argument(  '--element_type', default='Box', type=str)

pargs = parser.parse_args()
import phi
from phi.torch.flow import *
from geom import create_geometry3d, to_phi_t
from PIL import Image as im
from phi.geom._voxels import Voxels
from phi.geom import UniformGrid

torch.cuda.empty_cache()

torch.set_printoptions(profile="full")

from postprocess_utils import write_vtk_darcy3d

from typing import Union, Tuple, Dict, Any
    
## Simulation Function, Solves fluid flow equations specified transient number of time steps    
def step_darcy3d(v,p,obs_list,darcy_param, total_time_steps, wall_tensor, DOMAIN, BOUNDARY_MASK, VEL_BOUNDARY, num_fluid_cells, params, dt=4):
        
    time_steps = 0

    # No. of boundary/border cells: t
    t = int(params['t'])
    res = int(params['topopt_domain_res'])
    X = res + 2*t
    Y = res + 2*t
    Z = res + 2*t

    start_x, start_y, start_z = t,t,t
    end_x, end_y, end_z = res+t, res+t, res+t 

    qa = 100
    alpha_max = 1e15
    alpha_min = 0
    
    domain_grid = CenteredGrid(0, **DOMAIN)
    
    darcy_params = math.concat([wall_tensor['left'].x[:].y[start_y:end_y].z[start_z:end_z], darcy_param, wall_tensor['right'].x[:].y[start_y:end_y].z[start_z:end_z]], dim=spatial('x'))
    darcy_params = math.concat([wall_tensor['bottom'].x[:].y[:].z[start_z:end_z], darcy_params, wall_tensor['top'].x[:].y[:].z[start_z:end_z]], dim=spatial('y'))
    darcy_params = math.concat([wall_tensor['back'], darcy_params, wall_tensor['front']], dim=spatial('z'))

    darcy_params = CenteredGrid(darcy_params, ZERO_GRADIENT, **DOMAIN)

    norm = 1e10
    while(time_steps < total_time_steps):
        v_last = math.copy(field.sample(v, domain_grid))
        
        alpha_field = alpha_min + (alpha_max - alpha_min) * (1 - darcy_params)/(1 + qa*darcy_params)
        alpha_field = resample(alpha_field, to=v)

        v = v * (1 - BOUNDARY_MASK) + VEL_BOUNDARY
        v = advect.semi_lagrangian(v,v,dt)/(1+alpha_field)
        
        v = diffuse.explicit(v, params['viscosity'], dt, substeps=6)
        
        v,p = fluid.make_incompressible(v, obs_list, Solve('auto', 1e-5, x0=p, max_iterations=100000))
        
        sampled_vel = field.sample(v, domain_grid)
        sampled_vel_x, sampled_vel_y, sampled_vel_z = sampled_vel.vector['x'], sampled_vel.vector['y'], sampled_vel.vector['z']

        norm = math.native( math.sqrt(math.sum( (v_last.vector['x'] - sampled_vel_x)**2 + (v_last.vector['y'] - sampled_vel_y)**2 + (v_last.vector['z'] - sampled_vel_z)**2, dim="x,y,z")/num_fluid_cells) )
        print(f'VNorm = {norm}, Time = {time_steps*dt} , Time_steps = {time_steps}, Num Fluid Cells = {num_fluid_cells}')
        
        time_steps+=1


    write_vtk_darcy3d(u=math.reshaped_native(sampled_vel.vector['x'], groups=['x','y', 'z']), 
                        v=math.reshaped_native(sampled_vel.vector['y'], groups=['x','y', 'z']),
                        w=math.reshaped_native(sampled_vel.vector['z'], groups=['x','y','z']),
                        index=int(params['start_t']), cell_locs=None,  
                        additional_args=(X, Y, Z, start_x, start_y, start_z, end_x, end_y, end_z, [], 
                        math.reshaped_native(field.sample(darcy_params, domain_grid), groups=['x','y', 'z'])), params=params)

    return v,p, darcy_params


def TopOpt(params):
    
    start = datetime.now()
    pressure = None

    obs_list = []
    ## Domain Resolution: res
    res = int(params['topopt_domain_res'])
    ## Wall thickness t
    t = int(params['t'])

    ## Resolution 
    X = res + 2*t 
    Y = res + 2*t
    Z = res + 2*t
    DOMAIN = dict(x=X, y=Y, z=Z, bounds=Box(x=X, y=Y, z=Z))

    domain_grid = CenteredGrid(0, **DOMAIN)
    velocity = StaggeredGrid((0.,0., 0.), ZERO_GRADIENT, **DOMAIN)
    darcy_param = CenteredGrid(1, ZERO_GRADIENT, **DOMAIN)

    
    start_x, start_y, start_z = t,t,t
    end_x, end_y, end_z = res+t ,res+t ,res+t
    darcy_param_torch = torch.ones(end_x-start_x, end_y-start_y, end_z-start_z).to(params['device'])

    num_fluid_cells = (end_x-start_x)*(end_y-start_y)*(end_z-start_z)

    print(params)

    darcy_param = to_phi_t(darcy_param_torch)

    OBS_WALL_GEOMETRY, wall_tensor, INLET_MASK, OUTLET_MASK, BOUNDARY_MASK, VEL_BOUNDARY,wall_obst_idxs, INLET_NORMAL_MASK, OUTLET_NORMAL_MASK = create_geometry3d(params)
    
    def loss_fn(v,p,obs_list, darcy_param, total_time_steps, params):
        v, p, darcy_params = step_darcy3d(v,p,obs_list, darcy_param, total_time_steps, wall_tensor, DOMAIN, BOUNDARY_MASK, VEL_BOUNDARY, num_fluid_cells=num_fluid_cells, params=params)

        loss = math.sum( field.sample( p*(INLET_MASK - OUTLET_MASK) , domain_grid), dim='x,y,z')
        
        return loss, (v,p, darcy_params)

    obs_list.append(union(OBS_WALL_GEOMETRY))
    gradient_fn = math.functional_gradient(loss_fn, 'darcy_param', get_output=True)
    
    while True:
        if params['start_t'] == 0:
            total_time_steps = 100
        else:
           total_time_steps = 50
        
        
        (loss, (velocity,pressure, darcy_params) ), dJ_da = gradient_fn(velocity, pressure, obs_list, darcy_param, total_time_steps, params)
        
        J1 = math.sum(loss, dim='x,y,z')
        
        print('math.sum(velocity*INLET_NORMAL_MASK)')
        print(math.sum(field.sample(velocity*INLET_NORMAL_MASK, domain_grid) , dim='x,y,z,vector'))
        
        print('math.sum(velocity*OUTLET_NORMAL_MASK)')
        print(math.sum(field.sample(velocity*OUTLET_NORMAL_MASK, domain_grid), dim='x,y,z,vector'))
        
        print('math.sum(p*INLET_MASK)')
        print(math.sum(field.sample(pressure*INLET_MASK, domain_grid)))
        
        print('math.sum(p*OUTLET_MASK)')
        print(math.sum(field.sample(pressure*OUTLET_MASK, domain_grid)))

        print(params['start_t'])
        print('J1= ')
        math.print(J1)

        thresh_solid_cells = int(params['tightness'] * res**3)
        dJ_da_torch = math.reshaped_native(dJ_da, groups=['x','y','z'])
        sorted, _ = torch.sort(dJ_da_torch.reshape(-1), descending=True)
        thresh_val = sorted[thresh_solid_cells]
        darcy_param = math.where(dJ_da>=thresh_val, 0, 1)

        top_opt_obst_idxs = math.nonzero(darcy_param==0)
        top_opt_obst_idxs += (start_x, start_y, start_z)

        top_opt_obst_idxs = math.reshaped_native(top_opt_obst_idxs, groups=['nonzero', 'vector', 'x', 'y', 'z']).int()

        top_opt_obst_idxs = top_opt_obst_idxs.squeeze(-1).squeeze(-1).squeeze(-1)
        total_obst_idxs = torch.cat([top_opt_obst_idxs, wall_obst_idxs.to(params['device'])], dim=0)
        
        num_fluid_cells = params['topopt_domain_res']**3 - int(top_opt_obst_idxs.shape[0])

        print(num_fluid_cells)
        
        if params['element_type'] != 'Box':
            obst_grid = torch.zeros(X,Y,Z).to(params['device'])
            obst_grid[total_obst_idxs[:,0],total_obst_idxs[:,1],total_obst_idxs[:,2]] = 1
            OBS_GEOMS = Voxels(UniformGrid(resolution=domain_grid.shape, bounds=Box(x=X, y=Y, z=Z)), to_phi_t(obst_grid.int()))
            obs_list = [Obstacle(OBS_GEOMS)]

        else:
            cell_locs_torch = math.reshaped_native(darcy_param, groups=['x','y','z']).float() 
            
            OBS_GEOMS = []
            for x_val in range(int(darcy_param.shape['x'])):
                for y_val in range(int(darcy_param.shape['y'])):
                    for z_val in range(int(darcy_param.shape['z'])):
                        if cell_locs_torch[x_val, y_val, z_val] == 0:
                            lx, ly, lz = start_x+x_val, start_y+y_val, start_z+z_val
                            OBS_GEOMS.append(Box['x,y,z',lx:lx+1, ly:ly+1, lz:lz+1])
            
            obs_list = [Obstacle(union(OBS_GEOMS + OBS_WALL_GEOMETRY))]

        params['start_t']+=1

        velocity = StaggeredGrid((0.,0., 0.), ZERO_GRADIENT, **DOMAIN)
        pressure = None

        print(f'Time for execution: {datetime.now() - start}')
        gradient_fn.traces.clear()
        gradient_fn.recorded_mappings.clear()
        if params['start_t'] > 10:
           break

if __name__ == '__main__':
    assert backend.default_backend().set_default_device('GPU')
    ## Example sim
    params = {'orifice_locs': [(5, 2), (2, 2), (2, 2)], 
              'orifice_widths': [2.0, 2.0, 2.0], 
              'orifice_types': ['outlet', 'inlet', 'inlet'], 
              'orifice_walls': ['right', 'top', 'back'], 
              'inlet_velocities': [0.0, 1, 1], 
              'topopt_domain_res': 60, 
              'viscosity': 0.2, 
              'convergence_norm': 0.0001, 
              'time_step': 4, 
              'restart_file': None, 
              'start_t': 0, 
              'orifice_locs_on_each_wall': {'left': [], 'bottom': [], 'right': [(5, 2)], 'top': [(2, 2)], 'back': [(2, 2)], 'front': []}, 
              'outlet_wall': 'right', 'device': 'cuda', 
              'sim_name': 'TopOpt_res60_orifices[0_0_1_1_1_0]_outlets[right]_0', 
              't': 5, 'element_type': 'Voxel', 'tightness': 0.75}
    TopOpt(params)
    
